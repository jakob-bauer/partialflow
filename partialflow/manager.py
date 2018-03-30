import tensorflow as tf
from .sections import GraphSection, section_from_name
from tensorflow.python.client.session import _FetchMapper
from .utils import VerboseTimer
from contextlib import ExitStack


def _flatten_list(nested_list):
    """
    Given a nested list, returns flat list of all its elements

    :param nested_list: nested list
    :return: flat list
    """
    if not isinstance(nested_list, list):
        return [nested_list]

    res = []
    for sub_list in nested_list:
        res += _flatten_list(sub_list)

    return res


class GraphSectionManager(object):
    """
    Optimizes a graph with GraphSections by running partial backward passes.
    This reduces the memory consumption at the expense of an additional forward pass and multiple data transfers
    between GPU and main memory.
    """
    def __init__(self, graph=None):
        """
        Constructor
        :param graph: graph that is split into sections, defaults to tf.get_default_graph()
        """
        self._graph = tf.get_default_graph() if graph is None else graph

        self._sections = []
        self._cache = {}

        # which tensors to cache during full forward pass
        self._tensors_to_cache_in_fwd = []

        # caches for request op information
        self._req_input_tensors = {}  # requested op -> [list of input tensors]
        self._req_eval_sections = {}  # requested op -> section to evaluate in (or None for forward pass)
        self._req_reduced_inputs = {}  # requested op -> [list of input tensors minus those from eval_section]


    ####################################################################################################################
    # Section handling
    ####################################################################################################################

    def new_section(self):
        """
        Constructs a new graph section and adds it to the manager.

        :return: newly created section
        """
        section = GraphSection(graph=self._graph, manager=self)
        return section

    def add_section(self, section):
        """
        Adds a GraphSection to the list of managed sections.

        :param section: instance of GraphSection
        :return: index of added section
        """
        self._sections.append(section)
        return len(self._sections) - 1

    def get_sections(self):
        """
        Returns a list of the sections
        Sections are ordered in the way there were added.

        :return: list of sections
        """
        return self._sections


    ####################################################################################################################
    # Methods to prepare the graph for training
    ####################################################################################################################

    def add_training_ops(self, optimizer, loss=None, var_list=None, grads=None, global_step=None, verbose=False,
                         summaries=None):
        """
        Constructs a training operation for each section. If `grads` is not given, it is computed by
            grads = optimizer.compute_gradients(loss, var_list)

        Each section's training operations applies the gradients of the section's variables and runs all operations in
        the section's GraphKeys.UPDATE_OPS collection. All variables are assumed to be contained in the section's
        GLOBAL_VARIABLES collection.

        :param optimizer: tensorflow optimizer to use
        :param loss: loss tensor to optimize
        :param var_list: variable list to compute gradients on
        :param grads: gradients as returned by optimizer.compute_gradients, alternative to loss and var_list
        :param global_step: global step tensor to increment after full backward pass
        :param verbose: if True, adds tf.Print operations to log backward passes over sections
        :param summaries: optional list of collections to add gradient histogram summaries to. Defaults to None
        """

        # add gradient computation nodes for all trainable variables
        if grads is None:
            assert loss is not None, 'Either gradients or loss have to be given.'
            grads = optimizer.compute_gradients(loss, var_list=var_list)

        # store references to gradient ops for simple access
        grad_dict = {v: g for g, v in grads}

        for s, section in reversed(list(enumerate(self.get_sections()))):
            # construct gradient application
            xs = section.get_collection(tf.GraphKeys.GLOBAL_VARIABLES).intersection(grad_dict.keys())
            apply_op = None
            if len(xs) > 0:
                cur_grads = [(grad_dict[v], v) for v in xs]

                if summaries is not None:
                    for v in xs:
                        tf.summary.histogram('gradients/%s' + v.name, grad_dict[v], collections=summaries)

                # if we should be verbose, log backward passes
                if verbose:
                    cur_grads[0] = (tf.Print(cur_grads[0][0], [cur_grads[0][0]],
                                             'Running backward pass on section %d' % s), cur_grads[0][1])

                # only increment global step in last partial backward pass
                apply_op = optimizer.apply_gradients(cur_grads, global_step=global_step if s == 0 else None)

                print("Found %d gradient application operations in section %d. Adding to training op."
                      % (len(cur_grads), s))

            # no gradients to apply
            else:
                print("Section %d does not contain gradient application operations." % s)

                # in last section's backward pass -> need to increment global step separately
                if s == 0 and global_step is not None:
                    apply_op = tf.assign_add(global_step, 1)


            # group update operations
            update_op = None
            update_ops = section.get_collection(tf.GraphKeys.UPDATE_OPS)
            if len(update_ops) > 0:
                print("Found %d update operations in section %d. Adding to training op." % (len(update_ops), s))
                update_op = tf.group(*update_ops)
            else:
                print("Section %d does not contain update operations." % s)

            # construct final training operation
            if apply_op is not None and update_op is not None:
                train_op = tf.group(apply_op, update_op)
            elif apply_op is not None:
                train_op = apply_op
            elif update_op is not None:
                train_op = update_op
            else:
                train_op = tf.no_op()

            section.set_training_op(train_op)

    def prepare_training(self):
        """
        Prepares the partial training by computing metadata about sections and creating training operations.
        Should be run after the full construction of the graph, including training ops.
        """
        assert len(self.get_sections()) > 0, 'There has to be at least one GraphSection in this graph.'

        # compute list of unique tensors to cache in forward pass
        self._tensors_to_cache_in_fwd = list(set([t for s in self.get_sections() for t in s.get_incoming()]))

        # compute tensors to feed into backward passes
        all_tensors_to_feed = set()
        for s, section in reversed(list(enumerate(self.get_sections()))):
            # find list of tensors to feed into training operation
            tensors_to_feed = self._find_feeds_from_other_sections(section.get_training_op(),
                                                                   ignore=[section], given=section.get_incoming())

            # store info in section
            section.set_tensors_to_feed([t[0] for t in tensors_to_feed])
            all_tensors_to_feed.update(tensors_to_feed)

        # tell sections to cache those tensors needed by other sections
        self._mark_tensors_for_caching(all_tensors_to_feed, only_next_run=False)

    def _mark_tensors_for_caching(self, tensors, only_next_run=False):
        """
        Stores which tensors to cache in each section's backward pass.

        :param tensors: list of tuples as output by _find_feeds_from_other_sections
        :param only_next_run: if True, only store these temporarily for the next run
        """
        for s, section in list(enumerate(self.get_sections())):
            to_cache = [t[0] for t in tensors if t[1] == section]
            section.add_tensors_to_cache(to_cache, only_next_run)


    ####################################################################################################################
    # Methods for running the graph forward or backward, full or section-wise
    ####################################################################################################################

    def run_forward(self, sess, fetches=None, basic_feed=None):
        """
        Runs a forward pass over all sections, clears cache, and caches intermediate results for backward passes.
        Should not be used to request tensors for which gradients need to be computed! Otherwise potential OOM

        :param sess: session to run in
        :param fetches: list of tensors to fetch during forward pass, e.g. loss
        :param basic_feed:
        :return results for fetched tensors
        """
        if fetches is None:
            fetches = []

        if basic_feed is None:
            basic_feed = {}

        cache_values, results = sess.run([self._tensors_to_cache_in_fwd, fetches], basic_feed)

        # clear cache and store intermediate results of forward pass
        self._cache = basic_feed.copy()
        for v, k in zip(cache_values, self._tensors_to_cache_in_fwd):
            self._cache[k] = v

        return results

    def run_backward(self, sess, fetches=None, verbose_timing=False):
        """
        Runs section-wise training pass over the graph, caches intermediate results as defined by sections.

        :param sess: session to run in
        :param fetches: list of fetches for each section.
                        Tensors in i-th sub-list are fetched in backward pass of i-th section
        :param verbose_timing: if True, time forward and backward passes verbosely
        :return: list of results, same structure as `fetches`
        """
        results = []
        if fetches is None:
            fetches = [[] for _ in self.get_sections()]

        timer = VerboseTimer if verbose_timing else lambda _: ExitStack()

        for s, section in reversed(list(enumerate(self.get_sections()))):
            # cache intermediate results to be used in other sections
            tensors_to_cache = list(section.get_tensors_to_cache())

            # construct feed dictionary
            feed = {}
            for t in list(section.get_tensors_to_feed()):
                feed[t] = self._cache[t]

            request = fetches[s] if len(fetches) > s else []
            tensors_to_compute = [section.get_training_op(), tensors_to_cache, request]

            with timer('backward on section %d' % s):
                _, cache_vals, result = sess.run(tensors_to_compute, feed)

            results.append(result)

            # store all computed values in cache
            for i in range(len(tensors_to_cache)):
                self._cache[tensors_to_cache[i]] = cache_vals[i]

        results.reverse()

        return results

    def run_full_cycle(self, sess, fetches=None, basic_feed=None, verbose_timing=False):
        """
        Runs forward and backward pass through the graph and fetches results, similar to session.run().
        Mimics tensorflow's session.run for structure of fetches and returned values.

        :param sess: session to run in
        :param fetches: arbitrarily nested structure of graph elements to fetch
        :param basic_feed: dictionary of tensors/placeholders and values to feed into the graph
        :param verbose_timing: if True, time forward and backward passes verbosely
        :return: resulting values for fetches
        """
        if fetches is None:
            fetches = []

        if basic_feed is None:
            basic_feed = {}

        timer = VerboseTimer if verbose_timing else lambda _: ExitStack()

        # multiple GraphSections -> train step-wise
        if len(self.get_sections()) > 1:
            # for all requested tensors, find sections in which they are computed
            with timer('split fetches'):
                unique_fetches, fwd_requests, bwd_requests, fetch_mapper = self._split_requests_for_sections(fetches)

            # run cycle
            with timer('forward'):
                fwd_values = self.run_forward(sess, fwd_requests, basic_feed=basic_feed)

            with timer('backward'):
                bwd_values = self.run_backward(sess, bwd_requests, verbose_timing=verbose_timing)

            with timer('post cycle'):
                # reconstruct output
                flat_requests = _flatten_list([fwd_requests, bwd_requests])
                flat_values = _flatten_list([fwd_values, bwd_values])
                req_val = list(zip(flat_requests, flat_values))
                values = [e[1] for fetch in unique_fetches for e in req_val if e[0] == fetch]
                results = fetch_mapper.build_results(values)

                # clean intermediate cache fetches
                for section in self.get_sections():
                    section.cleanup_after_cycle()

        # only a single GraphSection (no real splits) -> fall back to default training in one go
        else:
            train_op = self.get_sections()[0].get_training_op()
            _, results = sess.run([train_op, fetches], basic_feed)

        return results

    def _split_requests_for_sections(self, fetches):
        """
        Internal helper function that assigns each fetch to a specific part of the graph evaluation.

        :param fetches: arbitrarily nested structure of graph elements
        :return: list of unique fetches, list of requests for forward pass,
                 list ob sub-lists with requests for backward passes, fetch_mapper to reconstruct result
        """
        # TODO: avoid using a tf core class here
        fetch_mapper = _FetchMapper.for_fetch(fetches)
        unique_fetches = fetch_mapper.unique_fetches()
        sections = []
        all_input_tensors = set()
        forward_fetches = []
        backward_fetches = [[] for _ in self.get_sections()]

        for fetch in unique_fetches:
            fetch_op = fetch if isinstance(fetch, tf.Operation) else fetch.op
            section = self._get_op_section(fetch_op)
            sections.append(section)

            # check cache for pre-computed input tensors
            if fetch_op in self._req_input_tensors:
                input_tensors = self._req_input_tensors[fetch_op]

            # not cached, compute and store
            else:
                input_tensors = self._find_feeds_from_other_sections(fetch_op)
                self._req_input_tensors[fetch_op] = input_tensors

            # fetch independent from all sections? -> evaluate in first general forward pass
            if section is None and len(input_tensors) == 0:
                forward_fetches.append(fetch)
                # print('will evaluate ', fetch, 'in forward pass')

            # fetch depends on at least one section
            else:
                # not yet cached
                if fetch_op not in self._req_eval_sections:
                    # fetch is part of a section? -> evaluate in backward pass of this section (includes forward pass)
                    if section is not None:
                        eval_section = section

                    # fetch is not part of any section
                    elif section is None:
                        # select the last section this fetch depends on, in order of backward pass
                        eval_section = min([a for t, a in input_tensors], key=lambda a: a.get_index())

                        # remove input tensors inside this section, since they are evaluated anyway
                        input_tensors = self._find_feeds_from_other_sections(fetch_op, ignore=[eval_section])

                    # print('will evaluate ', fetch, 'in backward pass of section %d' % eval_section.get_index())

                    # cache infos
                    self._req_eval_sections[fetch_op] = eval_section
                    self._req_reduced_inputs[fetch_op] = input_tensors

                # load data from cache
                else:
                    eval_section = self._req_eval_sections[fetch_op]
                    input_tensors = self._req_reduced_inputs[fetch_op]

                # fetch from backward pass of determined section
                backward_fetches[eval_section.get_index()].append(fetch)

                # needs input from other sections
                if len(input_tensors) > 0:
                    eval_section.add_tensors_to_feed([t[0] for t in input_tensors], only_next_run=True)
                    all_input_tensors.update(input_tensors)

        # store which tensors need to be cached
        # print('will cache', all_input_tensors)
        self._mark_tensors_for_caching(all_input_tensors, only_next_run=True)

        return unique_fetches, forward_fetches, backward_fetches, fetch_mapper


    ####################################################################################################################
    # Methods for analyzing the graph
    ####################################################################################################################

    def _get_op_section(self, op):
        """
        Returns the section which a given op resides in

        :param op: tf.Operation in question
        :return: GraphSection object or None if not in a section
        """
        if hasattr(op, 'graph_section'):
            return op.graph_section

        # backward pass ops are tracked using name scopes
        m = section_from_name.match(op.name)
        if not m:
            return None
        else:
            return self.get_sections()[int(m.group(2))]

    def _find_feeds_from_other_sections(self, op_or_t, ignore=None, given=None):
        """
        Traverses the graph to find tensors that need to be fed into an evaluation of a given operation or tensor.
        Only considers tensors that are computed by operations outside of `op_or_t`'s section (if any) and
        outside of the `ignore`d sections. Assumes that operations outside all sections are not cache-able.

        Example:
            We want to know which tensors are needed by a training operation of section S. Because
            the training includes a forward and backward pass of S anyway, we can ignore all tensors computed in S
            and only need to consider inputs from other sections. Since the training operation is typically not
            contained in the section though, one needs to specify which section to ignore.
            In a simple feed-forward network, this method would then return the output tensors of the predecessor
            section and the gradient tensors of the successor section.

        Since gradient information is often aggregated outside of sections, caching the immediate (non-aggregated)
        outputs of a section may be suboptimal. The method therefore enriches the graph by permanently merging such
        aggregation operations into the corresponding section if possible. Note that this is a heuristic based on the
        assumption that the output of an operation is usually smaller than the set of its inputs. It is therefore not
        guaranteed to always yield optimal results, i.e. smallest cache volumes.

        :param op_or_t: operation or tensor to evaluate
        :param ignore: list of sections whose operations are assumed to be non-cache-able, i.e. are computed anyway
        :param given: list of tensors that are assumed to be given to be fed in the evaluation
        :return: list of (tensor, section) tuples:
            - tensor is the tensor to be cached
            - section is the GraphSection whose backward pass it can be cached from, or -1 if tensor is `given`
        """
        def _remove_tensors_by_section(tensors, section):
            return filter(lambda t: t[1] != section, tensors)

        def _clear_graph_info():
            for op in self._graph.get_operations():
                if hasattr(op, 'section_deps'):
                    del op.section_deps

        # buffer initialization
        if given is None:
            given = set()
        if ignore is None:
            ignore = []
        ignore += [None]  # ops outside of sections are never cached

        def _recursion_helper(op_or_t):
            """
            This method enriches the graph with temporary information in the `section_deps` attribute of operations.
            The graph should be cleaned using _clean_graph_info() before running again, otherwise the results
            might be incorrect.

            :param op_or_t: current operation or tensor in question
            :return: tuple of two lists. First list is in same format as parent method output, second list contains all
                     sections the op_or_t depends on (projection of first list to second argument in each tuple)
            """
            # if tensor is given, convert to corresponding op
            op = op_or_t.op if isinstance(op_or_t, tf.Tensor) else op_or_t

            origin_section = self._get_op_section(op)
            depends_on_tensors = set()
            depends_on_sections = set()

            # we have already been here before -> just return cached results
            if hasattr(op, 'section_deps') and set(op.section_deps[1]).isdisjoint(ignore):
                return op.section_deps

            # mark this node as visited to avoid loops
            op.section_deps = ([], [])

            # check all the input tensors to this op
            for t in op.inputs:
                is_leaf = False
                input_dep_tensors = set()
                input_dep_sections = set()

                # which section does the current tensor belong to?
                cur_section = self._get_op_section(t.op)

                # this is a variable? -> no dependency
                if t.op.op_def is not None and t.op.op_def.name == 'Variable':
                    continue

                # this tensor is given -> add to dependencies and continue to next
                if t in given:
                    is_leaf = True
                    cur_section = -1  # section dependency unclear since expected to be given

                # this tensor is computed in a different section that is not ignored
                # i.e., we are at a leaf -> add to results and stop recursion
                elif cur_section != origin_section and cur_section not in ignore:
                    is_leaf = True

                # this tensor is computed in same section or in ignored section -> recursion
                else:
                    # compute all dependencies for the current input tensor
                    input_dep_tensors, input_dep_sections = _recursion_helper(t.op)

                    # we do in no case depend on our own section (since this is assumed anyway)
                    input_dep_tensors = _remove_tensors_by_section(input_dep_tensors, origin_section)
                    if origin_section in input_dep_sections:
                        input_dep_sections.remove(origin_section)

                    # if all dependencies belong to the same section and the current op is outside all sections,
                    # we might be able to aggregate the inputs and only cache the output tensor
                    if len(input_dep_sections) == 1 and cur_section is None \
                            and -1 not in input_dep_sections:  # do not merge dependencies on given inputs

                        # check if we are at the border of some ignored section
                        # in this case, we may not count our current tensor as belonging to the section it depends on,
                        # since we traversed some ignored section on the way.
                        immediate_input_sections = set([self._get_op_section(o)
                                                        for o in list(t.op.inputs) + t.op.control_inputs])
                        traversed_other_section = not immediate_input_sections.issubset(input_dep_sections)

                        # we can merge this tensor's op into its input section
                        if not traversed_other_section:
                            cur_section = list(input_dep_sections)[0]
                            t.op.graph_section = cur_section

                            # ignore all other ops in the tensor's subtree
                            is_leaf = True

                # this tensor is a leaf (or should be handled as one)
                # add to dependencies for current op
                if is_leaf:
                    depends_on_tensors.add((t, cur_section))
                    depends_on_sections.add(cur_section)

                # other dependencies (i.e. either multiple sections or None)
                # -> we inherit those
                else:
                    depends_on_tensors.update(input_dep_tensors)
                    depends_on_sections.update(input_dep_sections)

            # recursion on control inputs
            for dep in op.control_inputs:
                input_dep_tensors, input_dep_sections = _recursion_helper(dep)

                input_dep_tensors = _remove_tensors_by_section(input_dep_tensors, origin_section)
                if origin_section in input_dep_sections:
                    input_dep_sections.remove(origin_section)

                depends_on_tensors.update(input_dep_tensors)
                depends_on_sections.update(input_dep_sections)

            # store which tensors this op depends on
            op.section_deps = (list(depends_on_tensors), list(depends_on_sections))

            return op.section_deps

        # compute result and clear temporary graph information afterwards
        result = _recursion_helper(op_or_t)
        _clear_graph_info()

        return result[0]
