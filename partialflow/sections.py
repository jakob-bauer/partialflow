import tensorflow as tf
import re

# TODO: find a way to avoid using name-spaces to determine containing section for ops;
section_from_name = re.compile('(.*/)*graph_section_([0-9]+)/.*')
section_scope_name = 'graph_section_%d'


class _TrackingContext(object):
    """
    This context keeps track of operations added to the graph and values added to graph collections.
    """
    def __init__(self, graph=None):
        """
        Constructor
        :param graph: graph to collect data from. If None, defaults to tf.get_default_graph()
        :return:
        """
        self._graph = tf.get_default_graph() if graph is None else graph

        # data structures for storing ops and collections created in this context
        self._before_collections = {}
        self._before_ops = []
        self._context_collections = {}
        self._context_ops = []


    ####################################################################################################################
    # Context event handlers
    ####################################################################################################################

    def __enter__(self):
        """
        Stores which operations and collection variables have been present before the context
        :return: context object
        """
        for key in self._graph.get_all_collection_keys():
            self._before_collections[key] = tf.get_collection(key)

        self._before_ops = self._graph.get_operations()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Computes which operations and collection variables have been added in the context
        """
        # store which variables have been added to collections in this context
        for key in self._graph.get_all_collection_keys():
            self._context_collections[key] = set([v for v in tf.get_collection(key)
                                                  if key not in self._before_collections
                                                  or v not in self._before_collections[key]])
        del self._before_collections

        # store which ops have been created in this context
        self._context_ops = set([o for o in self._graph.get_operations()
                                 if o not in self._before_ops])
        del self._before_ops


    ####################################################################################################################
    # Methods for accessing the tracked graph elements
    ####################################################################################################################

    def get_collection(self, name):
        """
        Returns a list of values in the collection with the given name.
        Only values added inside the context are returned
        :param name:
        :return:
        """
        if name in self._context_collections:
            return self._context_collections[name]
        else:
            return set()

    def get_ops(self):
        """
        Returns a list of operations that were added in the context
        :return:
        """
        return self._context_ops

    def contains(self, op):
        """
        Returns True iff given operation is part of this context. Does not include gradient ops
        :return:
        """
        return op in self.get_ops()


class GraphSection(_TrackingContext):
    """
    Context that stores information about a section in a graph.
    A graph section is defined as the subgraph that is added to the default graph in the context environment.
    """
    def __init__(self, manager, graph=None):
        """
        Constructor for GraphSection
        :param manager: Instance of GraphSectionManager class. Section is automatically added to manager.
        :param graph: graph to collect data from. If None, defaults to tf.get_default_graph()
        """
        super(GraphSection, self).__init__(graph)

        # lists for incoming tensors, need to be cached for partial forward pass
        self._incoming = set()

        # sets for tensors which need to be cached from the forward/backward passes of this section
        self._tensors_to_cache = set()
        self._temp_tensors_to_cache = set()

        # sets for tensors which need to be fed into the backward passes of this section, EXCLUDING self._incoming
        self._tensors_to_feed = set()
        self._temp_tensors_to_feed = set()

        # which manager this section is part of and which index it has
        self._manager = manager
        self._index = -1

        # operation to run training for this section
        self._training_op = None

        # name scope to keep track of backward pass variables
        self._tf_name_scope = None


    ####################################################################################################################
    # Context event handlers
    ####################################################################################################################

    def __enter__(self):
        """
        Context enter, adds section to manager and transparently opens a tensorflow name scope.
        :return: section object
        """
        # trace graph modifications
        super(GraphSection, self).__enter__()

        # register section in corresponding GraphSectionManager
        self._index = self._manager.add_section(self)

        self._tf_name_scope = tf.name_scope(section_scope_name % self._index)
        self._tf_name_scope.__enter__()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context exit, transparently closes associated name scope, computes incoming tensors.
        """
        # trace graph modifications
        super(GraphSection, self).__exit__(exc_type, exc_val, exc_tb)

        self._tf_name_scope.__exit__(exc_type, exc_val, exc_tb)

        # mark all operations in this context as belonging to this section for later reference
        for op in self.get_ops():
            op.graph_section = self

        # compute incoming tensors
        self._compute_incoming()

    def _compute_incoming(self):
        """
        Helper function that finds the inputs and outputs of the subgraph added in the section
        """
        all_inputs = set([t for o in self.get_ops() for t in o.inputs])
        all_outputs = set([t for o in self.get_ops() for t in o.outputs])

        # we define as incoming tensors all inputs which are not outputs of an operation in the section
        self._incoming = all_inputs.difference(all_outputs)



    ####################################################################################################################
    # Management methods for meta information associated with section
    ####################################################################################################################

    def cleanup_after_cycle(self):
        """
        After a fwd/bwd cycle, clean up the temporary cache and feed tensor sets
        """
        self._temp_tensors_to_cache = set()
        self._temp_tensors_to_feed = set()

    def set_tensors_to_cache(self, tensors, only_next_run=False):
        """
        Sets list of tensors that should be cached during the backward pass of this section
        :param tensors: set or list of tensors
        :param only_next_run: if True, those Tensors are only returned as being cached in the next run
        """
        if only_next_run:
            self._temp_tensors_to_cache = set(tensors)
        else:
            self._tensors_to_cache = set(tensors)

    def add_tensors_to_cache(self, tensors, only_next_run=False):
        """
        Add tensors to list of tensors that should be cached during the backward pass of this section
        :param tensors: set or list of tensors
        :param only_next_run: if True, those Tensors are only returned as being cached in the next run
        """
        if only_next_run:
            self._temp_tensors_to_cache.update(tensors)
        else:
            self._tensors_to_cache.update(tensors)

    def get_tensors_to_cache(self):
        """
        Which tensors should be cached during the next backward pass of this section
        :return: set of tensors to cache
        """
        return self._tensors_to_cache.union(self._temp_tensors_to_cache)

    def set_tensors_to_feed(self, tensors, only_next_run=False):
        """
        Overrides the set of tensors *from other graph sections* that should be fed into the backward pass of
        this section. Automatically drops tensors that are also contained in self._incoming

        :param tensors: list or set of tensors
        :param only_next_run: store as temporary list for next run
        """
        tensors = set(tensors).difference(self._incoming)
        if only_next_run:
            self._temp_tensors_to_feed = tensors
        else:
            self._tensors_to_feed = tensors

    def add_tensors_to_feed(self, tensors, only_next_run=False):
        """
        Adds tensors *from other graph sections* that should be fed into the backward pass of this section
        Automatically drops tensors that are also contained in self._incoming

        :param tensors: list or set of tensors
        :param only_next_run: add to temporary list for next run
        """
        tensors = set(tensors).difference(self._incoming)

        if only_next_run:
            self._temp_tensors_to_feed.update(tensors)
        else:
            self._tensors_to_feed.update(tensors)

    def get_tensors_to_feed(self):
        """
        Which tensors should be fed into the next backward pass of this section
        INCLUDING self._incoming

        :return: set of tensors to be fed
        """
        return self._tensors_to_feed.union(self._incoming).union(self._temp_tensors_to_feed)

    def set_training_op(self, op):
        """
        Stores reference to operation that trains this section
        :param op: training operation
        """
        self._training_op = op

    def get_training_op(self):
        """
        Returns training operation for this section. This might include gradient computation and update operations
        :return: training operation
        """
        return self._training_op

    def get_incoming(self):
        """
        Returns set of incoming tensors used in this section
        :return: set of incoming tensors
        """
        return self._incoming

    def get_index(self):
        """
        Returns the index of this section in its manager's list of sections
        :return: index of section
        """
        return self._index


