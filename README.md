**PartialFlow** is a lightweight extension to [TensorFlow](https://www.tensorflow.org) that simplifies the training of 
large neural networks on graphic cards with limited memory resources. It allows to **trade off** additional 
computation **time** for lower **memory** consumption.

PartialFlow was primarily designed with back-propagation on neural networks in mind. It therefore assumes
that the computation can be decomposed into a cheap forward pass that computes intermediate results and an expensive 
backward pass that computes gradients and updates parameters.

The price for lower memory consumption roughly amounts to a **second forward pass**, plus data transfers between GPU and 
main memory.

**Note that PartialFlow is considered experimental.** Please report bugs in the GitHub issue tracker.

Tested with Tensorflow v0.12.1, not yet tested with Tensorflow v1.0.


## How to use
Please read the [MNIST Example Notebook](MNIST-example.ipynb) for an introduction.

### Subtleties
There are some subtleties when it comes to training a neural network with partial evaluations. Although PartialFlow takes
 care of some of them, some (still) need to be considered during development.

#### Caching
PartialFlow caches the _inputs_ to each graph section in a first forward pass and reuses them during the 
backward pass. It should therefore be ensured that all Tensors for which multiple evaluations may result in different values
are only used as input to a section (and hence cached).

**Problematic Example**: A batch queue is used for loading and processing of input images and labels. If the network's 
loss is defined outside the graph sections, the labels will not be cached during the forward pass, whereas the input 
images will be. In the backward pass the images are reused, but new labels are be drawn from the batch queue. This 
results in inconsistent loss and gradient information.

**Solution**: Define queues outside the graph's sections and all operations on their outputs inside (e.g. first layer, loss). In 
general, do not use Tensors outside of graph sections if their values change across multiple evaluations.

#### Batch Normalization
PartialFlow generates a single training operation for each section, which only computes and applies the gradients for
the corresponding part of the graph. There might be other operations that need to be run during training, e.g. moving 
average updates for batch normalization. PartialFlow groups all operations in the section's `UPDATE_OPS` collection
and runs them exactly once during the section's backward pass.

**Problematic Example**: A network architecture enforces updates of moving averages during the forward pass of the network.
This can e.g. be achieved by explicitly defining the operations as dependencies. Since PartialFlow runs multiple forward
passes over each section, those updates might or might not be executed multiple times.

**Solution**: Use the `UPDATE_OPS` graph collection for operations that need to be run once for each training batch.

## Comparison with basic Training
Take a look at the [Sanity Check Notebook](Sanity-Check.ipynb) for a simple comparison between a network training with 
and without PartialFlow. We compare training progress and update duration.

## How it works
PartialFlow allows to split the graph into multiple *sections* that are trained separately. It automatically 
analyzes the data flow between sections, caches intermediate results as needed, and 
abstracts away the logic of running forward and backward passes over multiple sections.

For a training cycle, PartialFlow first runs a forward pass over the graph and caches each section's 
inputs. It then runs separate backward passes over all sections in reversed order and caches gradient values needed for 
following sections. A section's backward pass may include a second forward pass, as gradient computations often
 require intermediate results computed inside a section.
 
## License
MIT