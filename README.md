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
Check out the [MNIST Example Notebook](MNIST-example.ipynb) for an introduction.

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
 requires intermediate results computed inside a section.
 
## License
MIT