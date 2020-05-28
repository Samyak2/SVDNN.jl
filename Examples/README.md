# Examples of SVD-DNN

## XOR

See [XOR.jl](XOR.jl) for a plain text version and [XOR-SVD.ipynb](XOR-SVD.ipynb) for a jupyter notebook version.

This is very basic example made only to test the implementation, it does not show any performance improvements since there are only 4 training examples and the model is quite small.

## Cats vs Non-cat Classification

See [Cats-SVD.ipynb](Cats-SVD.ipynb).

An example with the [cats vs non-cat dataset](https://www.floydhub.com/deeplearningai/datasets/cat-vs-noncat) which is again a toy dataset but much bigger than the XOR one.

The performance improvements can be seen here, although not by a lot.

## MNIST Digit Classification

See [MNIST-SVD.ipynb](MNIST-SVD.ipynb).

This is the most complete example, showing all the (extra) hyperparameters and their effects. Use this as a base if you're going to use it in a real project.
