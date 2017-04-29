cppNet
======
**cppNet** is small and easy to use library for neural networks. It is based on [Eigen](http://eigen.tuxfamily.org), C++ template library for linear algebra.


The library supports:
- Fully connected layers
- Convolution layers
- Max pooling layers
- Dropout
- SGD - stochastic gradient decent
- Loading and storing trained network
- Live graphs (Python)

Installation
------------
Here are the steps to install **cppNet**.
1. First thing you need are the repository files, get them by typing `git clone https://github.com/soCzech/cppNet.git` into terminal.
2. Run the `download.sh` script. It downloads MNIST dataset (used in the example files) and Eigen library which is needed for the vector operations.
3. You are ready to rock. You can try it by typing `make [all]` and running `fc` or `cn` demo (see example usage in docs.pdf).

___


***To find out how to build your own network, see `test/*test.cpp` files and documentation in docs directory. For technical details, see the source files.***
