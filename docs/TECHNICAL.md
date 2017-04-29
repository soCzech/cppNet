cppNet technical info
---------------------

**cppNet** consists of these core files
- *core.hpp* - definitions of data types and definitions of neuron functions
- *layer.hpp* - definition of the layer class, see the file for what a layer needs to implement
- *cost_layer.hpp* - definition of the cost_layer class - the last layer in the network
- *network.hpp* - class containing the layers; can train the network by calling compute, backpropagate and update_weights on each layer
- *summary.hpp* - creates summaries in log files that can be interpreted by the dashboard component

In layers directory, there are implementations of some common layer types used in neural networks.

In test directory, you can find *mnist.hpp/.cpp* file that can be used as a documentation how to create similar loader that suits your dataset.

Also there are two test files you can use as a reference how to build your own network.

___


***More details in docs.pdf and source files.***
