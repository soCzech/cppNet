#include<memory>
#include "../core/cppNet.hpp"

int main() {

	cppNet::mnist_loader i("dataset/mnist/train-images-idx3-ubyte", "dataset/mnist/train-labels-idx1-ubyte");
	cppNet::data_vector training_data;
	i.get_data(training_data, 60000);

	cppNet::mnist_loader j("dataset/mnist/t10k-images-idx3-ubyte", "dataset/mnist/t10k-labels-idx1-ubyte");
	cppNet::data_vector test_data;
	j.get_data(test_data, 10000);

	cppNet::network net;
	net.add_layer(
		std::unique_ptr<cppNet::fully_connected_layer>(
			new cppNet::fully_connected_layer(784, 100, cppNet::relu_function, cppNet::relu_prime_function)
	));
	net.add_layer(
		std::unique_ptr<cppNet::dropout_layer>(
			new cppNet::dropout_layer(100, 0.8)
	));
	net.add_layer(
		std::unique_ptr<cppNet::fully_connected_layer>(
			new cppNet::fully_connected_layer(100, 10, cppNet::identity_function, cppNet::identity_prime_function)
	));
	net.add_layer(
		std::unique_ptr<cppNet::cross_entropy_cost_layer>(
			new cppNet::cross_entropy_cost_layer(cppNet::softmax_function)
	));

	net.load("bin/models/fc.ckpt");

	net.SGD(training_data, 10, 50, 0.1, 0.1, test_data, "bin/logs/fc");

	net.save("bin/models/fc.ckpt");

	return 0;
}
