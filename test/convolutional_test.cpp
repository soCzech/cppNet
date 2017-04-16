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
		std::unique_ptr<cppNet::convolutional_layer>(
			new cppNet::convolutional_layer({ 28,28,1 }, { 5,5 }, { 2,2 }, 20, cppNet::relu_function, cppNet::relu_prime_function)
	));
	net.add_layer(
		std::unique_ptr<cppNet::max_pooling_layer>(
			new cppNet::max_pooling_layer({ 12,12,20 }, {2,2})
	));
	net.add_layer(
		std::unique_ptr<cppNet::fully_connected_layer>(
			new cppNet::fully_connected_layer(6*6*20, 100, cppNet::relu_function, cppNet::relu_prime_function)
	));
	net.add_layer(
		std::unique_ptr<cppNet::fully_connected_layer>(
			new cppNet::fully_connected_layer(100, 10, cppNet::identity_function, cppNet::identity_prime_function)
	));
	net.add_layer(
		std::unique_ptr<cppNet::cross_entropy_cost_layer>(
			new cppNet::cross_entropy_cost_layer(cppNet::softmax_function)
	));

	net.load("bin/models/cn.ckpt");

	net.SGD(training_data, 3, 50, 0.03, 0.1, test_data, "bin/logs/cn");

	net.save("bin/models/cn.ckpt");

	return 0;
}
