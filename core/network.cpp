/*
Implementation of the network itself.

Internaly there is vector of layers with public access via:
	void add_layer(layer_ptr l);
		adds a layer to the vector
	layer* operator[](size_t i);
		returns a pointer to the ith layer

For training there is SGD implemented:
	void SGD(data_vector& training_data, size_t epochs, size_t batch_size, float learning_rate,
			 float l2_lambda, data_vector& test_data, const std::string& summary_dir);

Dataset can be evaluated by:
	int evaluate_network(data_vector& test_data);
		returns number of correct predictions from th test_data

Single inptut can be evaluated by:
	VectorF feedforward(const VectorF& input);
		returns output vector of the last layer

For load and store operations:
	bool load(const std::string& filename);
		returns if load successful
	void save(const std::string& filename);
*/


#include<memory>
#include<iostream>
#include<algorithm>
#include<random>
#include<fstream>

#include "core.hpp"
#include "layer.hpp"
#include "cost_layer.hpp"
#include "network.hpp"
#include "summary.hpp"

namespace cppNet {

	void network::add_layer(layer_ptr l) {
		layers_.push_back(std::move(l));
	}

	layer* network::operator[](size_t i) {
		if (i >= layers_.size() || i < 0)
			throw cppnet_exception("Incorrect layer number.");
		return layers_[i].get();
	}

	void network::SGD(data_vector& training_data, size_t epochs, size_t batch_size, float learning_rate, float l2_lambda, data_vector& test_data, const std::string& summary_dir) {
		// open summary writer with list of features to treck (! at the end means the values are not smoothed)
		summary_.open(summary_dir, { "loss", "learning_rate!", "accuracy!" });
		
		// random engine for suffling the training_data vector
		std::random_device rd;
		auto engine = std::default_random_engine(rd());

		size_t a = 0;
		// iterate over epochs - each training example is used exactly once
		for (size_t i = 0; i < epochs; ++i) {
			// shuffle the training_data to prevent overfitting
			std::shuffle(std::begin(training_data), std::end(training_data), engine);

			// split the training_data to batches
			for (size_t j = 0; j < training_data.size(); j += batch_size) {
				if (a % 10 == 0) {
					// log loss each 10th batch
					summary_.log(training_data.size() * i + j, "loss", loss_);
				}

				++a;
				loss_ = 0;

				// do the SGD
				SGD_on_batch(training_data, j,
					j + batch_size > training_data.size() ? training_data.size() : j + batch_size,
					learning_rate, l2_lambda / training_data.size());
			}

			summary_.log(training_data.size() * (i + 1), "loss", loss_);
			// after epoch, evaluate the network and add results to the summary file
			int results = evaluate_network(test_data);

			summary_.log(training_data.size() * (i + 1), "learning_rate!", learning_rate);
			summary_.log(training_data.size() * (i + 1), "accuracy!", results / (float)test_data.size());
		}
		// flush the summaries
		summary_.close();
	}

	int network::evaluate_network(data_vector& test_data) {
		// returns number of correct answers given test example and the correct label
		int count = 0;
		VectorF x;
		// iterate over all test examples
		for (size_t i = 0; i < test_data.size(); i++) {
			// get the label distribution from the network
			x = feedforward(test_data[i].input);
			// find top 1 label
			float max = 0; int index = -1;
			for (size_t j = 0; j < x.rows(); j++) {
				if (x[j] > max) {
					max = x[j];
					index = j;
				}
			}
			// find the expected label
			max = 0; int index2 = -1;
			for (size_t j = 0; j < test_data[i].output.rows(); j++) {
				if (test_data[i].output[j] > max) {
					max = test_data[i].output[j];
					index2 = j;
				}
			}
			// if the network's label equals the correct one, increase the number
			// of correctly predicted examples
			if (index == index2) count++;
		}
		return count;
	}

	VectorF network::feedforward(const VectorF& input) {
		// feed the input vector through the network and get the output as a vector
		VectorF result(input);

		for (size_t i = 0; i < layers_.size(); i++) {
			result = layers_[i]->compute(result, false);
		}
		return result;
	}

	void network::SGD_on_batch(data_vector& training_data, size_t start_of_batch, size_t end_of_batch, float learning_rate, float l2regularizer) {
		// do the propagation on the batch
		for (size_t i = start_of_batch; i < end_of_batch; i++) {
			propagation(training_data[i]);
		}

		// update weights of the network based on the accumulated gradient over the batch
		for (size_t i = 0; i < layers_.size() - 1; i++) {
			layers_[i]->update_weights(learning_rate, l2regularizer, end_of_batch - start_of_batch);
		}
	}

	void network::propagation(training_data& data) {
		VectorF result(data.input);
		
		// computes the output of the network
		for (size_t i = 0; i < layers_.size(); i++) {
			result = layers_[i]->compute(result, true);
		}

		// get derivative of the cost function
		result = static_cast<cost_layer*>(layers_[layers_.size() - 1].get())->cost_function(layers_[layers_.size() - 2]->get_last_result(), data.output);
		// get loss for the summaries
		loss_ += static_cast<cost_layer*>(layers_[layers_.size() - 1].get())->get_last_loss(data.output);

		// backpropagate the error up the network
		for (size_t i = layers_.size() - 2; i > 0; i--) {
			result = layers_[i]->backpropagate(layers_[i - 1]->get_last_result(), result);
		}
		// the first layer's input is data.input
		layers_[0]->backpropagate(data.input, result);
	}

	bool network::load(const std::string& filename) {
		// open filename if possible and try to load the network
		// return true if successful, otherwise false
		std::ifstream net(filename);
		
		std::string line;
		net >> line;

		if (line != "__cppNet__") return false;
		// load each layer
		for (size_t i = 0; i < layers_.size(); i++) {
			if (!layers_[i]->load(net)) return false;
		}
		net >> line;
		if (line != "__cppNet__") return false;
		net.close();

		return true;
	}

	void network::save(const std::string& filename) {
		std::cout << "Saving trained model to " << filename << ". Be sure the directory exists." << std::endl;

		// create file to write the network to
		std::ofstream net(filename, std::ios::binary);
		net << "__cppNet__\n";
		// save each layer
		for (size_t i = 0; i < layers_.size(); i++) {
			layers_[i]->save(net);
		}
		net << "__cppNet__\n";
		net.close();
	}
}
