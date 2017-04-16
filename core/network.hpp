#include<vector>
#include<string>
#include "core.hpp"
#include "layer.hpp"
#include "summary.hpp"

#ifndef CPPNET_network_
#define CPPNET_network_

namespace cppNet {

	class network {
	public:
		network() : number_of_layers_added_(0), layers_size_(0) {}
		void add_layer(layer_ptr l);
		bool load(const std::string& filename);
		void save(const std::string& filename);
		layer* operator[](size_t i);
		void SGD(data_vector& training_data, size_t epochs, size_t batch_size, float learning_rate, float l2_lambda, data_vector& test_data, const std::string& summary_dir);
		int evaluate_network(data_vector& test_data);
		VectorF feedforward(const VectorF& input);
	private:
		void SGD_on_batch(data_vector& training_data, size_t start_of_batch, size_t end_of_batch, float learning_rate, float l2regularizer);
		void propagation(training_data& data);

		std::vector<layer_ptr> layers_;
		size_t layers_size_;
		size_t number_of_layers_added_;
		float loss_;
		summary summary_;
	};

}

#endif
