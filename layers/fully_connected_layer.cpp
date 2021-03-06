#include<string>
#include<iostream>
#include "fully_connected_layer.hpp"

namespace cppNet {

	fully_connected_layer::fully_connected_layer(size_t input, size_t output, activation_fn fn, activation_fn fn_prime) : fn_(fn), fn_prime_(fn_prime){
		// create matrix and bias vector, initialize with random values
		m_ = 0.2 * MatrixF::Random(output, input);
		b_ = VectorF::Random(output);

		aggreg_nabla_m_ = MatrixF::Zero(output, input);
		aggreg_nabla_b_ = VectorF::Zero(output);
	}

	const VectorF& fully_connected_layer::compute(const VectorF& v, bool training) {
		// compute the result of the layer by matrix multiplication
		result_ = m_*v + b_;
		// compute prime function and activation function
		prime_ = fn_prime_(result_);
		result_ = fn_(result_);
		return result_;
	}

	const VectorF& fully_connected_layer::get_last_result() {
		return result_;
	}

	VectorF fully_connected_layer::backpropagate(const VectorF& prev_result, VectorF delta) {
		// save the gradients for later update
		delta = delta.cwiseProduct(prime_);

		aggreg_nabla_b_ += delta;
		aggreg_nabla_m_ += delta * prev_result.transpose();

		// propagate the delta up the network
		return m_.transpose() * delta;
	}

	void fully_connected_layer::update_weights(float learning_rate, float l2regularizer, size_t batch_size) {
		// update weights and set aggregated gradients back to zero
		m_ = (1 - learning_rate * l2regularizer) * m_ - (learning_rate / batch_size) * aggreg_nabla_m_;
		aggreg_nabla_m_.setZero();
		// update bias
		b_ -= learning_rate * aggreg_nabla_b_;
		aggreg_nabla_b_.setZero();
	}

	void fully_connected_layer::save(std::ostream& os) {
		os << "fully_connected_layer\n";
		os << "input: " << m_.cols() << "\n";
		os << "output: " << m_.rows() << "\n";
		// dump weights and bias
		os << m_ << "\n" << b_.transpose() << "\n\n";
	}

	bool fully_connected_layer::load(std::istream& is) {
		std::string line;
		size_t input, output;

		is >> line;
		if (line != "fully_connected_layer") return false;
		
		// load dimensions
		is >> line;
		if (line != "input:") return false;
		is >> input;

		is >> line;
		if (line != "output:") return false;
		is >> output;

		// load weights
		for (size_t i = 0; i < output; i++) {
			for (size_t j = 0; j < input; j++) {
				if (is.eof()) return false;
				is >> m_(i, j);
			}
		}
		// load bias
		for (size_t i = 0; i < output; i++) {
			if (is.eof()) return false;
			is >> b_(i);
		}

		return true;
	}
}
