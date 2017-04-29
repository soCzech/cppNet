#include<memory>
#include<iostream>

#include "dropout_layer.hpp"

namespace cppNet {

	dropout_layer::dropout_layer(size_t input, float probability) {
		input_ = input;
		probability_ = probability;
	}

	const VectorF& dropout_layer::compute(const VectorF& v, bool training) {
		std::random_device rd_;
		std::mt19937 gen_(rd_());

		if (training) {
			// when training generate mask based on the probability
			std::discrete_distribution<> dis({ 1 - probability_, probability_ });

			mask_ = v;
			for (size_t i = 0; i < mask_.size(); i++) {
				mask_[i] = dis(gen_);
			}
			// apply mask to the input vector and multiply it by (1/probability)
			// to keep the sum of the output similar to the input
			result_ = (1/probability_) * mask_.cwiseProduct(v);
		} else {
			// if not training, skip the layer
			result_ = v;
		}
		return result_;
	}

	const VectorF& dropout_layer::get_last_result() {
		return result_;
	}

	VectorF dropout_layer::backpropagate(const VectorF& prev_result, VectorF delta) {
		// there are no parameter so update, just cancel the gradients to the turned off neurons
		return mask_.cwiseProduct(delta);
	}

}
