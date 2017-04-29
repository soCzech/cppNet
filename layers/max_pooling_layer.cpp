#include<memory>
#include<iostream>
#include "max_pooling_layer.hpp"

namespace cppNet {

	max_pooling_layer::max_pooling_layer(std::initializer_list<size_t> input, std::initializer_list<size_t> pool_size) {
		// get the input size and pool size
		std::copy(input.begin(), input.end(), input_);
		std::copy(pool_size.begin(), pool_size.end(), pool_size_);

		// calculate output based on input and pool size
		output_[0] = input_[0] / pool_size_[0];
		if (input_[0] % pool_size_[0] != 0) ++output_[0];
		output_[1] = input_[1] / pool_size_[1];
		if (input_[1] % pool_size_[1] != 0) ++output_[1];

		channels_ = input_[2];
	}

	const VectorF& max_pooling_layer::compute(const VectorF& v, bool training) {
		result_ = v;
		// reshape the input to the 2D representation
		/*
		+-----------+-----------+-- ... --+-----------+
		|           |           |   ...   |           |
		| channel 1 | channel 2 |   ...   | channel n |
		|           |           |   ...   |           |
		+-----------+-----------+-- ... --+-----------+
		*/
		Eigen::Map<MatrixF> reshaped(result_.data(), input_[0], input_[1] * channels_);

		MatrixF matrix(output_[0], output_[1] * channels_);

		for (size_t a = 0, i = 0; a < output_[0]; ++a, i += pool_size_[0]) {
			for (size_t b = 0, j = 0; b < output_[1]; ++b, j += pool_size_[1]) {
				for (size_t channel = 0; channel < channels_; channel++) {
					// max coefficient from the first column
					matrix(a, b * channels_ + channel) = reshaped.block(i, j * channels_ + channel, pool_size_[0], 1).maxCoeff();
					// max coefficient from other columns
					// check the bounds of the image
					for (size_t col = 1; col < pool_size_[1] && (j + col) < input_[1]; col++) {
						float max = reshaped.block(i, (j + col) * channels_ + channel, pool_size_[0], 1).maxCoeff();
						if (max > matrix(a, b * channels_ + channel)) matrix(a, b * channels_ + channel) = max;
					}
				}
			}
		}
		// reshape back to 1D
		result_ = Eigen::Map<VectorF>(matrix.data(), output_[0] * channels_ * output_[1]);
		return result_;
	}

	const VectorF& max_pooling_layer::get_last_result() {
		return result_;
	}

	VectorF max_pooling_layer::backpropagate(const VectorF& prev_result, VectorF delta) {
		// reshape to 2D like in compute
		Eigen::Map<MatrixF> reshaped(delta.data(), output_[0], output_[1] * channels_);

		MatrixF result(input_[0], input_[1] * channels_);

		for (size_t a = 0, i = 0; a < output_[0]; ++a, i += pool_size_[0]) {
			for (size_t b = 0, j = 0; b < output_[1]; ++b, j += pool_size_[1]) {
				for (size_t channel = 0; channel < channels_; channel++) {
					// propagate the gradients over the whole pool
					// check the bounds of the image
					for (size_t row = 0; row < pool_size_[0] && (i + row) < input_[0]; row++) {
						for (size_t col = 0; col < pool_size_[1] && (j + col) < input_[1]; col++) {
							result(i + row, (j+col) * channels_ + channel) = reshaped(a, b * channels_ + channel);
						}
					}
				}
			}
		}
		// reshape back to 1D
		return Eigen::Map<VectorF>(result.data(), input_[0] * channels_ * input_[1]);
	}

}
