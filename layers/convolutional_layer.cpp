#include<memory>
#include<iostream>
#include<initializer_list>
#include<Eigen/Core>
#include "convolutional_layer.hpp"

namespace cppNet {

	convolutional_layer::convolutional_layer(
			std::initializer_list<size_t> input,
			std::initializer_list<size_t> kernel,
			std::initializer_list<size_t> stride,
			size_t channels,
			activation_fn fn,
			activation_fn fn_prime
			): fn_(fn), fn_prime_(fn_prime) {

		std::copy(input.begin(), input.end(), input_);
		std::copy(stride.begin(), stride.end(), stride_);
		std::copy(kernel.begin(), kernel.end(), kernel_);

		output_[0] = (input_[0] - kernel_[0] + 1) / stride_[0];
		if ((input_[0] - kernel_[0] + 1) % stride_[0] != 0) ++output_[0];
		output_[1] = (input_[1] - kernel_[1] + 1) / stride_[1];
		if ((input_[1] - kernel_[1] + 1) % stride_[1] != 0) ++output_[1];

		output_[2] = channels;

		for (size_t i = 0; i < output_[2]; i++) {
			m_.push_back(MatrixF::Random(kernel_[0], input_[2] * kernel_[1]) * 0.2);
			aggreg_nabla_m_.push_back(MatrixF::Zero(kernel_[0], input_[2] * kernel_[1]));
		}

		b_ = VectorF::Random(output_[2]);
		aggreg_nabla_b_ = VectorF::Zero(output_[2]);
	}

	const VectorF& convolutional_layer::compute(const VectorF& v, bool training) {
		result_ = v;
		Eigen::Map<MatrixF> reshaped(result_.data(), input_[0], input_[1] * input_[2]);

		MatrixF result(output_[0], output_[1] * output_[2]);

		for (size_t a = 0, i = 0; a < output_[0]; ++a, i += stride_[0]) {
			for (size_t b = 0, j = 0; b < output_[1]; ++b, j += stride_[1]) {
				for (size_t channel = 0; channel < output_[2]; channel++) {

					result(a, b * output_[2] + channel) = b_(channel);
					// check the bounds of the image
					for (size_t col = 0; col < kernel_[1] && (j + col) < input_[1]; ++col) {
						result(a, b * output_[2] + channel) += reshaped.block(i, (j + col) * input_[2], kernel_[0], input_[2]).cwiseProduct(m_[channel].block(0, col * input_[2], kernel_[0], input_[2])).sum();
					}
				}
			}
		}
		Eigen::Map<VectorF> result_reshaped(result.data(), output_[0] * output_[2] * output_[1]);

		prime_ = fn_prime_(result_reshaped);
		result_ = fn_(result_reshaped);
		return result_;
	}

	const VectorF& convolutional_layer::get_last_result() {
		return result_;
	}

	VectorF convolutional_layer::backpropagate(const VectorF& prev_result, VectorF delta) {
		delta = delta.cwiseProduct(prime_);

		VectorF reshaped(prev_result);
		Eigen::Map<MatrixF> v_reshaped(reshaped.data(), input_[0], input_[1] * input_[2]);


		MatrixF result(output_[0], output_[1] * output_[2]);
		Eigen::Map<MatrixF> delta_re(delta.data(), output_[0], output_[1] * output_[2]);

		for (size_t a = 0, i = 0; a < output_[0]; ++a, i += stride_[0]) {
			for (size_t b = 0, j = 0; b < output_[1]; ++b, j += stride_[1]) {
				for (size_t channel = 0; channel < output_[2]; channel++) {
					aggreg_nabla_b_(channel) = aggreg_nabla_b_(channel) + delta_re(a, b * output_[2] + channel);
					// check the bounds of the image
					for (size_t col = 0; col < kernel_[1] && (j + col) < input_[1]; ++col) {
						aggreg_nabla_m_[channel].block(0, col * input_[2], kernel_[0], input_[2]) += delta_re(a, b * output_[2] + channel) * v_reshaped.block(i, (j + col) * input_[2], kernel_[0], input_[2]);
					}
				}
			}
		}

		v_reshaped.setZero();

		for (size_t a = 0, i = 0; a < output_[0]; ++a, i += stride_[0]) {
			for (size_t b = 0, j = 0; b < output_[1]; ++b, j += stride_[1]) {
				for (size_t channel = 0; channel < output_[2]; channel++) {
					// check the bounds of the image
					for (size_t col = 0; col < kernel_[1] && (j + col) < input_[1]; ++col) {
						v_reshaped.block(i, (j + col) * input_[2], kernel_[0], input_[2])
							+= delta_re(a, b * output_[2] + channel) * m_[channel].block(0, col * input_[2], kernel_[0], input_[2]);
					}
				}
			}
		}

		return Eigen::Map<VectorF>(v_reshaped.data(), input_[0] * input_[1] * input_[2]);
	}

	void convolutional_layer::update_weights(float learning_rate, float l2regularizer, size_t batch_size) {
		for (size_t i = 0; i < output_[2]; i++) {
			m_[i] = (1 - learning_rate * l2regularizer) * m_[i] - (learning_rate / batch_size) / (output_[0] * output_[1]) * aggreg_nabla_m_[i];
			aggreg_nabla_m_[i].setZero();
		}
		b_ -= learning_rate * aggreg_nabla_b_;
		aggreg_nabla_b_.setZero();
	}

	void convolutional_layer::save(std::ostream& os) {
		os << "convolutional_layer\n";
		os << "input: " << input_[0] << " " << input_[1] << " " << input_[2] << "\n";
		os << "output: " << output_[0] << " " << output_[1] << " " << output_[2] << "\n";
		os << "kernel: " << kernel_[0] << " " << kernel_[1] << "\n";
		for (size_t i = 0; i < output_[2]; i++) {
			os << m_[i] << "\n";
		}
		os << b_.transpose() << "\n\n";
	}

	bool convolutional_layer::load(std::istream& is) {
		std::string line;
		size_t input[3], output[3], kernel[2];

		is >> line;
		if (line != "convolutional_layer") return false;

		is >> line;
		if (line != "input:") return false;
		is >> input[0] >> input[1] >> input[2];

		is >> line;
		if (line != "output:") return false;
		is >> output[0] >> output[1] >> output[2];

		is >> line;
		if (line != "kernel:") return false;
		is >> kernel[0] >> kernel[1];

		for (size_t ch = 0; ch < output[2]; ch++) {
			for (size_t i = 0; i < kernel[0]; i++) {
				for (size_t j = 0; j < kernel[1]; j++) {
					if (is.eof()) return false;
					is >> m_[ch](i, j);
				}
			}
		}
		for (size_t i = 0; i < output[2]; i++) {
			if (is.eof()) return false;
			is >> b_(i);
		}

		return true;
	}
}
