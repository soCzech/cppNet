#include<vector>
#include "../core/layer.hpp"

#ifndef CPPNET_convolutional_layer_
#define CPPNET_convolutional_layer_


namespace cppNet {

	class convolutional_layer : public layer {
	public:
		convolutional_layer(std::initializer_list<size_t> input, std::initializer_list<size_t> kernel, std::initializer_list<size_t> stride, size_t channels, activation_fn fn, activation_fn fn_prime);
		virtual void save(std::ostream& os);
		virtual bool load(std::istream& is);
		virtual const VectorF& compute(const VectorF& v, bool training);
		virtual const VectorF& get_last_result();
		virtual VectorF backpropagate(const VectorF& prev_result, VectorF delta);
		virtual void update_weights(float learning_rate, float l2regularizer, size_t batch_size);
	private:
		std::vector<MatrixF> m_;
		std::vector<MatrixF> aggreg_nabla_m_;
		VectorF b_;
		VectorF aggreg_nabla_b_;
		VectorF result_;
		VectorF prime_;
		size_t input_[3];
		size_t stride_[2];
		size_t kernel_[2];
		size_t output_[3];
		activation_fn fn_;
		activation_fn fn_prime_;
	};

}

#endif