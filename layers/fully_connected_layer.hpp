#include<iostream>
#include "../core/layer.hpp"

#ifndef CPPNET_fully_connected_layer_
#define CPPNET_fully_connected_layer_


namespace cppNet {

	class fully_connected_layer : public layer {
	public:
		fully_connected_layer(size_t input, size_t output, activation_fn fn, activation_fn fn_prime);
		virtual void save(std::ostream& os);
		virtual bool load(std::istream& is);
		virtual const VectorF& compute(const VectorF& v, bool training);
		virtual const VectorF& get_last_result();

		virtual VectorF backpropagate(const VectorF& prev_result, VectorF delta);
		virtual void update_weights(float learning_rate, float l2regularizer, size_t batch_size);
	private:
		MatrixF m_;
		MatrixF aggreg_nabla_m_;
		VectorF b_;
		VectorF aggreg_nabla_b_;
		VectorF result_;
		VectorF prime_;
		activation_fn fn_;
		activation_fn fn_prime_;
	};

}

#endif