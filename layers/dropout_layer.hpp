#include<random>

#include "../core/layer.hpp"

#ifndef CPPNET_dropout_layer_
#define CPPNET_dropout_layer_


namespace cppNet {

	class dropout_layer : public layer {
	public:
		dropout_layer(size_t input, float probability);
		virtual const VectorF& compute(const VectorF& v, bool training);
		virtual const VectorF& get_last_result();
		virtual VectorF backpropagate(const VectorF& prev_result, VectorF delta);
		virtual void update_weights(float learning_rate, float l2regularizer, size_t batch_size) {}
		virtual void save(std::ostream& os) {}
		virtual bool load(std::istream& is) { return true; }
	private:
		VectorF result_;
		VectorF mask_;
		size_t input_;
		float probability_;
	};

}

#endif