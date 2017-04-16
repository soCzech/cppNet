#include "../core/layer.hpp"

#ifndef CPPNET_max_pooling_layer_
#define CPPNET_max_pooling_layer_


namespace cppNet {

	class max_pooling_layer : public layer {
	public:
		max_pooling_layer(std::initializer_list<size_t> input, std::initializer_list<size_t> pool_size);
		virtual const VectorF& compute(const VectorF& v, bool training);
		virtual const VectorF& get_last_result();

		virtual VectorF backpropagate(const VectorF& prev_result, VectorF delta);
		virtual void update_weights(float learning_rate, float l2regularizer, size_t batch_size) {}

		virtual void save(std::ostream& os) {}
		virtual bool load(std::istream& is) { return true; }
	private:
		VectorF result_;
		size_t input_[3];
		size_t pool_size_[2];
		size_t output_[2];
		size_t channels_;
	};

}

#endif