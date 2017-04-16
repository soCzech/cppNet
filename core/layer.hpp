#include<memory>
#include<iostream>

#include "core.hpp"

#ifndef CPPNET_layer_
#define CPPNET_layer_

namespace cppNet {

	class layer {
	public:
		virtual void save(std::ostream& os) = 0;
		virtual bool load(std::istream& is) = 0;

		virtual const VectorF& compute(const VectorF& v, bool training) = 0;
		virtual const VectorF& get_last_result() = 0;

		virtual VectorF backpropagate(const VectorF& prev_result, VectorF delta) = 0;
		virtual void update_weights(float learning_rate, float l2regularizer, size_t batch_size) = 0;
	};

	using layer_ptr = std::unique_ptr<layer>;
}

#endif
