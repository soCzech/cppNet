#include "layer.hpp"

#ifndef CPPNET_cost_layer_
#define CPPNET_cost_layer_

namespace cppNet {

	class cost_layer : public layer {
	public:
		virtual VectorF backpropagate(const VectorF& prev_result, VectorF delta) {
			throw cppnet_exception("Cost layer does not implement backpropagate function.");
		}
		virtual void update_weights(float learning_rate, float l2regularizer, size_t batch_size) {
			throw cppnet_exception("Cost layer does not implement update_weights function.");
		};

		virtual void save(std::ostream& os) {}
		virtual bool load(std::istream& is) { return true; }

		virtual VectorF cost_function(const VectorF& prev_result, const VectorF& desired_output) = 0;
		virtual float get_last_loss(const VectorF& desired_output) = 0;
	};

}

#endif
