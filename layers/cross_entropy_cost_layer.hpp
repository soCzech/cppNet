#include "../core/cost_layer.hpp"

#ifndef CPPNET_cross_entropy_cost_layer_
#define CPPNET_cross_entropy_cost_layer_


namespace cppNet {

	class cross_entropy_cost_layer : public cost_layer {
	public:
		cross_entropy_cost_layer(activation_fn fn) : fn_(fn) {};
		virtual const VectorF& compute(const VectorF& v, bool training);
		virtual const VectorF& get_last_result();

		virtual VectorF cost_function(const VectorF& prev_result, const VectorF& desired_output);
		virtual float get_last_loss(const VectorF& desired_output);
	private:
		VectorF normalize(VectorF v);
		VectorF result_;
		activation_fn fn_;
	};

}

#endif