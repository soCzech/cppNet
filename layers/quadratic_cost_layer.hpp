#include "../core/cost_layer.hpp"

#ifndef CPPNET_quadratic_cost_layer_
#define CPPNET_quadratic_cost_layer_


namespace cppNet {

	class quadratic_cost_layer : public cost_layer {
	public:
		quadratic_cost_layer(activation_fn fn, activation_fn fn_prime): fn_(fn), fn_prime_(fn_prime) {};

		virtual const VectorF& compute(const VectorF& v, bool training);
		virtual const VectorF& get_last_result();

		virtual VectorF cost_function(const VectorF& prev_result, const VectorF& desired_output);
		virtual float get_last_loss(const VectorF& desired_output);
	private:
		VectorF result_;
		activation_fn fn_;
		activation_fn fn_prime_;
	};

}

#endif