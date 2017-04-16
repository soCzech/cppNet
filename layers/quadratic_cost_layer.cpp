#include<Eigen/Core>
#include "quadratic_cost_layer.hpp"


namespace cppNet {

	const VectorF& quadratic_cost_layer::compute(const VectorF& v, bool training) {
		result_ = fn_(v);
		return result_;
	}

	const VectorF& quadratic_cost_layer::get_last_result() {
		return result_;
	}

	float quadratic_cost_layer::get_last_loss(const VectorF& desired_output) {
		return 0.5 * (result_ - desired_output).array().pow(2).sum();
	}

	VectorF quadratic_cost_layer::cost_function(const VectorF& prev_result, const VectorF& desired_output) {
		return (result_ - desired_output).cwiseProduct(fn_prime_(prev_result));
	}
}
