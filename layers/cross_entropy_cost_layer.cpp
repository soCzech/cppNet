#include<limits>
#include<Eigen/Core>
#include "cross_entropy_cost_layer.hpp"


namespace cppNet {

	const VectorF& cross_entropy_cost_layer::compute(const VectorF& v, bool training) {
		// compute the resukt
		result_ = fn_(v);
		return result_;
	}

	const VectorF& cross_entropy_cost_layer::get_last_result() {
		return result_;
	}

	float cross_entropy_cost_layer::get_last_loss(const VectorF& desired_output) {
		// return the loss
		// ylna + (1−y)ln(1−a)
		return -(
			desired_output.cwiseProduct(
				normalize(result_.array().log())
			) +
			((VectorF)(1 - desired_output.array())).cwiseProduct(
				normalize((-result_).array().log1p())
			)).sum();
	}

	VectorF cross_entropy_cost_layer::cost_function(const VectorF& prev_result, const VectorF& desired_output) {
		// return the gradients
		return result_ - desired_output;
	}

	VectorF cross_entropy_cost_layer::normalize(VectorF v) {
		// used to compute loss function in numerically stable way
		for (size_t i = 0; i < v.size(); i++) {
			if (!std::isfinite(v[i])) {
				v[i] = 0.0;
			}
		}
		return v;
	}

}
