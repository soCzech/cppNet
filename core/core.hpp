#include<vector>
#include<iostream>
#include<exception>

#include<Eigen/Dense>

#ifndef CPPNET_core_
#define CPPNET_core_

namespace cppNet {

	/*	=====
		cppNet data types
		=====*/

	using VectorF = Eigen::VectorXf;
	using MatrixF = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	// function accepting VectorF and returning VectorF
	using activation_fn = VectorF(*)(const VectorF& v);

	struct training_data {
		VectorF input;
		VectorF output;
	};
	using data_vector = std::vector<training_data>;

	/*	=====
		cppNet exception class
		=====*/

	class cppnet_exception : public std::exception {
	public:
		cppnet_exception(const char* msg) : err_msg(msg) {}
		virtual const char* what() const noexcept { return err_msg; }
	private:
		const char* err_msg;
	};

	/*	=====
		Neuron functions
		all functions described in user documentation
		=====*/

	inline float sign_func(float x) {
		if (x > 0) return 1.0;
		else return 0.0;
	}

	inline VectorF sigmoid_function(const VectorF& v) {
		return 1 / (1 + (-v).array().exp());
	}

	inline VectorF sigmoid_prime_function(const VectorF& v) {
		VectorF x(sigmoid_function(v));
		return x.array().cwiseProduct(1 - x.array());
	}

	inline VectorF relu_function(const VectorF& v) {
		return v.cwiseMax(0);
	}

	inline VectorF relu_prime_function(const VectorF& v) {
		return v.unaryExpr(std::ptr_fun(sign_func));
	}

	inline VectorF identity_function(const VectorF& v) {
		return v;
	}

	inline VectorF identity_prime_function(const VectorF& v) {
		return VectorF::Ones(v.size());
	}

	inline VectorF softmax_function(const VectorF& v) {
		float max = -std::numeric_limits<float>::infinity();

		for (size_t i = 0; i < v.size(); i++) {
			if (max < v[i])
				max = v[i];
		}
		VectorF x = (v.array() - max).exp();
		float sum = x.sum();

		x = (1 / sum) * x;
		return x;
	}

}

#endif
