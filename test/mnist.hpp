#include<string>
#include<fstream>
#include<iostream>

#include "../core/core.hpp"

#ifndef CPPNET_mnist_
#define CPPNET_mnist_

namespace cppNet {

	class mnist_loader {
	public:
		mnist_loader(const std::string& images, const std::string& labels) : images_file_(images), labels_file_(labels) {}
		void get_data(data_vector& td, size_t count);
	private:
		VectorF get_solution_vector(char c);

		std::ifstream images_;
		std::ifstream labels_;
		std::string images_file_;
		std::string labels_file_;
	};

}


#endif