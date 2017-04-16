#include "mnist.hpp"

namespace cppNet {

	void mnist_loader::get_data(data_vector & td, size_t count) {
		images_.open(images_file_, std::ios::binary);
		labels_.open(labels_file_, std::ios::binary);

		char c;
		for (size_t i = 0; i < 16; i++) {
			images_.get(c);
		}
		for (size_t i = 0; i < 8; i++) {
			labels_.get(c);
		}

		for (size_t j = 0; j < count; j++) {
			training_data d;
			d.input = VectorF(784);

			char c;
			for (size_t i = 0; i < 784; i++) {
				images_.get(c);
				d.input[i] = ((unsigned char)c) / (255.0);
			}
			labels_.get(c);

			d.output = get_solution_vector(c);

			td.push_back(d);
		}

		images_.close();
		labels_.close();
	}

	VectorF mnist_loader::get_solution_vector(char c) {
		VectorF x = VectorF::Zero(10);
		x[c] = 1;
		return x;
	}
}