#include <ctime>
#include <string>
#include <fstream>
#include <list>
#include <vector>

#ifndef CPPNET_summary_
#define CPPNET_summary_

namespace cppNet {
	class summary {
	public:
		void close();
		void open(const std::string& dir, const std::initializer_list<std::string>& graphs);
		void log(size_t after_steps, const std::string& type, float value);
	private:
		std::string addZero(int i);
		struct summ {
			std::string name;
			float value;
			size_t last_update;
		};

		std::string filename_;
		std::ofstream summary_;
		std::vector<summ> to_write_;
		size_t last_write_ = 0;
	};

}
#endif