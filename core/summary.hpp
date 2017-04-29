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
			// graph name
			std::string name;
			// last logged value
			float value;
			// when was the value last time logged
			size_t last_update;
		};

		std::string filename_;
		std::ofstream summary_;
		// vector of graphs that are being logged
		std::vector<summ> to_write_;
		// counter of writes
		size_t last_write_ = 0;
	};

}
#endif