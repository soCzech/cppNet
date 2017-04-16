#include<iostream>
#include "summary.hpp"

namespace cppNet {

	void summary::open(const std::string& dir, const std::initializer_list<std::string>& graphs) {
		time_t now = time(0);
		tm* ltm = localtime(&now);

		filename_ = dir + "/" +
			std::to_string(1900 + ltm->tm_year) +
			addZero(1 + ltm->tm_mon) +
			addZero(ltm->tm_mday) + "_" +
			addZero(ltm->tm_hour) +
			addZero(ltm->tm_min) +
			addZero(ltm->tm_sec) + ".log";

		std::cout << "Logging to " << filename_ << ". Be sure the directory exists." << std::endl;
		std::cout << "To see progress, start HTTP server in dashboard directory by command 'python server.py --logdir=$PATH_TO_LOGDIR'." << std::endl;

		summary_.open(filename_, std::ios::binary);
		for (auto&& s : graphs) {
			summary_ << s << '\n';
			to_write_.push_back({ s, 0, 0 });
		}
		summary_.flush();
	}

	void summary::log(size_t after_steps, const std::string& type, float value) {
		bool write = false;
		size_t type_pos;

		// find the type in vector
		for (size_t i = 0; i < to_write_.size(); i++) {
			if (to_write_[i].name == type) {
				type_pos = i;
				// if last value not saved
				if (to_write_[i].last_update != last_write_) {
					write = true;
				}
				break;
			}
		}

		if (write) {
			size_t write_before_this_one = last_write_;
			last_write_ = to_write_[type_pos].last_update;

			summary_ << last_write_ << '\n';
			for (size_t i = 0; i < to_write_.size(); i++) {
				if (to_write_[i].last_update > write_before_this_one) {
					summary_ << to_write_[i].value << '\n';
				} else {
					summary_ << '-' << '\n';
				}
				to_write_[i].last_update = last_write_;
			}
			summary_.flush();
		}
		to_write_[type_pos].last_update = after_steps;
		to_write_[type_pos].value = value;
	}

	void summary::close() {
		bool write = false;
		size_t update = last_write_;

		for (size_t i = 0; i < to_write_.size(); i++) {
			if (to_write_[i].last_update > update) {
				update = to_write_[i].last_update;
				write = true;
			}
		}
		if (write) {
			summary_ << update << '\n';
			for (size_t i = 0; i < to_write_.size(); i++) {
				summary_ << to_write_[i].value << '\n';
			}
			summary_.flush();
		}
		summary_.close();
	}

	std::string summary::addZero(int i) {
		auto s = std::to_string(i);
		if (s.length() == 2) return s;
		return "0" + s;
	}

}