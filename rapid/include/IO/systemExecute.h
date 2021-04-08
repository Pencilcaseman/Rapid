#pragma once

#include "../internal.h"

namespace rapid
{
	namespace io
	{
		bool pathExists(const std::string &dir)
		{
			struct stat buffer;
			return (stat(dir.c_str(), &buffer) == 0);
		}

	#ifdef RAPID_OS_WINDOWS
		std::string exec_(const char *cmd)
		{
			std::array<char, 128> buffer;
			std::string result;
			std::unique_ptr<FILE, decltype(&_pclose)> pipe(_popen(cmd, "r"), _pclose);
			if (!pipe)
			{
				throw std::runtime_error("popen() failed!");
			}
			while (fgets(buffer.data(), (int) buffer.size(), pipe.get()) != nullptr)
			{
				result += buffer.data();
			}
			return result;
		}
	#else
		std::string exec_(const char *cmd)
		{
			std::array<char, 128> buffer;
			std::string result;
			std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
			if (!pipe)
			{
				throw std::runtime_error("popen() failed!");
			}
			while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr)
			{
				result += buffer.data();
			}
			return result;
		}
	#endif

		std::string exec(const std::string &cmd)
		{
			return exec_(cmd.c_str());
		}
	}
}
