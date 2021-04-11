#pragma once

#include "../internal.h"
#include <sys/stat.h>

namespace rapid
{
	namespace io
	{
		template<typename t>
		std::vector<std::vector<t>> loadCSV(const std::string &dir, uint64 start = 0, uint64 end = 0, bool verbose = false)
		{
			std::vector<std::vector<t>> res;

			std::fstream file;
			file.open(dir, std::ios::in);

			if (!file.is_open())
				message::RapidError("File IO Error", "Unable to open file for reading\n");

			std::string line;
			std::string delimiter = ",";

			for (uint64 i = 0; i < start; i++)
				std::getline(file, line);

			uint64 count = 0;

			while (std::getline(file, line) && count < (end == 0 ? (uint64) -1 : (end - start)))
			{
				if (verbose && count % 100 == 0)
					std::cout << "Loaded " << count << " lines\n";

				std::vector<t> row;

				uint64 pos = 0;
				std::string token;
				while ((pos = line.find(delimiter)) != std::string::npos)
				{
					row.emplace_back(rapidCast<t>(line.substr(0, pos)));
					line.erase(0, pos + delimiter.length());
				}

				row.emplace_back(rapidCast<t>(line));

				res.emplace_back(row);
				count++;
			}

			return res;
		}
	}
}
