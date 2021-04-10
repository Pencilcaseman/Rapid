#pragma once

#include "../internal.h"

namespace rapid
{
	namespace parser
	{
		std::vector<std::string> splitString(const std::string &string, const std::vector<std::string> delimiters = {" "})
		{
			std::vector<std::string> res;

			uint64 start = 0;
			uint64 end = 0;

			while (end != std::string::npos)
			{
				// Find the nearest delimiter
				uint64 nearest = (uint64) -1;
				uint64 index = 0;

				for (uint64 i = 0; i < delimiters.size(); i++)
				{
					auto pos = string.find(delimiters[i], start);
					if (pos != std::string::npos && pos < nearest)
					{
						nearest = pos;
						index = i;
					}
				}

				if (nearest == (uint64) -1) // Nothing else was found
					break;

				end = nearest;
				res.emplace_back(std::string(string.begin() + start, string.begin() + end));
				res.emplace_back(delimiters[index]);
				start = end + 1;
			}

			res.emplace_back(std::string(string.begin() + start, string.end()));

			return res;
		}

		inline bool isalphanum(const std::string &string)
		{
			uint64 i = 0;
			for (const auto &c : string)
			{
				if (!((i == 0 && (c == '-' || c == '+') && string.length() > 1) || (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c == '.')))
					return false;
				i++;
			}
			return true;
		}

		inline bool isalpha(const std::string &string)
		{
			for (const auto &c : string)
				if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')))
					return false;
			return true;
		}

		inline bool isnum(const std::string &string)
		{
			uint64 i = 0;
			for (const auto &c : string)
			{
				if (!((i == 0 && (c == '-' || c == '+') && string.length() > 1) || (c >= '0' && c <= '9') || (c == '.')))
					return false;
				i++;
			}
			return true;
		}
	}
}
