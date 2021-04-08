#pragma once

#include "../internal.h"
#include "systemExecute.h"

namespace rapid
{
	namespace io
	{
		void createDirectory(const std::string &dir)
		{
			std::string command; // The command to run -- Cannot run sub-commands as each system call is an isolated process

			// Split directory path into individual folder paths
			std::vector<std::string> folders;

			std::string tempDir = dir;
			size_t index;

			while ((index = tempDir.find_first_of('/')) != std::string::npos) // Loop while valid directories exist
			{
				folders.emplace_back(tempDir.substr(0, index));
				tempDir.erase(0, index + 1);
			}

			// Check folders for drive directories, as these requires the extra slash
			for (auto &folder : folders)
			{
				if (folder.find_last_of(':') == folder.length() - 1)
				{
					command += folder + " && ";
					folder += "/";
				}
			}

			folders.emplace_back(tempDir);

			// Find the deepest folder that exists
			index = 0;
			bool goDeeper = true;

			command += "cd ";
			std::string cdTo;

			while (goDeeper && index < folders.size())
			{
				if (pathExists(cdTo + "/" + folders[index]))
				{
					cdTo += (index == 0 ? "" : "/") + folders[index];
					index++;
				}
				else
				{
					goDeeper = false;
				}
			}

			command += cdTo;

			// Create sub-directories from the deepest available folder
			for (; index < folders.size(); index++)
			{
				command += " && mkdir " + folders[index];
				command += " && cd " + folders[index];
			}

			exec(command);
		}
	}
}
