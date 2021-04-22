#pragma once

#include "internal.h"

namespace rapid
{
	namespace setup
	{
		class Setup
		{
		public:
			Setup()
			{
				// Put code in here to run it at program start
				ImGui::CreateContext();

				ImGuiIO &io = ImGui::GetIO();
				io.FontGlobalScale = 1;

				ImGui::DestroyContext();
			}
		};
	}
}
