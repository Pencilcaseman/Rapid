#pragma once

#include "internal.h"

namespace rapid
{
	namespace setup
	{
		class Setup
		{
		public:
			// Put code in here to run it at program start
			Setup()
			{
				ImGuiIO &io = ImGui::GetIO();
				ImGuiStyle &style = ImGui::GetStyle();
				ImFontAtlas *atlas = io.Fonts;
				
				for (int i = 0; i < atlas->Fonts.Size; i++)
				{
					ImFont *font = atlas->Fonts[i];
					font->Scale = 1;
				}

				atlas->Build();
				style.ScaleAllSizes(1);
				io.FontGlobalScale = 1;
			}
		};
	}
}