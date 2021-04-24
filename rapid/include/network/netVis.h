#pragma once

#include "../internal.h"
#include "networkCore.h"

namespace rapid
{
	namespace neural
	{
		template<typename t>
		class NetVis : public mahi::gui::Application
		{
		public:
			NetVis() : Application() // Application(640, 480, "Rapid NetVis")
			{}

			NetVis(Network<t> *net, const TrainConfig &config) : Application() // Application(640, 480, "Rapid NetVis")
			{
				m_Network = net;
				m_Config = config;
				m_Thread = std::thread(&Network<t>::_fit, net, config);
			}

			~NetVis()
			{
				if (!m_Open)
					return;

				m_Thread.join();
				quit();
			}

			void update() override
			{
				ImGui::Begin("Plot", &m_Open);

				if (m_Network == nullptr)
				{
					ImGui::End();
					return;
				}

				double maxX = m_Network->m_LossRecord.size();

				ImPlot::SetNextPlotLimitsX(0, maxX, ImGuiCond_FirstUseEver);
				if (ImPlot::BeginPlot("Loss vs Epoch", "Epoch", "Loss"))
				{
					;

					const std::vector<t> data = m_Network->m_LossRecord;

					if (data.size() > 0)
					{
						auto x = std::vector<t>(data.size());
						for (uint64 i = 0; i < x.size(); i++)
							x[i] = (t) i;

						ImPlot::PlotLine("Loss", x.data(), data.data(), x.size());
					}

					auto [mouseX, mouseY] = ImPlot::GetPlotMousePos();
					auto range = ImPlot::GetPlotLimits();
					auto windowMinX = range.X.Min;
					auto windowMaxX = range.X.Max;
					auto windowMinY = range.Y.Min;
					auto windowMaxY = range.Y.Max;
					auto [width, height] = ImPlot::GetPlotSize();

					auto xPos = math::min(math::max(math::round(mouseX), 0), data.size() - 1);

					ImPlot::PlotVLines("##LossLine", &xPos, 1);

					std::string format = "Loss: " + std::to_string(data[(uint64) xPos]);

					auto screenspaceX = math::map(xPos, windowMinX, windowMaxX, 0, width);
					auto screenspaceY = math::map(data[(uint64) xPos], windowMinY, windowMaxY, 0, height);
					bool left = true, up = true;
					if (screenspaceX < 50 + 105) left = false;
					if (screenspaceY > height - 50 - 20) up = false;

					ImPlot::Annotate(xPos, data[(uint64) xPos], ImVec2(left ? -50 : 50, up ? -50 : 50), ImVec4(175, 165, 180, 255), format.c_str());

					ImPlot::EndPlot();
				}

				ImGui::End();

				if (!m_Open)
				{
					m_Network->m_Training = false;
					m_Thread.join();
					quit();
				}
			}

		private:
			Network<t> *m_Network = nullptr;
			TrainConfig m_Config;
			std::thread m_Thread;

			bool m_Open = true;
		};
	}
}
