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
				if (m_Network == nullptr)
					return;

				ImGui::Begin("Statistics and Controls");

				std::string format = "Training time: " + rapidCast<std::string>(math::round(m_Network->getTrainingTime(), 3));
				ImGui::BulletText(format.c_str());

				ImGui::End();

				ImGui::Begin("Plot", &m_Open);

				double maxX = m_Network->m_LossRecord.size();
				const std::vector<t> data = m_Network->m_LossRecord;
				auto x = std::vector<t>(data.size());
				for (uint64 i = 0; i < x.size(); i++)
					x[i] = (t) i;

				ImPlot::SetNextPlotLimitsX(-(m_Network->m_TrainConfig.epochs * 0.1), m_Network->m_TrainConfig.epochs + (m_Network->m_TrainConfig.epochs * 0.1), ImGuiCond_FirstUseEver);
				ImPlot::SetNextPlotLimitsY(-0.1, 1.1, ImGuiCond_FirstUseEver);

				if (ImPlot::BeginPlot("Loss vs Epoch", "Epoch", "Loss", ImVec2(-1, ImGui::GetWindowHeight() / 2 - 20), ImPlotFlags_Query))
				{
					ImPlot::PlotLine("Loss", x.data(), data.data(), x.size(), 0, sizeof(t));

					auto [mouseX, mouseY] = ImPlot::GetPlotMousePos();
					auto range = ImPlot::GetPlotLimits();
					auto windowMinX = range.X.Min;
					auto windowMaxX = range.X.Max;
					auto windowMinY = range.Y.Min;
					auto windowMaxY = range.Y.Max;
					auto [width, height] = ImPlot::GetPlotSize();

					auto xPos = math::min(math::max(math::round(mouseX), 0), data.size() - 1);

					if (mouseX > windowMinX && mouseX < windowMaxX &&
						mouseY > windowMinY && mouseY < windowMaxY &&
						!ImGui::IsMouseDragging(0) &&
						!ImGui::IsMouseDragging(1) &&
						!ImGui::IsMouseDragging(2))
					{
						std::string format = "Loss: " + std::to_string(data[(uint64) xPos]);

						auto screenspaceX = math::map(xPos, windowMinX, windowMaxX, 0, width);
						auto screenspaceY = math::map(data[(uint64) xPos], windowMinY, windowMaxY, 0, height);
						bool left = true, up = true;
						if (screenspaceX < 50 + 105) left = false;
						if (screenspaceY > height - 50 - 20) up = false;

						ImPlot::PlotVLines("##LossLine", &xPos, 1);
						ImPlot::Annotate(xPos, data[(uint64) xPos], ImVec2(left ? -50 : 50, up ? -50 : 50), ImVec4(175, 165, 180, 255), format.c_str());
					}

					m_Query = ImPlot::GetPlotQuery();

					ImPlot::EndPlot();
				}

				ImPlot::SetNextPlotLimits(m_Query.X.Min, m_Query.X.Max, m_Query.Y.Min, m_Query.Y.Max, ImGuiCond_Always);
				if (ImPlot::BeginPlot("##View", "Epoch", "Loss", ImVec2(-1, ImGui::GetWindowHeight() / 2 - 20)))
				{
					ImPlot::PlotLine("Loss", x.data(), data.data(), x.size());
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
			ImPlotLimits m_Query;
		};
	}
}
