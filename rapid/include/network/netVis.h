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
			NetVis() : Application()
			{}

			NetVis(Network<t> *net, const TrainConfig &config) : Application()
			{
				m_Network = net;
				m_Config = config;

				// Start the network training process on creation
				m_Thread = std::thread(&Network<t>::_fit, net, config);
			}

			~NetVis()
			{
				// Close windows and join threads to clean everything up

				if (!m_Open)
					return;

				m_Thread.join();
				quit();
			}

			void update() override
			{
				if (m_Network == nullptr)
					return;

				m_CurrentTime = TIME;

				// Set the font size to 1 by default (for some reason some devices do not default to this)
				if (m_PrevTimeStep == 0)
				{
					auto &io = ImGui::GetIO();
					io.FontGlobalScale = 1.;
					m_PrevTimeStep = m_CurrentTime;
				}

				// Create a window for statistics and network control
				ImGui::SetNextWindowSize(ImVec2(300, 300), ImGuiCond_FirstUseEver);
				ImGui::Begin("Statistics and Controls", &m_Open);

				// Show the elapsed time
				std::string format = "Elapsed time: " + rapidCast<std::string>(math::round(m_Network->getTrainingTime(), 3));
				ImGui::BulletText(format.c_str());

				// Show the remaining time
				ImGui::BulletText(calculateTimeRemaining().c_str());

				// Show the training percentage
				ImGui::BulletText(calculateTrainingPercentage().c_str());

				// Show epoch and batch number
				ImGui::BulletText("Epoch: %llu", m_Network->m_Epoch);
				ImGui::BulletText("Batch number: %llu", m_Network->m_BatchNum);

				ImGui::End();

				// Create the loss graph window
				ImGui::SetNextWindowSize(ImVec2(1000, 800), ImGuiCond_FirstUseEver);
				ImGui::Begin("Neural Network Loss", &m_Open);

				uint64 epoch = m_Network->m_Epoch;
				double maxX = m_Network->m_LossRecord.size();
				const std::vector<t> data = m_Network->m_LossRecord;
				auto x = std::vector<t>(epoch);
				for (uint64 i = 0; i < x.size(); i++)
					x[i] = (t) i;

				ImPlot::SetNextPlotLimitsX(-(m_Network->m_TrainConfig.epochs * 0.1), m_Network->m_TrainConfig.epochs + (m_Network->m_TrainConfig.epochs * 0.1), ImGuiCond_FirstUseEver);
				ImPlot::SetNextPlotLimitsY(-0.1, 1.1, ImGuiCond_FirstUseEver);

				if (ImPlot::BeginPlot("Loss vs Epoch", "Epoch", "Loss", ImVec2(-1, -1)))
				{
					ImPlot::SetLegendLocation(ImPlotLocation_NorthWest, ImPlotOrientation_Horizontal, false);

					// Important values for plotting things
					auto [mouseX, mouseY] = ImPlot::GetPlotMousePos();
					auto range = ImPlot::GetPlotLimits();
					auto windowMinX = range.X.Min;
					auto windowMaxX = range.X.Max;
					auto windowMinY = range.Y.Min;
					auto windowMaxY = range.Y.Max;
					auto [width, height] = ImPlot::GetPlotSize();

					int lod = 1;
					if (windowMaxX - windowMinX > 100) lod = std::ceil((windowMaxX - windowMinX) * 0.001);

					ImPlot::PlotLine("Loss", x.data(), data.data(), epoch / lod, 0, sizeof(t) * lod);

					auto xPos = math::min(math::max(math::round(mouseX), 0), epoch - 1);

					if (mouseX > windowMinX && mouseX < windowMaxX &&
						mouseY > windowMinY && mouseY < windowMaxY &&
						!ImGui::IsMouseDragging(0) &&
						!ImGui::IsMouseDragging(1) &&
						!ImGui::IsMouseDragging(2))
					{
						auto screenspaceX = math::map(xPos, windowMinX, windowMaxX, 0, width);
						auto screenspaceY = math::map(data[(uint64) xPos], windowMinY, windowMaxY, 0, height);
						bool left = true, up = true;
						if (screenspaceX < 50 + 105) left = false;
						if (screenspaceY > height - 50 - 20) up = false;

						ImPlot::PlotVLines("##LossLineY", &xPos, 1);

						std::string format;
						format += "Epoch: " + std::to_string((uint64) xPos + 1) + "\n";
						format += "Loss: " + std::to_string(data[(uint64) xPos]);
						ImPlot::AnnotateClamped(xPos, data[(uint64) xPos], ImVec2(50, -50), ImVec4(0.341, 0.341, 0.341, 255), format.c_str());
					}

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

			inline std::string calculateTimeRemaining()
			{
				// Update remaining time

				if (m_CurrentTime - m_PrevTimeStep < 0.5)
					goto time_end;

				auto deltaT = m_CurrentTime - m_PrevTimeStep;
				auto deltaE = (double) (m_Network->m_Epoch - m_PrevEpoch);

				auto iterationsPerSecond = (double) deltaE / deltaT;
				m_TimeRemaining = (double) (m_Network->m_TrainConfig.epochs - m_Network->m_Epoch - 1) / iterationsPerSecond;
				m_PrevTimeStep = m_CurrentTime;
				m_PrevEpoch = m_Network->m_Epoch;

				if (m_Network->m_Epoch == m_Network->m_TrainConfig.epochs)
					m_TimeRemaining = 0;

			time_end:

				return "Remaining time: " + rapidCast<std::string>(math::round(m_TimeRemaining, 0));
			}

			inline std::string calculateTrainingPercentage() const
			{
				// Find the training progress as a percentage
				double percentage = math::round(((double) m_Network->m_Epoch / (double) m_Network->m_TrainConfig.epochs) * 100., 2);
				std::string percStr = rapidCast<std::string>(percentage);
				if (percentage < 99.999999) percStr += std::string(5 - percStr.length(), '0');
				return "Training " + percStr + "%% complete";
			}

		private:
			Network<t> *m_Network = nullptr;
			TrainConfig m_Config;
			std::thread m_Thread;

			double m_CurrentTime = 1;
			double m_PrevTimeStep = 0;
			double m_TimeRemaining = 0;
			uint64 m_PrevEpoch = 0;

			bool m_Open = true;
		};
	}
}
