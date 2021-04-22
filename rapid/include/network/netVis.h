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
			NetVis() : Application(640, 480, "Rapid NetVis")
			{}

			NetVis(Network<t> *net) : Application(640, 480, "Rapid NetVis")
			{
				m_Networks.emplace_back(net);
			}

			~NetVis()
			{
				quit();
			}

			void update() override
			{
				ImGui::Begin("Plot");

				ImPlot::FitNextPlotAxes();
				ImPlot::BeginPlot("Loss");

				auto x = std::vector<t>(m_Networks[0]->m_LossRecord.size());
				for (uint64 i = 0; i < x.size(); i++)
					x[i] = i;

				ImPlot::PlotLine("Loss", x.data(), m_Networks[0]->m_LossRecord.data(), x.size());

				ImPlot::EndPlot();

				ImGui::End();
			}

		private:
			std::vector<Network<t> *> m_Networks;
		};
	}
}
