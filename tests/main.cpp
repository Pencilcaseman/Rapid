// #define RAPID_NO_BLAS
// #define RAPID_NO_AMP

#include <iostream>
#include <rapid.h>

class MyApp : public mahi::gui::Application
{
public:
	MyApp() : Application()
	{}

	void update() override
	{
		static bool isOpen = true;
		static bool plotMetrics = false;
		static bool showStyleEditor_imGui = false;
		static bool showStyleEditor_imPlot = false;

		if (plotMetrics)
			ImGui::ShowMetricsWindow(&plotMetrics);
		
		if (showStyleEditor_imGui)
		{
			ImGui::SetNextWindowSize(ImVec2(415, 762), ImGuiCond_Appearing);

			ImGui::Begin("Window Style Editor", &showStyleEditor_imGui);
			ImGui::ShowStyleEditor();
			ImGui::End();
		}

		if (showStyleEditor_imPlot)
		{
			ImGui::SetNextWindowSize(ImVec2(415, 762), ImGuiCond_Appearing);
			ImGui::Begin("Graph Style Editor", &showStyleEditor_imPlot);
			ImPlot::ShowStyleEditor();
			ImGui::End();
		}

		ImGui::SetNextWindowSize(ImVec2(600, 750), ImGuiCond_FirstUseEver);

		ImGui::Begin("This is a window!", &isOpen, ImGuiWindowFlags_MenuBar);

		ImGui::BeginMenuBar();
		if (ImGui::BeginMenu("Tools"))
		{
			ImGui::MenuItem("Metrics", nullptr, &plotMetrics);
			ImGui::MenuItem("Window Style Editor", nullptr, &showStyleEditor_imGui);
			ImGui::MenuItem("Graph Style Editor", nullptr, &showStyleEditor_imPlot);
			ImGui::EndMenu();
		}
		ImGui::EndMenuBar();

		if (ImGui::CollapsingHeader("There should be a graph under here!"))
		{
			static double x[2000]{};
			static double y[2000]{};

			for (uint64_t i = 0; i < 1000; i++)
			{
				x[i] = (double) i / 100;
				y[i] = sin((double) i / 100 + TIME * 2) * 10;
			}

			for (uint64_t i = 0; i < 1000; i++)
			{
				x[i + 1000] = (double) i / 100;
				y[i + 1000] = cos((double) i / 100 + TIME * 2) * 10;
			}

			ImPlot::BeginPlot("Simple Waves");

			ImPlot::PlotLine("Sine Wave", x, y, 1000);
			ImPlot::PlotLine("Cosine Wave", x + 1000, y + 1000, 1000);

			ImPlot::EndPlot();
		}

		if (ImGui::CollapsingHeader("Views"))
		{
			// mimic's soulthread's imgui_plot demo
			static float x_data[512];
			static float y_data1[512];
			static float y_data2[512];
			static float y_data3[512];
			static float sampling_freq = 44100;
			static float freq = 500;
			for (size_t i = 0; i < 512; ++i)
			{
				const float t = i / sampling_freq;
				x_data[i] = t;
				const float arg = 2 * 3.14f * freq * t;
				y_data1[i] = sinf(arg);
				y_data2[i] = y_data1[i] * -0.6f + sinf(2 * arg) * 0.4f;
				y_data3[i] = y_data2[i] * -0.6f + sinf(3 * arg) * 0.4f;
			}
			ImGui::BulletText("Query the first plot to render a subview in the second plot (see above for controls).");
			ImPlot::SetNextPlotLimits(0, 0.01, -1, 1);
			ImPlotAxisFlags flags = 0; //  ImPlotAxisFlags_NoTickLabels;
			ImPlotLimits query;
			if (ImPlot::BeginPlot("##View1", NULL, NULL, ImVec2(-1, 300), ImPlotFlags_Query, flags, flags))
			{
				ImPlot::PlotLine("Signal 1", x_data, y_data1, 512);
				ImPlot::PlotLine("Signal 2", x_data, y_data2, 512);
				ImPlot::PlotLine("Signal 3", x_data, y_data3, 512);
				query = ImPlot::GetPlotQuery();
				ImPlot::EndPlot();
			}
			ImPlot::SetNextPlotLimits(query.X.Min, query.X.Max, query.Y.Min, query.Y.Max, ImGuiCond_Always);
			if (ImPlot::BeginPlot("##View2", NULL, NULL, ImVec2(-1, 300), ImPlotFlags_CanvasOnly, ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_NoDecorations))
			{
				ImPlot::PlotLine("Signal 1", x_data, y_data1, 512);
				ImPlot::PlotLine("Signal 2", x_data, y_data2, 512);
				ImPlot::PlotLine("Signal 3", x_data, y_data3, 512);
				ImPlot::EndPlot();
			}
		}

		ImGui::End();

		if (!isOpen)
			quit();
	}
};

int main()
{
	std::cout << "Hello, World!\n";

	MyApp app;
	app.run();

	return 0;
}
