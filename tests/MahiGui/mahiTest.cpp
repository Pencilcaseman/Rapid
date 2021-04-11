// #define MAHI_GUI_NO_CONSOLE

#include <iostream>
#include <rapid.h>

class MyApp : public mahi::gui::Application
{
public:
	MyApp() : Application(500, 500, "Window!")
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

		ImGui::BulletText("This is an ImGui window being used from Rapid");
		ImGui::BulletText("In the menu above are some tools you can use");
		ImGui::BulletText("Try the dropdown menus below");

		if (ImGui::CollapsingHeader("Framerate Settings"))
		{
			ImGui::BulletText("Change whether the framerate is limited with the checkbox");
			ImGui::BulletText("Adjust the limiting framerate with the slider");

			static int fps = 120;
			static bool limit = false;

			ImGui::SliderInt("Framerate", &fps, 1, 200, "%iHz"); ImGui::SameLine();
			ImGui::Checkbox("Limit FPS", &limit);

			if (limit) set_frame_limit(mahi::util::hertz(fps));
			else set_frame_limit(mahi::util::hertz(0));
		}

		if (ImGui::CollapsingHeader("There should be a graph under here!"))
		{
			ImGui::BulletText("Double click the graph below to rescale the axes and focus the graph");
			ImGui::BulletText("Click and drag to pan around the plot");
			ImGui::BulletText("Use the scroll wheel to zoom in and out");
			ImGui::BulletText("You can also drag the axes to move horizontally or vertically");

			static float64 x[2000]{};
			static float64 y[2000]{};

			for (uint64 i = 0; i < 1000; i++)
			{
				x[i] = (float64) i / 100;
				y[i] = sin((float64) i / 100 + TIME * 2) * 10;
			}

			for (uint64 i = 0; i < 1000; i++)
			{
				x[i + 1000] = (float64) i / 100;
				y[i + 1000] = cos((float64) i / 100 + TIME * 2) * 10;
			}

			ImPlot::BeginPlot("Simple Waves");

			ImPlot::PlotLine("Sine Wave", x, y, 1000);
			ImPlot::PlotLine("Cosine Wave", x + 1000, y + 1000, 1000);

			ImPlot::EndPlot();
		}

		if (ImGui::CollapsingHeader("Views"))
		{
			ImGui::BulletText("Click and drag with the right mouse button to select an area of the graph to zoom into");
			ImGui::BulletText("Click and drag with the middle mouse button to enlarge a portion of the plot");

			ImGui::Indent(50);
			{
				ImGui::BulletText("Hold shift to select only on the X axis");
				ImGui::BulletText("Hold alt to select only on the Y axis");
			}
			ImGui::Unindent(50);

			ImGui::BulletText("Click and drag the box to move the selected area around");

			static float32 x_data[512]{};
			static float32 y_data1[512]{};
			static float32 y_data2[512]{};
			static float32 y_data3[512]{};
			static float32 sampling_freq = 44100;
			static float32 freq = 500;

			for (uint64 i = 0; i < 512; ++i)
			{
				const float32 t = i / sampling_freq;
				x_data[i] = t;
				const float32 arg = 2 * 3.14f * freq * t;
				y_data1[i] = sinf(arg);
				y_data2[i] = y_data1[i] * -0.6f + sinf(2 * arg) * 0.4f;
				y_data3[i] = y_data2[i] * -0.6f + sinf(3 * arg) * 0.4f;
			}

			ImPlot::SetNextPlotLimits(0, 0.01, -1, 1);
			ImPlotAxisFlags flags = 0; //  ImPlotAxisFlags_NoTickLabels;
			ImPlotLimits query;

			if (ImPlot::BeginPlot("##View1", nullptr, nullptr, ImVec2(-1, 300), ImPlotFlags_Query, flags, flags))
			{
				ImPlot::PlotLine("Signal 1", x_data, y_data1, 512);
				ImPlot::PlotLine("Signal 2", x_data, y_data2, 512);
				ImPlot::PlotLine("Signal 3", x_data, y_data3, 512);
				query = ImPlot::GetPlotQuery();
				ImPlot::EndPlot();
			}

			ImPlot::SetNextPlotLimits(query.X.Min, query.X.Max, query.Y.Min, query.Y.Max, ImGuiCond_Always);
			if (ImPlot::BeginPlot("##View2", nullptr, nullptr, ImVec2(-1, 300), ImPlotFlags_CanvasOnly, ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_NoDecorations))
			{
				ImPlot::PlotLine("Signal 1", x_data, y_data1, 512);
				ImPlot::PlotLine("Signal 2", x_data, y_data2, 512);
				ImPlot::PlotLine("Signal 3", x_data, y_data3, 512);
				ImPlot::EndPlot();
			}
		}

		if (ImGui::CollapsingHeader("Random Number Generation"))
		{
			// static std::random_device randomNumberGenerator;
			static std::mt19937 randomNumberGenerator;
			static std::uniform_real_distribution<float64> realDistribution(0, 10);

			static float64 x[100000]{};
			static float64 y[100000]{};

			static int points = 10;
			static bool randomize = true;

			// ImGui::Text("Entropy: %f", randomNumberGenerator.entropy());
			ImGui::SliderInt("Points", &points, 1, 100000, "%i points");
			ImGui::SameLine(); ImGui::Checkbox("Randomize", &randomize);

			if (randomize)
			{
				for (uint64 i = 0; i < points; i++)
				{
					x[i] = (float64) i / ((float64) points / 10);
					y[i] = realDistribution(randomNumberGenerator);
				}
			}

			if (ImPlot::BeginPlot("Random Noise Generators", nullptr, nullptr, ImVec2(-1, 400), ImPlotAxisFlags_NoDecorations))
			{
				ImPlot::PlotScatter("Random Noise", x, y, points);
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
	std::cout << "Has blas: " << RAPID_HAS_BLAS << "\n";

	MyApp app;
	app.run();

	return 0;
}
