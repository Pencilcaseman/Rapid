// #define MAHI_GUI_NO_CONSOLE

#include <iostream>
#include <rapid.h>

// A simple application
class MyApp : public mahi::gui::Application
{
public:
	// Constructor uses a Config object as a parameter
	// for the Application that is inherited
	MyApp(mahi::gui::Application::Config config) : Application(config)
	{
		// Find the pixel ratio and create the NanoVG frame buffer
		auto pxRatio = get_pixel_ratio();
		m_Fb = nvgluCreateFramebuffer(m_vg, (int) (100 * pxRatio), (int) (100 * pxRatio), NVG_IMAGE_REPEATX | NVG_IMAGE_REPEATY);
		set_background({0.3f, 0.3f, 0.32f, 1.0f});
	}

	~MyApp()
	{
		// Remove the frame buffer
		nvgluDeleteFramebuffer(m_Fb);
	}

	// This is called every frame to render things to the screen
	void update() override
	{
		// Define some static variables to keep track of what
		// should be shown, and what shouldn't
		static bool isOpen = true;
		static bool plotMetrics = false;
		static bool showStyleEditor_imGui = false;
		static bool showStyleEditor_imPlot = false;

		if (plotMetrics) // Show the ImGui metrics window
			ImGui::ShowMetricsWindow(&plotMetrics);

		if (showStyleEditor_imGui) // Show the ImGui style editor window
		{
			ImGui::SetNextWindowSize(ImVec2(415, 762), ImGuiCond_Appearing);

			ImGui::Begin("Window Style Editor", &showStyleEditor_imGui);
			ImGui::ShowStyleEditor();
			ImGui::End();
		}

		if (showStyleEditor_imPlot) // Show the ImPlot style editor window
		{
			ImGui::SetNextWindowSize(ImVec2(415, 762), ImGuiCond_Appearing);
			ImGui::Begin("Graph Style Editor", &showStyleEditor_imPlot);
			ImPlot::ShowStyleEditor();
			ImGui::End();
		}

		// Set the ImGui window size -- this will only take effect on the first
		// call, so the window can still be resized
		ImGui::SetNextWindowSize(ImVec2(600, 750), ImGuiCond_FirstUseEver);

		// Begin an ImGui window
		ImGui::Begin("This is a window!", &isOpen, ImGuiWindowFlags_MenuBar);

		// Create a menu bar
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

		ImGui::Checkbox("Show colorful boxes", &m_ColorfulBoxes);

		// Create a collapsing header
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

		// End the window
		ImGui::End();

		// If we close the window, actually close it...
		if (!isOpen)
			quit();
	}

	// This is also called each frame, but this allows you to make
	// calls to NanoVG to draw graphics to a window
	void draw(NVGcontext *vg) override
	{
		// Only render if the frame buffer is valid
		// (or if we actually want to draw the boxes)
		if (m_Fb != NULL && m_ColorfulBoxes)
		{
			NVGpaint img = nvgImagePattern(vg, 0, 0, 100, 100, rapid::math::halfPi, m_Fb->image, 1.0f);
			nvgSave(vg);

			auto [width, height] = get_window_size();
			auto [mouseX, mouseY] = get_mouse_pos();

			for (int i = 0; i < 20; i++)
			{
				for (int j = 0; j < 20; j++)
				{
					nvgBeginPath(vg);

					float x, y, w, h;
					x = ((float) i / 20.f) * width + 10;
					y = ((float) j / 20.f) * width + 10;
					w = 10;
					h = 10;

					nvgRect(vg, x, y, w, h);

					float s, l, a;
					h = std::sqrt(((mouseX - x) * (mouseX - x) + (mouseY - y) * (mouseY - y))) / width;
					s = 0.5f;
					l = 0.5f;
					a = 255;

					nvgFillColor(vg, nvgHSLA(h, s, l, a));
					nvgFill(vg);
				}
			}

			nvgBeginPath(vg);
			nvgRoundedRect(vg, get_mouse_pos().x - 150, get_mouse_pos().y - 150, 300, 300, 40);
			nvgFillPaint(vg, img);
			nvgFill(vg);
			nvgStrokeColor(vg, nvgRGBA(220, 160, 0, 255));
			nvgStrokeWidth(vg, 3.0f);
			nvgStroke(vg);
			nvgRestore(vg);
		}
	}

private:

	// Frame buffer and a member variable
	NVGLUframebuffer *m_Fb = NULL;
	bool m_ColorfulBoxes = false;
};

int main()
{
	// Create the config
	mahi::gui::Application::Config conf;
	conf.title = "NanoVG FBO Demo";
	conf.width = 1000;
	conf.height = 600;
	conf.msaa = 0;      // pick one
	conf.nvg_aa = true; // pick one

	// Run the app
	MyApp app(conf);
	app.run();

	return 0;
}
