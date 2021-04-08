// #define RAPID_NO_BLAS
// #define RAPID_NO_AMP

#include <iostream>
#include <rapid.h>

class MyApp : public mahi::gui::Application
{
public:
	MyApp() : Application(650, 480, "MyApp")
	{}

	void update() override
	{
		ImGui::Begin("This is a window!\n");
		if (ImGui::CollapsingHeader("There should be a graph under here!"))
		{
			static float x[1000]{};
			static float y[1000]{};

			for (uint64_t i = 0; i < 1000; i++)
			{
				x[i] = (double) i / 100;
				y[i] = sin((double) i / 100 + TIME * 2) * 10;
			}

			ImPlot::BeginPlot("Simple Sine Wave");

			ImPlot::PlotLine("Sine Wave", x, y, 1000);

			ImPlot::EndPlot();
		}
		ImGui::End();
	}
};

int main()
{
	std::cout << "Hello, World!\n";

	std::cout << rapid::ndarray::Array<double, rapid::ndarray::CPU>::fromData({{1, 2, 3}, {4, 5, 6}}) << "\n";

	MyApp app;
	app.run();

	return 0;
}
