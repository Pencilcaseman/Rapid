#define RAPID_NO_BLAS
#define RAPID_NO_OMP

#include <iostream>
#include <rapid.h>

class TestApp : public mahi::gui::Application
{
public:

	TestApp(rapid::neural::Network<float> &network) : Application(500, 500, "Neural Network Thing"), m_Network(network)
	{}

	void update() override
	{
		auto [x, y] = get_mouse_pos();
		auto [w, h] = get_window_size();

		std::map<std::string, rapid::ndarray::Array<float>> networkInput = {
			{"x1", rapid::ndarray::fromScalar<float>(x / w)},
			{"x2", rapid::ndarray::fromScalar<float>(y / h)}
		};

		std::cout << m_Network.forward(networkInput)[0][0] << "\n";
	}

	rapid::neural::Network<float> &m_Network;
};

int main()
{
	using namespace rapid::neural;
	using namespace rapid::ndarray;
	using namespace rapid::math;
	using dtype = float;

	Config<dtype> config {
		{{"x1", 1},       // Input 1
		 {"x2", 1}},      // Input 2

		{{"y", 1}},       // Output

		{3},              // Hidden layer nodes

		{"LeakyRelu"},    // Activation functions
		{"ADAM"},         // Optimizers
		{0.1}             // Learning rates
	};

	auto network = Network<dtype>(config);

	// Inputs for XOR
	std::vector<Array<dtype>> input = {
		Array<dtype>::fromData({0, 0}),
		Array<dtype>::fromData({0, 1}),
		Array<dtype>::fromData({1, 0}),
		Array<dtype>::fromData({1, 1})
	};

	// Targets for XOR
	std::vector<Array<dtype>> output = {
		Array<dtype>::fromData({0}),
		Array<dtype>::fromData({1}),
		Array<dtype>::fromData({1}),
		Array<dtype>::fromData({0})
	};

	network.compile();

	std::cout << "Train\n";
	START_TIMER(0, 5000);
	auto index = random<int>(0, 3);
	network.backward(input[index], output[index]);
	END_TIMER(0);

	std::map<std::string, Array<dtype>> networkInput = {
		{"x1", fromScalar<dtype>(0)},
		{"x2", fromScalar<dtype>(1)}
	};

	std::cout << network.forward(networkInput) << "\n";

	std::getchar();

	TestApp myApp(network);
	myApp.run();

	return 0;
}
