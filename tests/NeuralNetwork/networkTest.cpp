#define RAPID_NO_BLAS

#include <iostream>
#include <rapid.h>

int main()
{
	/*
	PUT ACTIVATIONS INTO A POLYMORPHIC CLASS
	FUNCTION TO GET INITIALIZATION TYPE
	INITIALIZATION TYPE ALSO OVERLOADABLE
	*/

	auto *activation = (activationPtr<float64>) rapid::neural::activation::leakyRelu<float64>;
	auto *derivative = (activationPtr<float64>) rapid::neural::activation::leakyReluDerivative<float64>;

	auto optim1 = new rapid::neural::optim::ADAM<float64>(0.01);
	auto optim2 = new rapid::neural::optim::ADAM<float64>(0.01);

	auto layer1 = new rapid::neural::layers::Input<float64>(2);
	auto layer2 = new rapid::neural::layers::Affine<float64>(5, std::make_pair(activation, derivative), optim1);
	auto layer3 = new rapid::neural::layers::Affine<float64>(1, std::make_pair(activation, derivative), optim2);

	auto network = rapid::neural::Network<float64>();
	network.addLayers({layer1, layer2, layer3});

	// Inputs for XOR
	std::vector<rapid::ndarray::Array<float64>> input = {
		rapid::ndarray::Array<float64>::fromData({0, 0}).reshaped({2, 1}),
		rapid::ndarray::Array<float64>::fromData({0, 1}).reshaped({2, 1}),
		rapid::ndarray::Array<float64>::fromData({1, 0}).reshaped({2, 1}),
		rapid::ndarray::Array<float64>::fromData({1, 1}).reshaped({2, 1})
	};

	// Targets for XOR
	std::vector<rapid::ndarray::Array<float64>> output = {
		rapid::ndarray::Array<float64>::fromData({0}).reshaped({1, 1}),
		rapid::ndarray::Array<float64>::fromData({1}).reshaped({1, 1}),
		rapid::ndarray::Array<float64>::fromData({1}).reshaped({1, 1}),
		rapid::ndarray::Array<float64>::fromData({0}).reshaped({1, 1})
	};

	network.compile();

	std::cout << "Train\n";
	START_TIMER(0, 5000);
	auto index = rapid::math::random<int>(0, 3);
	network.backward(input[index], output[index]);
	END_TIMER(0);

	std::cout << "Predict\n";
	for (int i = 0; i < 4; i++)
		std::cout << input[i] << " => " << network.forward(input[i]) << " (" << output[i] << ")" << "\n\n";

	delete optim1;
	delete optim2;

	delete layer1;
	delete layer2;
	delete layer3;

	return 0;
}
