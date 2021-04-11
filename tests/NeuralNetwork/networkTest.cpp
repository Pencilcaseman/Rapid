#include <iostream>
#include <rapid.h>

int main()
{
	auto optim1 = new rapid::neural::optim::SGD<float32>(0.01);
	auto optim2 = new rapid::neural::optim::SGD<float32>(0.01);

	auto *activation = (activationPtr<float32>) rapid::neural::activation::sigmoid<float32>;
	auto *derivative = (activationPtr<float32>) rapid::neural::activation::sigmoidDerivative<float32>;

	auto layer1 = new rapid::neural::layers::Input<float32>(2);
	auto layer2 = new rapid::neural::layers::Affine<float32>(3, std::make_pair(activation, derivative), optim1);
	auto layer3 = new rapid::neural::layers::Affine<float32>(1, std::make_pair(activation, derivative), optim2);

	auto network = rapid::neural::Network<float32>();
	network.addLayers({layer1, layer2, layer3});

	std::vector<rapid::ndarray::Array<float32>> input = {
		rapid::ndarray::Array<float32>::fromData({0, 0}).reshaped({2, 1}),
		rapid::ndarray::Array<float32>::fromData({0, 1}).reshaped({2, 1}),
		rapid::ndarray::Array<float32>::fromData({1, 0}).reshaped({2, 1}),
		rapid::ndarray::Array<float32>::fromData({1, 1}).reshaped({2, 1})
	};

	std::vector<rapid::ndarray::Array<float32>> output = {
		rapid::ndarray::Array<float32>::fromData({0}).reshaped({1, 1}),
		rapid::ndarray::Array<float32>::fromData({1}).reshaped({1, 1}),
		rapid::ndarray::Array<float32>::fromData({1}).reshaped({1, 1}),
		rapid::ndarray::Array<float32>::fromData({0}).reshaped({1, 1})
	};

	network.compile();

	std::cout << "Test\n";
	for (int i = 0; i < 1000; i++)
	{
		auto index = rapid::math::random<int>(0, 3);
		network.backward(input[index], output[index]);
	}

	for (int i = 0; i < 4; i++)
		std::cout << input[i] << " => " << network.forward(input[i]) << " (" << output[i] << ")" << "\n\n";

	delete optim1;
	delete optim2;

	delete layer1;
	delete layer2;
	delete layer3;

	return 0;
}
