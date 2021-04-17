#define RAPID_NO_BLAS

#include <iostream>
#include <rapid.h>

int main()
{
	using namespace rapid::neural;
	using namespace rapid::ndarray;
	using namespace rapid::math;
	using dtype = float;

	auto *activation1 = new activation::LeakyRelu<dtype>();
	auto *activation2 = new activation::LeakyRelu<dtype>();

	auto optim1 = new optim::ADAM<dtype>(0.05);
	auto optim2 = new optim::ADAM<dtype>(0.05);

	auto layer1 = new layers::Input<dtype>(2);
	auto layer2 = new layers::Affine<dtype>(3, activation1, optim1);
	auto layer3 = new layers::Affine<dtype>(1, activation2, optim2);

	auto network = Network<dtype>();
	network.addLayers({layer1, layer2, layer3});

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
	START_TIMER(0, 2000);
	auto index = random<int>(0, 3);
	network.backward(input[index], output[index]);
	END_TIMER(0);

	std::cout << "Predict\n";
	for (int i = 0; i < 4; i++)
		std::cout << (int) input[i][0] << "^" << (int) input[i][1] << " => "
		<< network.forward(input[i])[0][0] << " (Correct: " << output[i][0] << ")" << "\n\n";

	return 0;
}
