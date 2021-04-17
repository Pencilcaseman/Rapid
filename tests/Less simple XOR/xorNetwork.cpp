#define RAPID_NO_BLAS

#include <iostream>
#include <rapid.h>

int main()
{
	// Use the namespaces to make the code neater. This is
	// not required, and is actually advised against. It's
	// purely for aesthetic purposes
	using namespace rapid::neural;
	using namespace rapid::ndarray;
	using namespace rapid::math;
	using dtype = float;

	// Create the activation functions
	auto *activation1 = new activation::LeakyRelu<dtype>();
	auto *activation2 = new activation::LeakyRelu<dtype>();

	// Create the optimizers
	auto optim1 = new optim::ADAM<dtype>(0.05);
	auto optim2 = new optim::ADAM<dtype>(0.05);

	// Create the layers
	auto layer1 = new layers::Input<dtype>(2);
	auto layer2 = new layers::Affine<dtype>(3, activation1, optim1);
	auto layer3 = new layers::Affine<dtype>(1, activation2, optim2);

	// Create the network and add the three layers
	auto network = Network<dtype>();
	network.addLayers({layer1, layer2, layer3});

	// Create the input data and labels for the neural network

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

	// Compile the network
	network.compile();

	// Train the network
	std::cout << "Train\n";
	START_TIMER(0, 2000);
	auto index = random<int>(0, 3);
	network.backward(input[index], output[index]);
	END_TIMER(0);

	// Generate predictions from the network (i.e. use it)
	std::cout << "Predict\n";
	for (int i = 0; i < 4; i++)
		std::cout << (int) input[i][0] << "^" << (int) input[i][1] << " => "
		<< network.forward(input[i])[0][0] << " (Correct: " << output[i][0] << ")" << "\n\n";

	return 0;
}
