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

	// Create the neural network config
	Config<dtype> config{
		{{"x1", 1},       // Input 1
		 {"x2", 1}},      // Input 2

		{{"y", 1}},       // Output

		{3},              // Hidden layer nodes

		{"LeakyRelu"},    // Activation functions
		{"ADAM"},         // Optimizers
		{0.1}             // Learning rates
	};

	// Create the network from the config
	auto network = Network<dtype>(config);

	// Create the input data and labels for the network.
	// These are created using the names specified in
	// the config for the network

	// Inputs for XOR
	std::vector<NetworkInput<float>> input = {
		{{"x1", fromScalar<float>(0)},
		 {"x2", fromScalar<float>(0)}},

		{{"x1", fromScalar<float>(0)},
		 {"x2", fromScalar<float>(1)}},

		{{"x1", fromScalar<float>(1)},
		 {"x2", fromScalar<float>(0)}},

		{{"x1", fromScalar<float>(1)},
		 {"x2", fromScalar<float>(1)}}
	};

	// Targets for XOR
	std::vector<NetworkOutput<float>> output = {
		{{"y", fromScalar<float>(0)}},

		{{"y", fromScalar<float>(1)}},

		{{"y", fromScalar<float>(1)}},

		{{"y", fromScalar<float>(0)}}
	};

	// Compile the network
	network.compile();

	// Train the network
	std::cout << "Train\n";
	START_TIMER(0, 5000);
	auto index = random<int>(0, 3);
	network.backward(input[index], output[index]);
	END_TIMER(0);

	// Test the accuracy of the network by printing
	// it's output compared to the labeled output
	std::cout << "Predict\n";
	for (int i = 0; i < 4; i++)
		std::cout << (int) input[i]["x1"] << "^" << (int) input[i]["x1"] << " => "
		<< network.forward(input[i])["y"][0][0] << " (Correct: " << output[i]["y"] << ")" << "\n\n";

	return 0;
}
