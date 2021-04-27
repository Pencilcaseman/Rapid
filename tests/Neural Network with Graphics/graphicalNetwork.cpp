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
	using dtype = double;

	// Create the neural network config
	NetworkConfig<dtype> config{
		{{"x1", 1},       // Input 1 = one node
		 {"x2", 1}},      // Input 2 = one node

		{{"y", 2}},       // Output = two nodes

		{5, 5, 5},        // Hidden layer nodes

		{"LeakyRelu"},    // Activation functions
		{"SGD"},          // Optimizers
		{0.01}            // Learning rates
	};

	// Create the network from the config
	auto network = Network<dtype>(config);

	// Create the input data and labels for the network.
	// These are created using the names specified in
	// the config for the network

	// Inputs for XOR and OR
	std::vector<NetworkInput<dtype>> input = {
		{{"x1", fromScalar<dtype>(0)},
		 {"x2", fromScalar<dtype>(0)}},

		{{"x1", fromScalar<dtype>(0)},
		 {"x2", fromScalar<dtype>(1)}},

		{{"x1", fromScalar<dtype>(1)},
		 {"x2", fromScalar<dtype>(0)}},

		{{"x1", fromScalar<dtype>(1)},
		 {"x2", fromScalar<dtype>(1)}}
	};

	// Targets for XOR and OR
	std::vector<NetworkOutput<dtype>> output = {
		{{"y", fromData<dtype>({0, 0})}},

		{{"y", fromData<dtype>({1, 1})}},

		{{"y", fromData<dtype>({1, 1})}},

		{{"y", fromData<dtype>({0, 1})}}
	};

	// Add the data to the network
	network.addData(input, output);
	network.record("loss");

	// Compile the network
	network.compile();

	NetVis netvis(&network, TrainConfig(-1, 1000000));
	netvis.start();

	// Test the accuracy of the network by printing
	// it's output compared to the labeled output
	std::cout << "Predict\n";
	for (int i = 0; i < 4; i++)
	{
		std::cout << "Calculations: ";
		std::cout << (int) input[i]["x1"] << "^" << (int) input[i]["x2"] << ", ";
		std::cout << (int) input[i]["x1"] << "|" << (int) input[i]["x2"];

		std::cout << " => ";

		auto netOut = network.forward(input[i]);
		std::cout << round((dtype) netOut["y"][0][0], 3) << ", ";
		std::cout << round((dtype) netOut["y"][1][0], 3);

		std::cout << " (3 s.f.)  | Correct = " << output[i]["y"] << " |\n";
	}

	return 0;
}
