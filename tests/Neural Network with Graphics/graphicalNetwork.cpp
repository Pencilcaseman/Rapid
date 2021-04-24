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

	// Create the neural network config
	NetworkConfig<dtype> config1{
		{{"x1", 1},       // Input 1 = one node
		 {"x2", 1}},      // Input 2 = one node

		{{"y", 2}},       // Output = one node

		{3, 3},           // Hidden layer nodes

		{"LeakyRelu"},    // Activation functions
		{"ADAM"},         // Optimizers
		{0.01}            // Learning rates
	};

	// Create the network from the config
	auto network1 = Network<dtype>(config1);

	// Create the input data and labels for the network.
	// These are created using the names specified in
	// the config for the network

	// Inputs for XOR
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

	// Targets for XOR
	std::vector<NetworkOutput<dtype>> output = {
		{{"y", fromData<dtype>({0, 0})}},

		{{"y", fromData<dtype>({1, 1})}},

		{{"y", fromData<dtype>({1, 1})}},

		{{"y", fromData<dtype>({0, 1})}}
	};

	// Add the data to the network
	network1.addData(input, output);
	network1.record("loss");

	// Compile the network
	// Note: This does not necessarily need to be
	//       called after adding the data, though
	//       in future updates, this may construct
	//       the network to be a particular size
	//       or with different activations if they
	//       have not already been specified
	network1.compile();

	// network1.fit(-1, 2000);
	NetVis netvis(&network1, TrainConfig(-1, 100000));
	netvis.run();

	// Test the accuracy of the network by printing
	// it's output compared to the labeled output
	// std::cout.precision(3);
	std::cout << "Predict\n";
	for (int i = 0; i < 4; i++)
	{
		std::cout << "Input: ";
		std::cout << (int) input[i]["x1"] << "^" << (int) input[i]["x2"] << ", ";
		std::cout << (int) input[i]["x1"] << "|" << (int) input[i]["x2"];

		std::cout << " => ";

		auto netOut = network1.forward(input[i]);
		std::cout << round((dtype) netOut["y"][0][0], 3) << ", ";
		std::cout << round((dtype) netOut["y"][1][0], 3);

		std::cout << " (3 s.f.)  | Correct = " << output[i]["y"] << " |\n";
	}

	return 0;
}
