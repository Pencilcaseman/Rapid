#include <iostream>
#include <rapid.h>

int main()
{
	auto optim1 = new rapid::network::optimizers::SGD<double>();
	std::cout << optim1->getParam("momentum") << "\n";

	delete optim1;

	return 0;
}
