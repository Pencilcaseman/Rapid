#include <iostream>
#include <rapid.h>

int main()
{
	auto optim1 = new rapid::neural::optimizers::ADAM<float64>();
	std::cout << optim1->getParam("beta2") << "\n";

	delete optim1;

	return 0;
}
