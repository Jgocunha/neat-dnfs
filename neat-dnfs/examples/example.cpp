
// This is a personal academic project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: https://pvs-studio.com

#include <exception>
#include <iostream>

#include "population.h"

int main(int argc, char* argv[])
{
	try
	{
		using namespace neat_dnfs;
		Population population(10);
		// number of output genes should be just 1
		population.initialize(1, 1, 1);
		population.evaluate();

		return 0;
	}
	catch (const std::exception& ex)
	{
		std::cerr << ex.what() << std::endl;
		return 1;
	}
	catch (...)
	{
		std::cerr << "Unknown exception" << std::endl;
		return 1;
	}
}
