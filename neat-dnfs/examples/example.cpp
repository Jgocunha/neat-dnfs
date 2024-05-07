
// This is a personal academic project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: https://pvs-studio.com

#include <exception>
#include <iostream>

#include "population.h"
#include "template_solution.h"

int main(int argc, char* argv[])
{
	try
	{
		using namespace neat_dnfs;

		const SolutionTopology initialTopology{3, 1};
		TemplateSolution solution(initialTopology);
		Population population(100, std::make_shared<TemplateSolution>(solution));
		constexpr int numGenerations = 1000;

		population.initialize();
		population.evolve(numGenerations);

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
