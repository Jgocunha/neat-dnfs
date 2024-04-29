
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

		const SolutionParameters parameters{3, 1};
		TemplateSolution solution(parameters);
		const Population population(10, std::make_shared<TemplateSolution>(solution));

		population.initialize();
		for (size_t i = 0; i < 100; i++)
		{
			population.evaluate();
			population.evolve();
		}

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
