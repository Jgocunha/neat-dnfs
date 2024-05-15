// This is a personal academic project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: https://pvs-studio.com

#include <exception>
#include <iostream>

#include "neat/population.h"
#include "solutions/template_solution.h"

int main(int argc, char* argv[])
{
	try
	{
		using namespace neat_dnfs;

		TemplateSolution solution{ SolutionTopology{2, 1} };
		const PopulationParameters parameters{ 10, 1000, 0.95 };
		Population population{ parameters, std::make_shared<TemplateSolution>(solution) };

		population.initialize();

		population.evolve();

		const auto bestSolution = population.getBestSolution();
		const auto phenotype = bestSolution->getPhenotype();

		return 0;
	}
	catch (const dnf_composer::Exception& ex)
	{
		log(neat_dnfs::tools::logger::LogLevel::FATAL, "Exception caught: " + std::string(ex.what()) + ".");
		return static_cast<int>(ex.getErrorCode());
	}
	catch (const std::exception& ex)
	{
		log(neat_dnfs::tools::logger::LogLevel::FATAL, "Exception caught: " + std::string(ex.what()) + ".");
		return 1;
	}
	catch (...)
	{
		log(neat_dnfs::tools::logger::LogLevel::FATAL, "Unknown exception occurred.");
		return 1;
	}
}
