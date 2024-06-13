// This is a personal academic project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: https://pvs-studio.com

#include <exception>
#include <iostream>

#include "neat/population.h"
#include "solutions/self_sustained_single_bump.h"

int main(int argc, char* argv[])
{
	try
	{
		using namespace neat_dnfs;

		SelfSustainedSingleBumpSolution solution{ SolutionTopology{1, 1} };
		const PopulationParameters parameters{ 100, 1000, 0.90 };
		Population population{ parameters, std::make_shared<SelfSustainedSingleBumpSolution>(solution) };

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
