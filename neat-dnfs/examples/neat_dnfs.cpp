 // This is a personal academic project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: https://pvs-studio.com

#include <exception>
#include <iostream>
#include "dnf_composer/application/application.h"
#include <dnf_composer/tools/logger.h>

#include "neat/population.h"
#include "tools/logger.h"
#include "solutions/detection_instability.h"
#include "solutions/memory_instability.h"
#include "solutions/and.h"
#include "solutions/selection_instability.h"
#include "solutions/memory_trace.h"
#include "solutions/xor.h"

 int main(int argc, char* argv[])
{
	try
	{
		dnf_composer::tools::logger::Logger::setMinLogLevel(dnf_composer::tools::logger::LogLevel::ERROR);
		using namespace neat_dnfs;

		// load a previous solution
		//const std::shared_ptr<dnf_composer::Simulation> previous_solution = std::make_shared<dnf_composer::Simulation>();
		//const dnf_composer::SimulationFileManager sfm(previous_solution, std::string(PROJECT_DIR) + "/solution 36074 generation 182 species 300 fitness 0.924742.json");
		//sfm.loadElementsFromJson();
		//const dnf_composer::Simulation template_solution = *previous_solution;

		// select the type of solution here and in the population init.
		MemoryTrace solution{
			SolutionTopology{ {
				{FieldGeneType::INPUT, {DimensionConstants::xSize, DimensionConstants::dx}},
				//{FieldGeneType::INPUT, {DimensionConstants::xSize, DimensionConstants::dx}},
				//{FieldGeneType::INPUT, {DimensionConstants::xSize, DimensionConstants::dx}},
				{FieldGeneType::OUTPUT, {DimensionConstants::xSize, DimensionConstants::dx}},
				//{FieldGeneType::OUTPUT, {DimensionConstants::xSize, DimensionConstants::dx}},
			}
			},
			//template_solution // load a previous solution
		};

		constexpr size_t number_runs = 100;

		for (int i = 0; i < number_runs; i++)
		{
			constexpr size_t population_size	= 250;
			constexpr size_t number_generations = 150;
			constexpr double target_fitness		= 0.95;

			const PopulationParameters parameters{ population_size, number_generations, target_fitness };
			Population population{ parameters, std::make_unique<MemoryTrace>(solution) };

			population.initialize();
			population.evolve();
		}

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
