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
#include "solutions/delayed_match_to_sample.h"
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
		const auto previous_solution = std::make_shared<dnf_composer::Simulation>();
		const dnf_composer::SimulationFileManager sfm(previous_solution,
			std::string(PROJECT_DIR) + "/templates/test-dmts.json");
		sfm.loadElementsFromJson();
		const dnf_composer::Simulation& template_solution = *previous_solution;

		// select the type of solution here and in the population init.
		const dnf_composer::element::ElementDimensions dims{ DimensionConstants::xSize, DimensionConstants::dx };
		DelayedMatchToSample solution{
			SolutionTopology{ {
				{FieldGeneType::INPUT, dims},
				//{FieldGeneType::INPUT, dims},
				//{FieldGeneType::INPUT, dims},
				{FieldGeneType::OUTPUT, dims},
				//{FieldGeneType::OUTPUT, dims},
			}
			},
			template_solution // load a previous solution
		};

		constexpr size_t number_runs = 100;

		for (int i = 0; i < number_runs; i++)
		{
			constexpr size_t population_size	= 200;
			constexpr size_t number_generations = 100;
			constexpr double target_fitness		= 0.9;

			const PopulationParameters parameters{ population_size, number_generations, target_fitness };
			Population population{ parameters, std::make_unique<DelayedMatchToSample>(solution) };

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
