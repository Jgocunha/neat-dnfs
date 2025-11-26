 // This is a personal academic project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: https://pvs-studio.com

#include <exception>
#include <iostream>
#include "dnf_composer/application/application.h"
#include <dnf_composer/tools/logger.h>

#include "neat/population.h"
#include "tools/logger.h"
#include "solutions/single_bump.h"
#include "solutions/self_sustained_single_bump.h"
#include "solutions/and.h"
#include "solutions/action_simulation_layer.h"
#include "solutions/action_execution_layer.h"
#include "solutions/selective_output_field.h"
#include "solutions/timing_response.h"
#include "solutions/select_the_object.h"
#include "solutions/two_robot_team.h"
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
		XOR solution{
			SolutionTopology{ {
				{FieldGeneType::INPUT, {DimensionConstants::xSize, DimensionConstants::dx}},
				{FieldGeneType::INPUT, {DimensionConstants::xSize, DimensionConstants::dx}},
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
			constexpr size_t population_size	= 200;
			constexpr size_t number_generations = 250;
			constexpr double target_fitness		= 0.95;

			const PopulationParameters parameters{ population_size, number_generations, target_fitness };
			Population population{ parameters, std::make_unique<XOR>(solution) };

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
