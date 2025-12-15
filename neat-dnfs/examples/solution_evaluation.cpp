 // This is a personal academic project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: https://pvs-studio.com

#include <exception>
#include <iostream>
#include "dnf_composer/application/application.h"
#include <dnf_composer/tools/logger.h>

#include "neat/population.h"
#include "tools/logger.h"
#include "solutions/propagation.h"
#include "solutions/memory.h"
#include "solutions/and.h"
#include "solutions/action_simulation_layer.h"
#include "solutions/action_execution_layer.h"
#include "solutions/selection.h"
#include "solutions/timing.h"
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
		const auto previous_solution = std::make_shared<dnf_composer::Simulation>();
		const dnf_composer::SimulationFileManager sfm(previous_solution,
			std::string(PROJECT_DIR) + "/templates/xor.json");
		sfm.loadElementsFromJson();
		const dnf_composer::Simulation& template_solution = *previous_solution;

		// select the type of solution here
		XOR solution{
			SolutionTopology{ {
				{FieldGeneType::INPUT, {DimensionConstants::xSize, DimensionConstants::dx}},
				{FieldGeneType::INPUT, {DimensionConstants::xSize, DimensionConstants::dx}},
				//{FieldGeneType::INPUT, {DimensionConstants::xSize, DimensionConstants::dx}},
				{FieldGeneType::OUTPUT, {DimensionConstants::xSize, DimensionConstants::dx}},
			}
			},
			template_solution // load a previous solution
		};

		constexpr size_t number_evaluations = 20;
		for (size_t i = 0; i < number_evaluations; i++)
		{
			solution.evaluate();
			solution.print();
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
