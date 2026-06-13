 // This is a personal academic project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: https://pvs-studio.com

#include <exception>
#include <iostream>
#include "dnf_composer/application/application.h"
#include <dnf_composer/simulation/simulation_file_manager.h>
#include <dnf_composer/tools/logger.h>

#include "neat/population.h"
#include "neat_tools/logger.h"
#include "solutions/detection_instability.h"
#include "solutions/memory_instability.h"
#include "solutions/and.h"
#include "solutions/selection_instability.h"
#include "solutions/memory_trace.h"
#include "solutions/xor.h"
#include "solutions/delayed_match_to_sample.h"
#include "solutions/inhibition_of_return.h"

 int main(int argc, char* argv[])
{
	try
	{
		dnf_composer::tools::logger::Logger::setMinLogLevel(dnf_composer::tools::logger::LogLevel::ERROR);
		using namespace neat_dnfs;

		// load a previous solution
		const auto previous_solution = std::make_shared<dnf_composer::Simulation>();
		const dnf_composer::SimulationFileManager sfm(previous_solution,
			std::string(PROJECT_DIR) + "/templates/inhibition-of-return.json");
			//std::string(PROJECT_DIR) + "/data/Delayed Match to Sample/2026-01-13 21h39m07s/best_solutions/last_generation/solution 16039 generation 81 species 140 fitness 0.964459.json");
		sfm.loadElementsFromJson();
		const dnf_composer::Simulation& template_solution = *previous_solution;

		// select the type of solution here
		InhibitionOfReturn solution{
			SolutionTopology{ {
				{FieldGeneType::INPUT, dnf_composer::element::ElementDimensions{DimensionConstants::xSize, DimensionConstants::dx}},
				//	{FieldGeneType::INPUT, dnf_composer::element::ElementDimensions{DimensionConstants::xSize, DimensionConstants::dx}},
				//{FieldGeneType::INPUT, dnf_composer::element::ElementDimensions{DimensionConstants::xSize, DimensionConstants::dx}},
				{FieldGeneType::OUTPUT, dnf_composer::element::ElementDimensions{DimensionConstants::xSize, DimensionConstants::dx}},
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
