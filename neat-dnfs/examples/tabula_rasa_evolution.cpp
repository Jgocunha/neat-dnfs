 // This is a personal academic project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: https://pvs-studio.com

#include <exception>
#include <iostream>
#include "dnf_composer/application/application.h"
#include <dnf_composer/tools/logger.h>

#include "neat/population.h"
#include "neat/ablation_presets.h"
#include "tools/logger.h"
#include "solutions/detection_instability.h"
#include "solutions/memory_instability.h"
#include "solutions/and.h"
#include "solutions/delayed_match_to_sample.h"
#include "solutions/selection_instability.h"
#include "solutions/memory_trace.h"
#include "solutions/xor.h"
#include "solutions/hri_packaging_task.h"

 int main(int argc, char* argv[])
{
	try
	{
		dnf_composer::tools::logger::Logger::setMinLogLevel(dnf_composer::tools::logger::LogLevel::ERROR);
		using namespace neat_dnfs;

		// select the type of solution here and in the population init.
		const dnf_composer::element::ElementDimensions dims{ DimensionConstants::xSize, DimensionConstants::dx };
		HRIPackagingTask solution{
			SolutionTopology{ {
				{FieldGeneType::INPUT, dims},
				{FieldGeneType::INPUT, dims},
				{FieldGeneType::INPUT, dims},
				{FieldGeneType::OUTPUT, dims},
				//{FieldGeneType::OUTPUT, dims},
			}
			},
		};

		for (int i = 0; i < AblationProtocol::numberRuns; i++)
		{
			const PopulationParameters parameters{
				AblationProtocol::populationSize, AblationProtocol::numberGenerations, AblationProtocol::targetFitness };
			Population population{ parameters, std::make_unique<HRIPackagingTask>(solution) };

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
