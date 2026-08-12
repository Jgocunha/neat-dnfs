 // This is a personal academic project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: https://pvs-studio.com

#include <exception>
#include <iostream>
#include "dnf_composer/application/application.h"
#include <dnf_composer/tools/logger.h>

#include "neat/population.h"
#include "tools/logger.h"
#include "solutions/hri_packaging_task.h"

int main(int argc, char* argv[])
{
	try
	{
		dnf_composer::tools::logger::Logger::setMinLogLevel(dnf_composer::tools::logger::LogLevel::ERROR);
		using namespace neat_dnfs;

		// Condition: no growth, I/O only. All legal connections seeded at init,
		// field/connection/toggle mutations disabled. Only parameters evolve.
		AblationConstants::label = " No Growth IO Only";
		AblationConstants::disableAddFieldGene = true;
		AblationConstants::disableAddConnectionGene = true;
		AblationConstants::disableToggleConnectionGene = true;
		AblationConstants::seedAllLegalConnections = true;

		// select the type of solution here and in the population init.
		const dnf_composer::element::ElementDimensions dims{ DimensionConstants::xSize, DimensionConstants::dx };
		HRIPackagingTask solution{
			SolutionTopology{ {
				{FieldGeneType::INPUT, dims},
				{FieldGeneType::INPUT, dims},
				{FieldGeneType::INPUT, dims},
				{FieldGeneType::OUTPUT, dims},
			}
			},
		};

		constexpr size_t number_runs = 30;

		for (int i = 0; i < number_runs; i++)
		{
			constexpr size_t population_size	= 1000;
			constexpr size_t number_generations = 200;
			constexpr double target_fitness		= 0.95;

			const PopulationParameters parameters{ population_size, number_generations, target_fitness };
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
