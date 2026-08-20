#include <exception>
#include <iostream>
#include <dnf_composer/application/application.h>
#include <dnf_composer/tools/logger.h>

#include "neat/population.h"
#include "neat_tools/logger.h"
#include "neat_tools/key_listener.h"
#include "solutions/delayed_match_to_sample.h"

 int main(int argc, char* argv[])
{
	try
	{
		dnf_composer::tools::logger::Logger::setMinLogLevel(dnf_composer::tools::logger::LogLevel::ERROR);
		using namespace neat_dnfs;

		// This experiment is tuned for a 360-wide field: the sample sits at
		// position 50 and the second sample at position 100. Set
		// DimensionConstants::xSize to 360 in include/constants.h before running
		// it. The default is 100 -- this leaves the second sample outside the
		// field and makes evaluate() throw.
		DelayedMatchToSample solution{
			SolutionTopology{ {
				{FieldGeneType::INPUT, dnf_composer::element::ElementDimensions{DimensionConstants::xSize, DimensionConstants::dx}},
				{FieldGeneType::OUTPUT, dnf_composer::element::ElementDimensions{DimensionConstants::xSize, DimensionConstants::dx}},
			}
			},
		};

		constexpr size_t number_runs = 50;

		for (int i = 0; i < number_runs; i++)
		{
			constexpr size_t population_size	= 1000;
			constexpr size_t number_generations = 200;
			constexpr double target_fitness		= 0.95;

			const PopulationParameters parameters{ population_size, number_generations, target_fitness };
			Population population{ parameters, std::make_unique<DelayedMatchToSample>(solution) };

			population.initialize();
			KeyListener keyListener{ population };
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
