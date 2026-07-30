#include <exception>
#include <iostream>
#include <dnf_composer/application/application.h>
#include <dnf_composer/tools/logger.h>

#include "neat/population.h"
#include "neat_tools/logger.h"
#include "neat_tools/key_listener.h"
#include "solutions/selection_instability.h"

 int main(int argc, char* argv[])
{
	try
	{
		dnf_composer::tools::logger::Logger::setMinLogLevel(dnf_composer::tools::logger::LogLevel::ERROR);
		using namespace neat_dnfs;

		SelectionInstability solution{
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
			Population population{ parameters, std::make_unique<SelectionInstability>(solution) };

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
