 // This is a personal academic project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: https://pvs-studio.com

#include <exception>
#include <iostream>
#include <dnf_composer/tools/logger.h>

#include "neat/population.h"
#include "solutions/color_space_map_stabilized.h"
//#include "solutions/color_space_map_in_sustained.h"
//#include "solutions/color_space_map_out_sustained.h"

int main(int argc, char* argv[])
{
	try
	{
		dnf_composer::tools::logger::Logger::setMinLogLevel(dnf_composer::tools::logger::LogLevel::FATAL);
		using namespace neat_dnfs;

		//TestZeroSolution solution{ SolutionTopology{ {{FieldGeneType::INPUT, {50, 1.0}}, {FieldGeneType::OUTPUT, {100, 1.0}} } } };
		//TestOneSolution solution{ SolutionTopology{ {{FieldGeneType::INPUT, {360, 1.0}}, {FieldGeneType::OUTPUT, {100, 1.0}} } } };
		ColorSpaceMapStabilizedSolution solution{
			SolutionTopology{ {
				{FieldGeneType::INPUT, {360, 1.0}},
				{FieldGeneType::OUTPUT, {100, 1.0}}
			}}
		};
		// ColorSpaceMapOutputSustainedSolution solution{
		// 	SolutionTopology{ {
		// 		{FieldGeneType::INPUT, {360, 1.0}},
		// 		{FieldGeneType::OUTPUT, {100, 1.0}}
		// 	}}
		// };
		/*ColorSpaceMapInputSustainedSolution solution{
			SolutionTopology{ {
				{FieldGeneType::INPUT, {360, 1.0}},
				{FieldGeneType::OUTPUT, {100, 1.0}}
			}}
		};*/

		static constexpr int runs = 100;

		for (int run = 0; run < runs; ++run)
		{
			const PopulationParameters parameters{ 100, 100, 0.90 };
			Population population{ parameters, std::make_shared<ColorSpaceMapStabilizedSolution>(solution) };

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
