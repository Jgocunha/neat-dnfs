// This is a personal academic project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: https://pvs-studio.com

#include <dnf_composer/tools/logger.h>

#include "application/application.h"
#include "solutions/color_space_map_stabilized.h"

int main(int argc, char* argv[])
{
	try
	{
		using namespace neat_dnfs;
		
		dnf_composer::tools::logger::Logger::setMinLogLevel(dnf_composer::tools::logger::LogLevel::WARNING);

		ColorSpaceMapStabilizedSolution solution{
			SolutionTopology{ {
				{FieldGeneType::INPUT, {360, 1.0}},
				{FieldGeneType::OUTPUT, {100, 1.0}}
			}}
		};
		const PopulationParameters parameters{ 1000, 1000, 0.8 };
		Population population{ parameters, std::make_shared<ColorSpaceMapStabilizedSolution>(solution) };

		const Application app(std::make_shared<Population>(population));
		app.init();

		do
		{
			app.step();
			Sleep(10);
		} while (!app.hasGUIBeenClosed());

		app.close();

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
