 // This is a personal academic project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: https://pvs-studio.com

#include <exception>
#include <iostream>
#include "dnf_composer/application/application.h"
#include <dnf_composer/user_interface/element_window.h>
#include <dnf_composer/user_interface/main_window.h>
#include <dnf_composer/user_interface/plot_control_window.h>
#include <dnf_composer/user_interface/simulation_window.h>
#include <dnf_composer/tools/logger.h>
#include <user_interface/node_graph_window.h>
#include <user_interface/plots_window.h>

#include "neat/population.h"
#include "solutions/test_zero.h"
#include "solutions/test_one.h"
#include "solutions/color_space_map_stabilized.h"
#include "solutions/color_space_map_in_sustained.h"
#include "solutions/color_space_map_out_sustained.h"

int main(int argc, char* argv[])
{
	try
	{
		dnf_composer::tools::logger::Logger::setMinLogLevel(dnf_composer::tools::logger::LogLevel::WARNING);
		using namespace neat_dnfs;

		//TestZeroSolution solution{ SolutionTopology{ {{FieldGeneType::INPUT, {50, 1.0}}, {FieldGeneType::OUTPUT, {100, 1.0}} } } };
		//TestOneSolution solution{ SolutionTopology{ {{FieldGeneType::INPUT, {360, 1.0}}, {FieldGeneType::OUTPUT, {100, 1.0}} } } };
		//ColorSpaceMapStabilizedSolution solution{
		//	SolutionTopology{ {
		//		{FieldGeneType::INPUT, {360, 1.0}},
		//		{FieldGeneType::OUTPUT, {100, 1.0}}
		//	}}
		//};
		ColorSpaceMapOutputSustainedSolution solution{
			SolutionTopology{ {
				{FieldGeneType::INPUT, {360, 1.0}},
				{FieldGeneType::OUTPUT, {100, 1.0}}
			}}
		};
		/*ColorSpaceMapInputSustainedSolution solution{
			SolutionTopology{ {
				{FieldGeneType::INPUT, {360, 1.0}},
				{FieldGeneType::OUTPUT, {100, 1.0}}
			}}
		};*/

		const PopulationParameters parameters{ 100, 1000, 0.9 };
		Population population{ parameters, std::make_shared<ColorSpaceMapOutputSustainedSolution>(solution) };

		population.initialize();
		population.evolve();

		const auto bestSolution = population.getBestSolution();
		const auto phenotype = bestSolution->getPhenotype();

		// run dnf-composer
		using namespace dnf_composer;
		Application app{ std::make_shared<Simulation>(phenotype) };
		app.addWindow<user_interface::MainWindow>();
		app.addWindow<user_interface::SimulationWindow>();
		app.addWindow<user_interface::ElementWindow>();
		app.addWindow<imgui_kit::LogWindow>();
		app.addWindow<user_interface::PlotControlWindow>();
		app.addWindow<user_interface::PlotsWindow>();
		app.addWindow<user_interface::NodeGraphWindow>();
		app.init();
		do
		{
			app.step();
		} while(!app.hasGUIBeenClosed());
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
