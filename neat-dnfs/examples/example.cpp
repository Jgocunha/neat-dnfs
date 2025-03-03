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
#include <user_interface/field_metrics_window.h>

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

#define _CRTDBG_MAP_ALLOC
#include <cstdlib>
#include <crtdbg.h>


int main(int argc, char* argv[])
{
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	try
	{
		dnf_composer::tools::logger::Logger::setMinLogLevel(dnf_composer::tools::logger::LogLevel::ERROR);
		using namespace neat_dnfs;

		for (int i = 0; i < 1000; i++)
		{

			SelectTheObject solution{
				SolutionTopology{ {
					{FieldGeneType::INPUT, {DimensionConstants::xSize, DimensionConstants::dx}},
					{FieldGeneType::INPUT, {DimensionConstants::xSize, DimensionConstants::dx}},
					{FieldGeneType::INPUT, {DimensionConstants::xSize, DimensionConstants::dx}},
					{FieldGeneType::OUTPUT, {DimensionConstants::xSize, DimensionConstants::dx}}
				}}
			};
			const PopulationParameters parameters{ 100, 200, 0.95 };
			Population population{ parameters, std::make_shared<SelectTheObject>(solution) };

			population.initialize();
			population.evolve();
		}

		//const auto bestSolution = population.getBestSolution();
		//bestSolution->createPhenotypeEnvironment();
		//bestSolution->print();
		//const auto phenotype = bestSolution->getPhenotype();
		//
		//// run dnf-composer
		//using namespace dnf_composer;
		//const Application app{ phenotype };
		//app.addWindow<user_interface::MainWindow>();
		//app.addWindow<user_interface::SimulationWindow>();
		//app.addWindow<user_interface::ElementWindow>();
		//app.addWindow<imgui_kit::LogWindow>();
		//app.addWindow<user_interface::PlotControlWindow>();
		//app.addWindow<user_interface::PlotsWindow>();
		//app.addWindow<user_interface::NodeGraphWindow>();
		//app.addWindow<user_interface::FieldMetricsWindow>();
		//app.init();
		//do
		//{
		//	app.step();
		//} while(!app.hasGUIBeenClosed());
		//app.close();

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
