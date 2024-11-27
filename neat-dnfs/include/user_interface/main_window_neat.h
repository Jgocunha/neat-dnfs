#pragma once

#include <thread>
#include <windows.h>//Sleep
#include <cstdio>

#include <imgui-platform-kit/user_interface_window.h>
#include <user_interface/simulation_window.h>
#include <user_interface/element_window.h>
#include <user_interface/field_metrics_window.h>
#include <user_interface/node_graph_window.h>
#include <user_interface/plots_window.h>
#include <user_interface/plot_control_window.h>
#include <user_interface/main_window.h>
#include <visualization/visualization.h>

#include "neat/population.h"
#include "solutions/single_bump.h"
#include "solutions/self_sustained_single_bump.h"
#include "solutions/asl.h"

namespace neat_dnfs
{
	class MainWindow : public imgui_kit::UserInterfaceWindow
	{
	private:
		std::shared_ptr<Population> population;
		PopulationParameters populationParameters;
		SolutionTopology solutionParameters;
		SolutionPtr bestSolution;
		std::shared_ptr<std::thread> evolveThread;
		std::shared_ptr<std::thread> simulationThread;
		std::shared_ptr<dnf_composer::Simulation> simulation;
		std::shared_ptr<dnf_composer::Visualization> visualization;
		std::shared_ptr<dnf_composer::user_interface::MainWindow> mainWindow;
		std::shared_ptr<dnf_composer::user_interface::SimulationWindow> simulationWindow;
		std::shared_ptr<dnf_composer::user_interface::ElementWindow> elementWindow;
		std::shared_ptr<dnf_composer::user_interface::FieldMetricsWindow> fieldMetricsWindow;
		std::shared_ptr<dnf_composer::user_interface::NodeGraphWindow> nodeGraphWindow;
		std::shared_ptr<dnf_composer::user_interface::PlotsWindow> plotsWindow;
		std::shared_ptr < dnf_composer::user_interface::PlotControlWindow > plotControlWindow;
	public:
		MainWindow();
		void render() override;
		~MainWindow() override;
	private:
		void renderPopulationInfo();
		void renderSolutionInfo();
		void renderPopulationMethods();
		void renderSolutionMethods();
		void renderShowBestSolution();
	};
}
