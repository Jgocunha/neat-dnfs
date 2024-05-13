#pragma once

#include <thread>
#include <imgui-platform-kit/user_interface_window.h>
#include <user_interface/simulation_window.h>
#include <user_interface/plot_window.h>
#include <user_interface/element_window.h>
#include <user_interface/field_metrics_window.h>

#include "neat/population.h"
#include "solutions/template_solution.h"

namespace neat_dnfs
{
	class PopulationControlWindow : public imgui_kit::UserInterfaceWindow
	{
	private:
		std::shared_ptr<std::thread> evolveThread;
		std::shared_ptr<std::thread> simulationThread;
		std::shared_ptr<dnf_composer::user_interface::SimulationWindow> simulationWindow;
		std::shared_ptr<dnf_composer::user_interface::ElementWindow> elementWindow;
		std::shared_ptr<dnf_composer::user_interface::FieldMetricsWindow> centroidMonitoringWindow;
		std::vector<std::shared_ptr<dnf_composer::user_interface::PlotWindow>> plotWindows;
		std::shared_ptr<Population> population;
		PopulationParameters populationParameters;
		SolutionTopology solutionParameters;
	public:
		PopulationControlWindow(const std::shared_ptr<Population>& population);
		void render() override;
		~PopulationControlWindow() override;
	private:
		void renderPopulationInfo();
		void renderSolutionInfo();
		void renderPopulationMethods();
		void renderShowBestSolution();

		void runSimulation(const std::shared_ptr<dnf_composer::Simulation>& simulation);
	};
}
