#pragma once

#include <user_interface/simulation_window.h>
#include <user_interface/element_window.h>
#include <user_interface/field_metrics_window.h>
#include <user_interface/plot_window.h>

namespace neat_dnfs
{
	class SolutionWindow
	{
	private:
		std::shared_ptr<dnf_composer::Simulation> simulation;
		std::shared_ptr<dnf_composer::user_interface::SimulationWindow> simulationWindow;
		std::shared_ptr<dnf_composer::user_interface::ElementWindow> elementWindow;
		std::shared_ptr<dnf_composer::user_interface::FieldMetricsWindow> fieldMetricsWindow;
		std::vector<std::shared_ptr<dnf_composer::user_interface::PlotWindow>> plotWindows;
	public:
		SolutionWindow(const std::shared_ptr<dnf_composer::Simulation>& simulation);
	};
}