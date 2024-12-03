#pragma once

#include <dnf_composer/user_interface/simulation_window.h>
#include <dnf_composer/user_interface/element_window.h>
#include <dnf_composer/user_interface/field_metrics_window.h>

namespace neat_dnfs
{
	class SolutionWindow
	{
	private:
		std::shared_ptr<dnf_composer::Simulation> simulation;
		std::shared_ptr<dnf_composer::user_interface::SimulationWindow> simulationWindow;
		std::shared_ptr<dnf_composer::user_interface::ElementWindow> elementWindow;
		std::shared_ptr<dnf_composer::user_interface::FieldMetricsWindow> fieldMetricsWindow;
	public:
		SolutionWindow(const std::shared_ptr<dnf_composer::Simulation>& simulation);
	};
}