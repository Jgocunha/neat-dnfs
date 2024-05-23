#pragma once

#include "neat/solution.h"
#include "tools/utils.h"

#include <user_interface/simulation_window.h>
#include <user_interface/element_window.h>
#include <user_interface/field_metrics_window.h>
#include <user_interface/plot_window.h>


namespace neat_dnfs
{
	class SingleBumpSolution : public Solution
	{
	public:
		SingleBumpSolution(const SolutionTopology& topology);

		void evaluate() override;
		SolutionPtr clone() const override;

		SolutionPtr crossover(const SolutionPtr& other) override;
	private:
		void testPhenotype();
		void updateFitness();
		void runSimulation();
		void addStimulus(const std::string& targetElement, const double& position);
		void removeStimulus();
	};
}