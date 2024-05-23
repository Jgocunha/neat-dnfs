#pragma once

#include "neat/solution.h"
#include "tools/utils.h"




namespace neat_dnfs
{
	class SelectionSolution : public Solution
	{
	public:
		SelectionSolution(const SolutionTopology& topology);

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