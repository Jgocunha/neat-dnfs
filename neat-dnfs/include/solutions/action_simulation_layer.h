#pragma once

#include "neat/solution.h"
#include "tools/utils.h"

namespace neat_dnfs
{
	class ActionSimulationSolution : public Solution
	{
	public:
		ActionSimulationSolution(const SolutionTopology& topology);
		SolutionPtr clone() const override;
	private:
		void testPhenotype() override;
		void updateFitness() override {}
	};
}