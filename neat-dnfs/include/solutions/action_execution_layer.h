#pragma once

#include "neat/solution.h"
#include "tools/utils.h"

namespace neat_dnfs
{
	class ActionExecutionSimulation : public Solution
	{
	public:
		ActionExecutionSimulation(const SolutionTopology& topology);
		SolutionPtr clone() const override;
	private:
		void testPhenotype() override;
		void updateFitness() override {}
	};
}