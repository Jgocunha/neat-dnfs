#pragma once

#include "neat/solution.h"
#include "neat_tools/utils.h"

namespace neat_dnfs
{
	class EmptySolution : public Solution
	{
	public:
		EmptySolution(const SolutionTopology& topology);
		SolutionPtr clone() const override;
		SolutionPtr copy() const override;
	private:
		void testPhenotype() override;
		void updateFitness();
		void createPhenotypeEnvironment() override {}
	};
}