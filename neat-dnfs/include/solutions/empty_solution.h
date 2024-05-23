#pragma once

#include "neat/solution.h"
#include "tools/utils.h"

namespace neat_dnfs
{
	class EmptySolution : public Solution
	{
	public:
		EmptySolution(const SolutionTopology& topology);
		SolutionPtr clone() const override;
	private:
		void updateFitness() override;
		void testPhenotype() override;
	};
}