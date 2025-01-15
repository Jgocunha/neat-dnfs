#pragma once

#include "neat/solution.h"
#include "tools/utils.h"

namespace neat_dnfs
{
	class SelectiveOutputSolution : public Solution
	{
	public:
		SelectiveOutputSolution(const SolutionTopology& topology);
		SolutionPtr clone() const override;
	private:
		void testPhenotype() override;
		void createPhenotypeEnvironment() override;
	};
}