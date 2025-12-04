#pragma once

#include "neat/solution.h"
#include "tools/utils.h"

namespace neat_dnfs
{
	class SelectiveOutputSolution final : public Solution
	{
	public:
		explicit SelectiveOutputSolution(const SolutionTopology& topology);
		SelectiveOutputSolution(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype);
		SolutionPtr clone() const override;
	private:
		void testPhenotype() override;
		void createPhenotypeEnvironment() override;
	};
}