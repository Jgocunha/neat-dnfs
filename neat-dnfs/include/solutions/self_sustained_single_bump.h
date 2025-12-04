#pragma once

#include "neat/solution.h"
#include "tools/utils.h"

namespace neat_dnfs
{
	class SelfSustainedSingleBumpSolution final : public Solution
	{
	public:
		explicit SelfSustainedSingleBumpSolution(const SolutionTopology& topology);
		SelfSustainedSingleBumpSolution(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype);
		SolutionPtr clone() const override;
	private:
		void testPhenotype() override;
		void createPhenotypeEnvironment() override;
	};
}