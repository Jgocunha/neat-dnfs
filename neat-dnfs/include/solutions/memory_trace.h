#pragma once

#include "neat/solution.h"
#include "neat_tools/utils.h"

namespace neat_dnfs
{
	class MemoryTrace final : public Solution
	{
	public:
		explicit MemoryTrace(const SolutionTopology& topology);
		MemoryTrace(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype);
		SolutionPtr clone() const override;
		SolutionPtr copy() const override;
	private:
		void testPhenotype() override;
		void createPhenotypeEnvironment() override;
	};
}


