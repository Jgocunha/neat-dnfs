#pragma once

#include "neat/solution.h"
#include "tools/utils.h"

namespace neat_dnfs
{
	class Memory final : public Solution
	{
	public:
		explicit Memory(const SolutionTopology& topology);
		Memory(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype);
		SolutionPtr clone() const override;
	private:
		void testPhenotype() override;
		void createPhenotypeEnvironment() override;
	};
}