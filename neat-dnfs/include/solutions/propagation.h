#pragma once

#include "neat/solution.h"
#include "tools/utils.h"

namespace neat_dnfs
{
	class Propagation final : public Solution
	{
	public:
		explicit Propagation(const SolutionTopology& topology);
		Propagation(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype);
		SolutionPtr clone() const override;
	private:
		void testPhenotype() override;
		void createPhenotypeEnvironment() override;
	};
}