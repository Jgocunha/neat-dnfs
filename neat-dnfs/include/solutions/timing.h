#pragma once

#include "neat/solution.h"
#include "tools/utils.h"

namespace neat_dnfs
{
	class Timing final : public Solution
	{
	public:
		explicit Timing(const SolutionTopology& topology);
		Timing(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype);
		SolutionPtr clone() const override;
	private:
		void testPhenotype() override;
		void createPhenotypeEnvironment() override;
	};
}


