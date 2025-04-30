#pragma once

#include "neat/solution.h"
#include "tools/utils.h"

namespace neat_dnfs
{
	class XOR : public Solution
	{
	public:
		XOR(const SolutionTopology& topology);
		XOR(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype);

		SolutionPtr clone() const override;
	private:
		void testPhenotype() override;
		void createPhenotypeEnvironment() override;
	};
}