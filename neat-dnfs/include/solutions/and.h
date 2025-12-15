#pragma once

#include "neat/solution.h"
#include "tools/utils.h"

namespace neat_dnfs
{
	class AND final : public Solution
	{
	public:
		explicit AND(const SolutionTopology& topology);
		AND(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype);
		SolutionPtr clone() const override;
	private:
		void testPhenotype() override;
		void createPhenotypeEnvironment() override;
	};
}