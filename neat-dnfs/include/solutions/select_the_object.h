#pragma once

#include "neat/solution.h"
#include "tools/utils.h"

namespace neat_dnfs
{
	class SelectTheObject : public Solution
	{
	public:
		SelectTheObject(const SolutionTopology& topology);
		SolutionPtr clone() const override;
	private:
		void testPhenotype() override;
		void createPhenotypeEnvironment() override;
	};
}