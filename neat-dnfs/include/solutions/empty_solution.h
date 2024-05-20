#pragma once

#include "neat/solution.h"
#include "tools/utils.h"

namespace neat_dnfs
{
	class EmptySolution : public Solution
	{
	public:
		EmptySolution(const SolutionTopology& topology);

		void evaluate() override;
		SolutionPtr clone() const override;

		SolutionPtr crossover(const SolutionPtr& other) override;
	};
}