#pragma once

#include "neat/solution.h"
#include "tools/utils.h"

namespace neat_dnfs
{
	class TestZeroSolution : public Solution
	{
	private:
		std::vector<std::vector<double>> inputOutputBumps =
		{
			{25.0, 37.5},
			{50.0, 62.5},
			//{75.0, 87.5},
			//{0.0, 12.5},
		};
	public:
		TestZeroSolution(const SolutionTopology& topology);
		SolutionPtr clone() const override;
	private:
		void updateFitness() override;
		void testPhenotype() override;
		void learning();
		void testing();
	};
}