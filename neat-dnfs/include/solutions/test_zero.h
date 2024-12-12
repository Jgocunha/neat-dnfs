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
			{0.0, 75.0},
			{120.0, 50.0},
			{240.0, 25.0},
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