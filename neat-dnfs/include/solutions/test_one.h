#pragma once

#include "neat/solution.h"
#include "tools/utils.h"

namespace neat_dnfs
{
	class TestOneSolution : public Solution
	{
	private:
		std::vector<std::vector<double>> inputOutputBumps =
		{
			{25.0, 37.5}, // 1
			{50.0, 62.5}, // 2
			//{75.0, 87.5}, // 3
			//{0.0, 12.5}, //4
			//{90, 95}, // 5
			//{180, 25}, // 6
			//{270, 10}, // 7
			//{10, 50}, // 8
		};
	public:
		TestOneSolution(const SolutionTopology& topology);
		SolutionPtr clone() const override;
	private:
		void updateFitness() override;
		void testPhenotype() override;
		void learning();
		void testing();
	};
}