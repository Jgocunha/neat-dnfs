#pragma once

#include "neat/solution.h"
#include "tools/utils.h"

namespace neat_dnfs
{
	class SelfSustainedSingleBumpSolution : public Solution
	{
	public:
		SelfSustainedSingleBumpSolution(const SolutionTopology& topology);
		SolutionPtr clone() const override;
	private:
		void testPhenotype() override;
		void updateFitness() override;
		double selfStabilityFitness() const;
		double selfSustainabilityFitness() const;
		bool isHighestActivationOfFieldEqualTo(const std::string& field, double target_u) const;
	};
}