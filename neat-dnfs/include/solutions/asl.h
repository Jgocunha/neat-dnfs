#pragma once

#include "neat/solution.h"
#include "tools/utils.h"

namespace neat_dnfs
{
	class ActionSimulationLayerSolution : public Solution
	{
	public:
		ActionSimulationLayerSolution(const SolutionTopology& topology);
		SolutionPtr clone() const override;
	private:
		void testPhenotype() override;
		void updateFitness() override;
		bool isHighestActivationOfFieldEqualTo(const std::string& field, 
			double target_u) const;
		void evaluateConditionWithCoincidentalStimulus();
		void evaluateConditionWithNonCoincidentalStimulus();
		double fitnessCoincidentalStimulus() const;
		double fitnessNonCoincidentalStimulus() const;
	};
}