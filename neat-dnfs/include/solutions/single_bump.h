#pragma once

#include "neat/solution.h"
#include "tools/utils.h"

namespace neat_dnfs
{
	struct Analysis
	{
		double previousFitness = 0.0;
		dnf_composer::element::NeuralFieldBump previousBump = {};
		Genome previousGenome = {};
		dnf_composer::element::GaussKernelParameters pgkps = {};
	};


	class SingleBumpSolution : public Solution
	{
	public:
		Analysis analysis;

		SingleBumpSolution(const SolutionTopology& topology);
		SolutionPtr clone() const override;
	private:
		void testPhenotype() override;
		void updateFitness() override;
	};
}