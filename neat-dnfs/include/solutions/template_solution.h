#pragma once

#include "neat/solution.h"


namespace neat_dnfs
{
	class TemplateSolution : public Solution
	{
	public:
		TemplateSolution(const SolutionTopology& initialTopology);

		void evaluate() override;
		SolutionPtr clone() const override;
	private:
		void evaluatePhenotype() override; 
		void createRandomInitialConnectionGenes() override;
	};
}