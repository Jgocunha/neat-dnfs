#pragma once

#include "solution.h"


namespace neat_dnfs
{
	class TemplateSolution : public Solution
	{
	public:
		TemplateSolution(const SolutionParameters& parameters);
		~TemplateSolution() override = default;

		void evaluate() override;
		void mutate() override;
		std::shared_ptr<Solution> clone() const override;
	private:
		void buildPhenotype() override;
		void evaluatePhenotype() override; 
		void createRandomInitialConnectionGenes() override;
	};
}