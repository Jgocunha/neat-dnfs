#pragma once

#include "solution.h"


namespace neat_dnfs
{
	class TemplateSolution : public Solution
	{
	public:
		TemplateSolution(const SolutionParameters& parameters);
		~TemplateSolution() override = default;

		void buildPhenotype() override;
		void evaluatePhenotype() override;
		std::shared_ptr<Solution> clone() const override;
	private:
		void createRandomConnectionGenes() override;
	};
}