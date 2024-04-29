#include "template_solution.h"

namespace neat_dnfs
{
	TemplateSolution::TemplateSolution(const SolutionParameters& parameters)
		:Solution(parameters)
	{
	}

	void TemplateSolution::buildPhenotype()
	{
		
	}

	void TemplateSolution::evaluatePhenotype()
	{
		
	}

	std::shared_ptr<Solution> TemplateSolution::clone() const
	{
		TemplateSolution solution(parameters);
		auto clonedSolution = std::make_shared<TemplateSolution>(solution);

		return clonedSolution;
	}

	void TemplateSolution::createRandomConnectionGenes()
	{
		static constexpr double connectionProbability = 0.3;
		for(int i = 0; i < parameters.numInputGenes * parameters.numOutputGenes; ++i)
			if(dnf_composer::tools::utils::generateRandomNumber(0.0, 1.0) < connectionProbability)
				genome.addConnectionGene();
	}
}