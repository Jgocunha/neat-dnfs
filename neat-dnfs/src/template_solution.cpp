#include "template_solution.h"

namespace neat_dnfs
{
	TemplateSolution::TemplateSolution(const SolutionParameters& parameters)
		:Solution(parameters)
	{
	}

	void TemplateSolution::buildPhenotype()
	{
		phenotype.buildFromGenome(genome);
	}

	std::shared_ptr<Solution> TemplateSolution::clone() const
	{
		TemplateSolution solution(parameters);
		auto clonedSolution = std::make_shared<TemplateSolution>(solution);

		return clonedSolution;
	}

	void TemplateSolution::evaluate()
	{
		buildPhenotype();
		evaluatePhenotype();
	}

	void TemplateSolution::mutate()
	{
		genome.mutate();
	}

	void TemplateSolution::evaluatePhenotype()
	{
		parameters.fitness = phenotype.evaluateFitness();
		std::cout << "Fitness: " << parameters.fitness << std::endl;
		if( parameters.fitness > 0.8)
		{
			std::cout << "Solution found!" << std::endl;
			//std::cout << "Genome: " << genome << std::endl;
			//std::cout << "Phenotype: " << phenotype << std::endl;
		}
	}

	void TemplateSolution::createRandomInitialConnectionGenes()
	{
		static constexpr double connectionProbability = 0.0;
		for(int i = 0; i < parameters.numInputGenes * parameters.numOutputGenes; ++i)
			if(dnf_composer::tools::utils::generateRandomNumber(0.0, 1.0) < connectionProbability)
				genome.addRandomInitialConnectionGene();
	}
}