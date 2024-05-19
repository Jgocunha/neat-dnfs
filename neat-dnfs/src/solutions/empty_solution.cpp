#include "solutions/empty_solution.h"

namespace neat_dnfs
{
	EmptySolution::EmptySolution(const SolutionTopology& topology)
		: Solution(topology)
	{}

	void EmptySolution::evaluate()
	{
		parameters.fitness = 0.0;
	}

	SolutionPtr EmptySolution::clone() const
	{
		EmptySolution solution(initialTopology);
		auto clonedSolution = std::make_shared<EmptySolution>(solution);

		return clonedSolution;
	}

	SolutionPtr EmptySolution::crossover(const SolutionPtr& other)
	{
		const SolutionPtr moreFitParent = this->getFitness() > other->getFitness() ? this->shared_from_this() : other;
		const SolutionPtr lessFitParent = this->getFitness() > other->getFitness() ? other : this->shared_from_this();

		SolutionPtr offspring = std::make_shared<EmptySolution>(initialTopology);

		for (const auto& gene : moreFitParent->getGenome().getFieldGenes())
			offspring->addFieldGene(gene);

		const auto& parentConnectionGenes = moreFitParent->getGenome().getConnectionGenes();
		for (const auto& gene : parentConnectionGenes)
		{
			if (lessFitParent->containsConnectionGene(gene))
			{
				const auto lessFitGene = lessFitParent->getGenome().getConnectionGeneByInnovationNumber(gene.getInnovationNumber());
				if (dnf_composer::tools::utils::generateRandomNumber(0, 1))
					offspring->addConnectionGene(gene);
				else
					offspring->addConnectionGene(lessFitGene);
			}
			else
				offspring->addConnectionGene(gene);
		}

		return offspring;
	}


}
