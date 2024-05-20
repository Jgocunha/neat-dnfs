#include "solutions/empty_solution.h"

namespace neat_dnfs
{
	EmptySolution::EmptySolution(const SolutionTopology& topology)
		: Solution(topology)
	{}

	void EmptySolution::evaluate()
	{
		parameters.fitness = tools::utils::generateRandomDouble(0.0, 1.0);
	}

	SolutionPtr EmptySolution::clone() const
	{
		EmptySolution solution(initialTopology);
		auto clonedSolution = std::make_shared<EmptySolution>(solution);

		return clonedSolution;
	}

	SolutionPtr EmptySolution::crossover(const SolutionPtr& other)
	{
	    const SolutionPtr self = std::make_shared<EmptySolution>(*this);

		const double fitnessDifference = std::abs(this->getFitness() - other->getFitness());

	    const SolutionPtr moreFitParent = this->getFitness() > other->getFitness() ? self : other;
	    const SolutionPtr lessFitParent = this->getFitness() > other->getFitness() ? other : self;

	    SolutionPtr offspring = std::make_shared<EmptySolution>(initialTopology);

	    for (const auto& gene : moreFitParent->getGenome().getFieldGenes())
	        offspring->addFieldGene(gene);

	    const auto& parentConnectionGenes = moreFitParent->getGenome().getConnectionGenes();
	    for (const auto& gene : parentConnectionGenes)
	    {
			// Matching genes are inherited randomly from either parent
	        if (lessFitParent->containsConnectionGene(gene))
	        {
	            const auto lessFitGene = lessFitParent->getGenome().getConnectionGeneByInnovationNumber(gene.getInnovationNumber());
	            if (tools::utils::generateRandomInt(0, 1))
	                offspring->addConnectionGene(gene);
	            else
	                offspring->addConnectionGene(lessFitGene);
	        }
	        else
	        {
				// Disjoint and excess genes are inherited from the more fit parent
				// unless the fitness difference is 0, in which case the gene is inherited randomly
		        if (fitnessDifference == 0.0)
		        {
					const auto lessFitGene = lessFitParent->getGenome().getConnectionGeneByInnovationNumber(gene.getInnovationNumber());
					if (tools::utils::generateRandomInt(0, 1))
						offspring->addConnectionGene(gene);
					else
						offspring->addConnectionGene(lessFitGene);
		        }
				else
					offspring->addConnectionGene(gene);
	        }
	    }

	    return offspring;
	}

}
