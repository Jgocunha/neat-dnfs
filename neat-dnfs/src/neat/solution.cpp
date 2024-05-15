#include "neat/solution.h"

#include "solutions/template_solution.h" // this should not be here

namespace neat_dnfs
{
	SolutionTopology::SolutionTopology(int numInputGenes, int numOutputGenes, int numHiddenGenes, int numConnections)
		: numInputGenes(numInputGenes), numOutputGenes(numOutputGenes),
		numHiddenGenes(numHiddenGenes), numConnections(numConnections)
	{}

	SolutionParameters::SolutionParameters(double fitness, double adjustedFitness, int age)
		: fitness(fitness), adjustedFitness(adjustedFitness),
		reproductionProbability(0.0), age(age)
	{}

	Solution::Solution(const SolutionTopology& initialTopology)
		:initialTopology(initialTopology)
	{
		if (initialTopology.numInputGenes < 1 || initialTopology.numOutputGenes < 1)
			throw std::invalid_argument("Number of input and output genes must be greater than 0");
		parameters = SolutionParameters();
		genome = Genome();
		phenotype = Phenotype{ "simulation", 1, 0, 0 };
	}

	void Solution::initialize()
	{
		createInputGenes();
		createOutputGenes();
		createRandomInitialConnectionGenes();
	}

	void Solution ::mutate()
	{
		genome.mutate();
	}

	Phenotype Solution::getPhenotype() const
	{
		return phenotype;
	}

	Genome Solution::getGenome() const
	{
		return genome;
	}

	SolutionParameters Solution::getParameters() const
	{
		return parameters;
	}

	double Solution::getFitness() const
	{
		return parameters.fitness;
	}

	size_t Solution::getGenomeSize() const
	{
		return genome.getConnectionGenes().size();
	}

	std::vector<uint16_t> Solution::getInnovationNumbers() const
	{
		return genome.getInnovationNumbers();
	}

	void Solution::clearGenerationalInnovations() const
	{
		genome.clearGenerationalInnovations();
	}

	void Solution::incrementAge()
	{
		parameters.age++;
	}

	void Solution::setAdjustedFitness(double adjustedFitness)
	{
		parameters.adjustedFitness = adjustedFitness;
	}

	void Solution::buildPhenotype()
	{
		translateGenesToPhenotype();
		translateConnectionGenesToPhenotype();
	}

	void Solution::createInputGenes()
	{
		for (int j = 0; j < initialTopology.numInputGenes; j++)
			genome.addInputGene();
	}

	void Solution::createOutputGenes()
	{
		for (int j = 0; j < initialTopology.numOutputGenes; j++)
			genome.addOutputGene();
	}

	void Solution::translateGenesToPhenotype()
	{
		for (auto const& gene : genome.getFieldGenes())
		{
			const auto nf = gene.getNeuralField();
			const auto kernel = gene.getKernel();
			phenotype.addElement(nf);
			phenotype.addElement(kernel);
			phenotype.createInteraction(nf->getUniqueName(), "output", kernel->getUniqueName());
			phenotype.createInteraction(kernel->getUniqueName(), "output", nf->getUniqueName());
		}
	}

	void Solution::translateConnectionGenesToPhenotype()
	{
		for (auto const& connectionGene : genome.getConnectionGenes())
		{
			if (connectionGene.isEnabled())
			{
				const auto kernel = connectionGene.getKernel();
				const auto sourceId = connectionGene.getInFieldGeneId();
				const auto targetId = connectionGene.getOutFieldGeneId();

				phenotype.addElement(kernel);
				phenotype.createInteraction("nf " + std::to_string(sourceId),
					"output", kernel->getUniqueName());
				phenotype.createInteraction(kernel->getUniqueName(), 
					"output", "nf " + std::to_string(targetId));
			}
		}
	}


	SolutionPtr Solution::crossover(const SolutionPtr& parent1, const SolutionPtr& parent2)
	{
		const SolutionPtr moreFitParent = parent1->getFitness() > parent2->getFitness() ? parent1 : parent2;
		const SolutionPtr lessFitParent = parent1->getFitness() > parent2->getFitness() ? parent2 : parent1;

		SolutionPtr offspring = std::make_shared<TemplateSolution>(moreFitParent->initialTopology);

		for (const auto& gene : moreFitParent->getGenome().getFieldGenes())
			offspring->addFieldGene(gene);

		const auto& parentConnectionGenes = moreFitParent->getGenome().getConnectionGenes();
		for(const auto& gene : parentConnectionGenes)
		{
			if (lessFitParent->containsConnectionGene(gene))
			{
				const auto lessFitGene = lessFitParent->getGenome().getConnectionGeneByInnovationNumber(gene.getInnovationNumber());
				if (rand() % 2 == 0)
					offspring->addConnectionGene(gene);
				else
					offspring->addConnectionGene(lessFitGene);
			}
			else
				offspring->addConnectionGene(gene);
		}

		return offspring;
	}

	void Solution::addFieldGene(const FieldGene& gene)
	{
		genome.addFieldGene(gene);
	}

	void Solution::addConnectionGene(const ConnectionGene& gene)
	{
		genome.addConnectionGene(gene);
	}

	bool Solution::containsConnectionGene(const ConnectionGene& gene) const
	{
		return genome.containsConnectionGene(gene);
	}

}
