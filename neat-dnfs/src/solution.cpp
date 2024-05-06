#include "solution.h"

namespace neat_dnfs
{
	Solution::Solution(const SolutionParameters& parameters)
		:parameters(parameters)
	{
		if (parameters.numInputGenes < 1 || parameters.numOutputGenes < 1)
			throw std::invalid_argument("Number of input and output genes must be greater than 0");
		genome = Genome();
		phenotype = Phenotype();
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

	void Solution::clearGenerationalInnovations() const
	{
		genome.clearGenerationalInnovations();
	}

	std::vector<uint16_t> Solution::getInnovationNumbers() const
	{
		return genome.getInnovationNumbers();
	}

	void Solution::setSpecies(int species)
	{
		parameters.species = species;
	}

	void Solution::buildPhenotype()
	{
		translateGenesToPhenotype();
		translateConnectionGenesToPhenotype();
	}

	void Solution::createInputGenes()
	{
		for (int j = 0; j < parameters.numInputGenes; j++)
			genome.addInputGene();
	}

	void Solution::createOutputGenes()
	{
		for (int j = 0; j < parameters.numOutputGenes; j++)
			genome.addOutputGene();
	}

	void Solution::translateGenesToPhenotype()
	{
		for (auto const& gene : genome.getGenes())
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
				const auto sourceId = connectionGene.getInGeneId();
				const auto targetId = connectionGene.getOutGeneId();

				phenotype.addElement(kernel);
				phenotype.createInteraction("nf " + std::to_string(sourceId), "output", kernel->getUniqueName());
				phenotype.createInteraction(kernel->getUniqueName(), "output", "nf " + std::to_string(targetId));
			}
		}
	}

	int Solution::getGenomeSize() const
	{
		return static_cast<int>(genome.getGenes().size());
	}

	void Solution::incrementAge()
	{
		parameters.age++;
	}
}
