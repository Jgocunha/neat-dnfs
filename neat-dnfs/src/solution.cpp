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

}
