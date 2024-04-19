#include "population.h"

namespace neat_dnfs
{
	Population::Population(int size)
		: size(size), currentGeneration(0)
	{
		genomes.reserve(size);
	}

	void Population::initialize(int numInputGenes, int numOutputGenes, int numConnectionGenes)
	{
		if (numInputGenes < 1 || numOutputGenes < 1)
			throw std::invalid_argument("Number of input and output genes must be greater than 0");

		createInitialEmptyGenomes();
		addInputGenes(numInputGenes);
		addOutputGenes(numOutputGenes);
		addRandomConnectionGenes(numConnectionGenes);
	}

	void Population::evaluate()
	{
		for (auto& genome : genomes)
		{
			genome.evaluate();
		}
	}

	void Population::createInitialEmptyGenomes()
	{
		for (int i = 0; i < size; i++)
			genomes.emplace_back();
	}

	void Population::addInputGenes(int numInputGenes)
	{
		for (int i = 0; i < size; i++)
			for (int j = 0; j < numInputGenes; j++)
				genomes[i].addInputGene();
	}

	void Population::addOutputGenes(int numOutputGenes)
	{
		for (int i = 0; i < size; i++)
			for (int j = 0; j < numOutputGenes; j++)
				genomes[i].addOutputGene();
	}

	void Population::addRandomConnectionGenes(int numConnectionGenes)
	{
		for (int i = 0; i < size; i++)
			for (int j = 0; j < numConnectionGenes; j++)
				genomes[i].addConnectionGene();
	}
}
