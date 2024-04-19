#pragma once

#include "genome.h"

namespace neat_dnfs
{
	class Population
	{
	private:
		int size;
		unsigned long int currentGeneration;
		std::vector<Genome> genomes;
	public:
		Population(int size);
		void initialize(int numInputGenes, int numOutputGenes, int numConnectionGenes);
		void evaluate();
	private:
		void createInitialEmptyGenomes();
		void addInputGenes(int numInputGenes);
		void addOutputGenes(int numOutputGenes);
		void addRandomConnectionGenes(int numConnectionGenes);
	};
}
