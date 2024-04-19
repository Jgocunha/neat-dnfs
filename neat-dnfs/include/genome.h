#pragma once

#include "gene.h"
#include "connection_gene.h"

namespace neat_dnfs
{
	struct GenomeParameters
	{
		double fitness;
		int speciesId;
		double adjustedFitness;
		int age;

		GenomeParameters(double fitness = 0.0, 
			int speciesId = 0,
			double adjustedFitness = 0.0, 
			int age = 0)
			: fitness(fitness), 
			speciesId(speciesId), 
			adjustedFitness(adjustedFitness), 
			age(age)
		{}
	};

	class Genome
	{
	private:
		std::vector<Gene> genes;
		std::vector<ConnectionGene> connectionGenes;
		GenomeParameters parameters;
	public:
		Genome();

		void addInputGene();
		void addOutputGene();
		void addHiddenGene();
		void addConnectionGene();

		void evaluate();

		GenomeParameters getParameters() const;
	private:
		unsigned long int getRandomGeneIdByType(GeneType type) const;
	};
}
