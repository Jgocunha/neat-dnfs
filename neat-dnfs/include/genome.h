#pragma once

#include "gene.h"
#include "connection_gene.h"

namespace neat_dnfs
{
	class Genome
	{
	private:
		std::vector<Gene> genes;
		std::vector<ConnectionGene> connectionGenes;
	public:
		Genome() = default;

		void addInputGene();
		void addOutputGene();
		void addHiddenGene();
		void addRandomInitialConnectionGene();
		void mutate();

		std::vector<Gene> getGenes() const;
		std::vector<ConnectionGene> getConnectionGenes() const;
	private:
		void addConnectionGene();
		unsigned long int getRandomGeneIdByType(GeneType type) const;
	};
}
