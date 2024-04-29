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
		void addConnectionGene();

		std::vector<Gene> getGenes() const;
		std::vector<ConnectionGene> getConnectionGenes() const;
	private:
		unsigned long int getRandomGeneIdByType(GeneType type) const;
	};
}
