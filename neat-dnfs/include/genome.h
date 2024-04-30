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
		static std::map<std::tuple<uint16_t, uint16_t>, uint16_t> generationalInnovations;
	public:
		Genome() = default;

		void addInputGene();
		void addOutputGene();
		void addRandomInitialConnectionGene();
		void mutate();
		static void clearGenerationalInnovations();

		std::vector<Gene> getGenes() const;
		std::vector<ConnectionGene> getConnectionGenes() const;
	private:
		std::tuple<uint16_t, uint16_t> getNewRandomGeneTuple() const;
		void addConnectionGeneIfNewWithinGeneration(std::tuple<uint16_t, uint16_t> geneTuple);
		uint16_t getRandomGeneIdByType(GeneType type) const;
		ConnectionGene getEnabledConnectionGene() const;
		void addGene();
		void mutateGene();
		void addConnectionGene();
		void mutateConnectionGene();
	};
}
