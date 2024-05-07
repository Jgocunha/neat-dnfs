#pragma once

#include "field_gene.h"
#include "connection_gene.h"

namespace neat_dnfs
{
	class Genome
	{
	private:
		std::vector<FieldGene> fieldGenes;
		std::vector<ConnectionGene> connectionGenes;
		static std::map<ConnectionTuple, uint16_t> connectionToInnovationNumberMap;
	public:
		Genome() = default;

		void addInputGene();
		void addOutputGene();
		void addRandomInitialConnectionGene();
		void mutate();
		static void clearGenerationalInnovations();

		std::vector<FieldGene> getGenes() const;
		std::vector<ConnectionGene> getConnectionGenes() const;
		std::vector<uint16_t> getInnovationNumbers() const;
	private:
		ConnectionTuple getNewRandomConnectionGeneTuple() const;
		uint16_t getRandomGeneIdByType(FieldGeneType type) const;
		ConnectionGene getEnabledConnectionGene() const;

		void addConnectionGeneIfNewWithinGeneration(ConnectionTuple connectionTuple);
		void addGene();
		void mutateGene() const;
		void addConnectionGene();
		void mutateConnectionGene();
		void toggleConnectionGene();
	public:
		int excessGenes(const Genome& other) const;
		int disjointGenes(const Genome& other) const;
		double averageConnectionDifference(const Genome& other) const;
	};
}
