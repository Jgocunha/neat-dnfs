#pragma once

#include "constants.h"
#include "field_gene.h"
#include "connection_gene.h"
#include "tools/utils.h"

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

		void addInputGene(const dnf_composer::element::ElementDimensions& dimensions);
		void addOutputGene(const dnf_composer::element::ElementDimensions& dimensions);
		void addHiddenGene(const dnf_composer::element::ElementDimensions& dimensions);
		void addRandomInitialConnectionGene();
		void mutate();
		static void clearGenerationalInnovations();
		void removeConnectionGene(uint16_t innov);

		std::vector<FieldGene> getFieldGenes() const;
		std::vector<ConnectionGene> getConnectionGenes() const;
		std::vector<uint16_t> getInnovationNumbers() const;
	private:
		ConnectionTuple getNewRandomConnectionGeneTuple() const;
		int getRandomGeneId() const;
		int getRandomGeneIdByType(FieldGeneType type) const;
		int getRandomGeneIdByTypes(const std::vector<FieldGeneType>& types) const;
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

		void addFieldGene(const FieldGene& fieldGene);
		void addConnectionGene(const ConnectionGene& connectionGene);
		bool containsConnectionGene(const ConnectionGene& connectionGene) const;
		bool containsFieldGene(const FieldGene& fieldGene) const;
		bool containsConnectionGeneWithTheSameInputOutputPair(const ConnectionGene& gene) const;

		ConnectionGene getConnectionGeneByInnovationNumber(uint16_t innovationNumber) const;
		FieldGene getFieldGeneById(uint16_t id) const;

		static std::map<ConnectionTuple, uint16_t> getConnectionToInnovationNumberMap() { return connectionToInnovationNumberMap; }
		bool operator==(const Genome& other) const;
		std::string toString() const;
		void print() const;
	};
}
