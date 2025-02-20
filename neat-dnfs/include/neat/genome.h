#pragma once

#include "constants.h"
#include "field_gene.h"
#include "connection_gene.h"
#include "tools/utils.h"

namespace neat_dnfs
{
	static int globalInnovationNumber = 0;

	class Genome
	{
	private:
		std::vector<FieldGene> fieldGenes;
		std::vector<ConnectionGene> connectionGenes;
		static std::map<ConnectionTuple, int> connectionTupleAndInnovationNumberWithinGeneration;
	public:
		Genome() = default;

		void addInputGene(const dnf_composer::element::ElementDimensions& dimensions);
		void addOutputGene(const dnf_composer::element::ElementDimensions& dimensions);
		void addHiddenGene(const dnf_composer::element::ElementDimensions& dimensions);
		void mutate();
		static void clearGenerationalInnovations();
		void removeConnectionGene(int innov);

		std::vector<FieldGene> getFieldGenes() const;
		std::vector<ConnectionGene> getConnectionGenes() const;
		std::vector<int> getInnovationNumbers() const;
		static int getGlobalInnovationNumber() { return globalInnovationNumber; }
	private:
		ConnectionTuple getNewRandomConnectionGeneTuple() const;
		int getRandomGeneId() const;
		int getRandomGeneIdByType(FieldGeneType type) const;
		int getRandomGeneIdByTypes(const std::vector<FieldGeneType>& types) const;
		ConnectionGene getEnabledConnectionGene() const;

		void addConnectionGene(ConnectionTuple connectionTuple);
		void addGene();
		void mutateGene() const;
		void addConnectionGene();
		void mutateConnectionGene();
		void toggleConnectionGene();

		static int getInnovationNumberOfTupleWithinGeneration(const ConnectionTuple& tuple);
	public:
		int excessGenes(const Genome& other) const;
		int disjointGenes(const Genome& other) const;
		double averageConnectionDifference(const Genome& other) const;

		void addFieldGene(const FieldGene& fieldGene);
		void addConnectionGene(const ConnectionGene& connectionGene);
		bool containsConnectionGene(const ConnectionGene& connectionGene) const;
		bool containsFieldGene(const FieldGene& fieldGene) const;
		bool containsConnectionGeneWithTheSameInputOutputPair(const ConnectionGene& gene) const;

		ConnectionGene getConnectionGeneByInnovationNumber(int innovationNumber) const;
		FieldGene getFieldGeneById(int id) const;

		bool operator==(const Genome& other) const;
		std::string toString() const;
		void print() const;
	};
}
