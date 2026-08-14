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
		std::string mutationsInLastGeneration;
	public:
		Genome() = default;
		~Genome();

		void addInputGene(const dnf_composer::element::ElementDimensions& dimensions);
		void addOutputGene(const dnf_composer::element::ElementDimensions& dimensions);
		void addHiddenGene(const FieldGene& gene);
		void addHiddenGene(const dnf_composer::element::ElementDimensions& dimensions);

		void mutate();
		void checkForDuplicateConnectionGenes() const;
		static void clearGenerationalInnovations();
		static void resetGlobalInnovationNumber();
		void clearLastMutations();
		void removeConnectionGene(int innov);

		[[nodiscard]] std::vector<FieldGene> getFieldGenes() const;
		[[nodiscard]] std::vector<ConnectionGene> getConnectionGenes() const;
		[[nodiscard]] std::vector<int> getInnovationNumbers() const;
		static int getGlobalInnovationNumber();
		[[nodiscard]] std::string getMutationsInLastGeneration() const;

		[[nodiscard]] int excessGenes(const Genome& other) const;
		[[nodiscard]] int disjointGenes(const Genome& other) const;
		[[nodiscard]] double averageConnectionDifference(const Genome& other) const;

		void addFieldGene(const FieldGene& fieldGene);
		void addConnectionGene(const ConnectionGene& connectionGene);
		void seedAllLegalConnections();
		void seedRandomConnections(int count);
		[[nodiscard]] std::vector<ConnectionTuple> legalConnectionTuples() const;
		[[nodiscard]] int maxLegalConnectionCount() const;
		[[nodiscard]] bool containsConnectionGene(const ConnectionGene& connectionGene) const;
		[[nodiscard]] bool containsFieldGene(const FieldGene& fieldGene) const;
		[[nodiscard]] bool containsConnectionGeneWithTheSameInputOutputPair(const ConnectionGene& gene) const;

		[[nodiscard]] ConnectionGene getConnectionGeneByInnovationNumber(int innovationNumber) const;
		[[nodiscard]] FieldGene getFieldGeneById(int id) const;

		[[nodiscard]] bool isEmpty() const;
		bool operator==(const Genome& other) const;
		[[nodiscard]] std::string toString() const;
		void print() const;
	private:
		[[nodiscard]] ConnectionTuple getNewRandomConnectionGeneTuple() const;
		[[nodiscard]] int getRandomGeneId() const;
		[[nodiscard]] int getRandomGeneIdByType(FieldGeneType type) const;
		[[nodiscard]] int getRandomGeneIdByTypes(const std::vector<FieldGeneType>& types) const;
		[[nodiscard]] ConnectionGene* getEnabledConnectionGene() const;

		void addConnectionGene(ConnectionTuple connectionTuple);
		void addGene();
		void mutateGene();
		void addConnectionGene();
		void mutateConnectionGene();
		void toggleConnectionGene();

		static int getInnovationNumberOfTupleWithinGeneration(const ConnectionTuple& tuple);
	};
}
