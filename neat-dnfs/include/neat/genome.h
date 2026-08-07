#pragma once

#include <mutex>

#include "constants.h"
#include "field_gene.h"
#include "connection_gene.h"
#include "neat_tools/utils.h"

namespace neat_dnfs
{
	/// @brief Encodes a candidate solution as a set of field genes and connection genes.
	///
	/// Implements the NEAT genome representation: a graph of neural field nodes (FieldGene)
	/// connected by weighted edges (ConnectionGene), each tagged with a global innovation
	/// number to enable structural crossover across genomes.
	class Genome
	{
	private:
		std::vector<FieldGene> fieldGenes;
		std::vector<ConnectionGene> connectionGenes;
		static int globalInnovationNumber;
		static std::map<ConnectionTuple, int> connectionTupleAndInnovationNumberWithinGeneration;
		static std::mutex innovationMutex; ///< Guards globalInnovationNumber and the generational innovation map.
		std::string mutationsInLastGeneration;
	public:
		Genome() = default;
		~Genome() = default;
		Genome(const Genome& other) = default;
		Genome(Genome&& other) noexcept = default;
		Genome& operator=(const Genome& other) = default;
		Genome& operator=(Genome&& other) noexcept = default;

		void addInputGene(const dnf_composer::element::ElementDimensions& dimensions);
		void addOutputGene(const dnf_composer::element::ElementDimensions& dimensions);
		void addHiddenGene(const FieldGene& gene);

		void mutate();
		void checkForDuplicateConnectionGenes() const;
		/// @brief Clears the innovation-number lookup table built during a generation.
		/// Must be called at the end of each generation before the next round of mutations.
		static void clearGenerationalInnovations();
		static void resetGlobalInnovationNumber();
		void clearLastMutations();
		void removeConnectionGene(int innov);

		[[nodiscard]] std::vector<FieldGene> getFieldGenes() const;
		[[nodiscard]] std::vector<ConnectionGene> getConnectionGenes() const;
		[[nodiscard]] std::vector<int> getInnovationNumbers() const;
		static int getGlobalInnovationNumber();
		[[nodiscard]] std::string getMutationsInLastGeneration() const;

		/// @brief Number of connection genes present in @p other but beyond the range of this genome's innovation numbers.
		[[nodiscard]] int excessGenes(const Genome& other) const;
		/// @brief Number of connection genes that appear in one genome but not the other within the overlapping innovation range.
		[[nodiscard]] int disjointGenes(const Genome& other) const;
		/// @brief Mean absolute difference in kernel parameters across matching connection genes.
		[[nodiscard]] double averageConnectionDifference(const Genome& other) const;

		void addFieldGene(const FieldGene& fieldGene);
		void addConnectionGene(const ConnectionGene& connectionGene);
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
		/// @brief Draws two distinct field gene ids to form a candidate connection.
		/// @return The chosen {inFieldGeneId, outFieldGeneId} tuple, or {0, 0} if
		/// fewer than two field genes exist, no eligible candidate is found, the
		/// tuple already exists as a connection gene, or no distinct second gene
		/// could be drawn within the bounded number of retries.
		[[nodiscard]] ConnectionTuple getNewRandomConnectionGeneTuple() const;
		[[nodiscard]] int getRandomGeneId() const;
		[[nodiscard]] int getRandomGeneIdByType(FieldGeneType type) const;
		[[nodiscard]] int getRandomGeneIdByTypes(const std::vector<FieldGeneType>& types) const;
		[[nodiscard]] ConnectionGene* getEnabledConnectionGene();

		void addConnectionGene(ConnectionTuple connectionTuple);
		void addGene();
		void mutateGene();
		void addConnectionGene();
		void mutateConnectionGene();
		void toggleConnectionGene();

		/// @brief Looks up the innovation number assigned to @p tuple within the current generation, or -1 if none.
		/// @warning Caller MUST hold @c innovationMutex.
		static int getInnovationNumberOfTupleWithinGenerationUnlocked(const ConnectionTuple& tuple);
	};
}
