#include "genome.h"

namespace neat_dnfs
{
	void Genome::addInputGene()
	{
		const auto index = genes.size() + 1;
		genes.push_back(Gene({GeneType::INPUT, static_cast<unsigned long int>(index) }));
	}

	void Genome::addOutputGene()
	{
		const auto index = genes.size() + 1;
		genes.push_back(Gene({ GeneType::OUTPUT, static_cast<unsigned long int>(index) }));
	}

	void Genome::addHiddenGene()
	{
		const auto index = genes.size() + 1;
		genes.push_back(Gene({ GeneType::HIDDEN, static_cast<unsigned long int>(index) }));
	}

	void Genome::addRandomInitialConnectionGene()
	{
		if (genes.size() < 2)
			throw std::invalid_argument("There must be at least two genes to add a connection gene.");

		const auto inGeneId = getRandomGeneIdByType(GeneType::INPUT);
		const auto outGeneId = getRandomGeneIdByType(GeneType::OUTPUT);

		connectionGenes.emplace_back(inGeneId, outGeneId);
	}

	void Genome::mutate()
	{
		//Adding nodes : Splits an existing connection,
		// adding a new node in between and potentially increasing network depth.
		//Adding connections : Introduces a new connection between previously unconnected neurons,
		// increasing network complexity.
		//Adjusting weights : Modifies the weights of existing connections
		// to alter the behavior of the neural network.
		//Enabling / disabling connections : Can temporarily remove or add connections
		// without deleting them, allowing the algorithm to test the effect of different
		// network structures.


		static constexpr double addGeneProbability = 0.1;
		static constexpr double mutateGeneProbability = 0.1;
		static constexpr double addConnectionGeneProbability = 0.5;
		static constexpr double mutateConnectionGeneProbability = 0.1;
		static constexpr double toggleConnectionGeneProbability = 0.1;

		/*if (addGeneProbability > static_cast<double>(rand()) / RAND_MAX)
			addHiddenGene();

		if (mutateGeneProbability > static_cast<double>(rand()) / RAND_MAX)
		{
			const auto geneId = getRandomGeneIdByType(GeneType::HIDDEN);
			genes[geneId].mutate();
		}*/

		if (addConnectionGeneProbability > static_cast<double>(rand()) / RAND_MAX)
			addConnectionGene();

		/*if (mutateConnectionGeneProbability > static_cast<double>(rand()) / RAND_MAX)
		{
			const auto connectionGeneId = rand() % connectionGenes.size();
			connectionGenes[connectionGeneId].mutate();
		}

		if (toggleConnectionGeneProbability > static_cast<double>(rand()) / RAND_MAX)
		{
			const auto connectionGeneId = rand() % connectionGenes.size();
			connectionGenes[connectionGeneId].toggle();
		}*/
	}

	
	std::vector<Gene> Genome::getGenes() const
	{
		return genes;
	}

	std::vector<ConnectionGene> Genome::getConnectionGenes() const
	{
		return connectionGenes;
	}

	void Genome::addConnectionGene()
	{
		 // Select two different random genes
	    std::random_device rd;
	    std::mt19937 gen(rd());
	    std::uniform_int_distribution<unsigned long int> dis(1, genes.size());
	    unsigned long int geneIndex1 = dis(gen);

	    unsigned long int geneIndex2;
	    do {
	        geneIndex2 = dis(gen);
	    } while (geneIndex2 == geneIndex1); // Ensure geneIndex2 is different from geneIndex1

	    connectionGenes.emplace_back(geneIndex1, geneIndex2);
	}

	unsigned long int Genome::getRandomGeneIdByType(GeneType type) const
	{
		std::vector<unsigned long int> geneIds;
		for (const auto& gene : genes)
		{
			if (gene.getParameters().type == type)
			geneIds.push_back(gene.getParameters().id);
		}

		if (geneIds.empty())
			throw std::invalid_argument("There are no genes of the specified type.");

		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<unsigned long int> dis(0, geneIds.size() - 1);

		return geneIds[dis(gen)];
	}
}