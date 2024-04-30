#include "genome.h"

namespace neat_dnfs
{
	std::map<std::tuple<uint16_t, uint16_t>, uint16_t> Genome::generationalInnovations;

	void Genome::addInputGene()
	{
		const auto index = genes.size() + 1;
		genes.push_back(Gene({GeneType::INPUT, static_cast<uint16_t>(index) }));
	}

	void Genome::addOutputGene()
	{
		const auto index = genes.size() + 1;
		genes.push_back(Gene({ GeneType::OUTPUT, static_cast<uint16_t>(index) }));
	}

	void Genome::addHiddenGene()
	{
		const auto index = genes.size() + 1;
		genes.push_back(Gene({ GeneType::HIDDEN, static_cast<uint16_t>(index) }));
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
		static constexpr double addConnectionGeneProbability = 0.3;
		static constexpr double mutateConnectionGeneProbability = 0.5;
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

		if (mutateConnectionGeneProbability > static_cast<double>(rand()) / RAND_MAX)
		{
			if (connectionGenes.empty())
			{
				std::cout << "No connection genes to mutate." << std::endl;
				return;
			}
			const auto connectionGeneId = rand() % connectionGenes.size();
			connectionGenes[connectionGeneId].mutate();
		}

		/*if (toggleConnectionGeneProbability > static_cast<double>(rand()) / RAND_MAX)
		{
			const auto connectionGeneId = rand() % connectionGenes.size();
			connectionGenes[connectionGeneId].toggle();
		}*/
	}

	void Genome::clearGenerationalInnovations()
	{
		generationalInnovations.clear();
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
		const std::tuple<uint16_t, uint16_t> geneTuple = getNewRandomGeneTuple();
		if (geneTuple == std::tuple<uint16_t, uint16_t>{})
			return;
		addConnectionGeneIfNewWithinGeneration(geneTuple);
	}

	std::tuple<uint16_t, uint16_t> Genome::getNewRandomGeneTuple() const
	{
		// Select two different random genes
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<uint16_t> dis(1, genes.size());
		uint16_t geneIndex1 = dis(gen);

		// Ensure geneIndex2 is different from geneIndex1
		uint16_t geneIndex2;
		do {
			geneIndex2 = dis(gen);
		} while (geneIndex2 == geneIndex1);

		// If a connection gene between the two genes already exists, return
		if (std::find_if(connectionGenes.begin(), connectionGenes.end(),
			[geneIndex1, geneIndex2](const ConnectionGene& connectionGene) 
			{
				return connectionGene.getParameters().inGeneId == geneIndex1 && connectionGene.getParameters().outGeneId == geneIndex2;
			} ) != connectionGenes.end())
			return {};

		return std::tuple{geneIndex1, geneIndex2};
	}

	void Genome::addConnectionGeneIfNewWithinGeneration(std::tuple<uint16_t, uint16_t> geneTuple)
	{
		if (generationalInnovations.contains(geneTuple))
		{
			const uint16_t innovationNumber = generationalInnovations[geneTuple];
			connectionGenes.emplace_back(std::get<0>(geneTuple), std::get<1>(geneTuple));
			connectionGenes.back().setInnovationNumber(innovationNumber);
		}
		else
		{
			connectionGenes.emplace_back(std::get<0>(geneTuple), std::get<1>(geneTuple));
			generationalInnovations.insert({ geneTuple,
				connectionGenes[connectionGenes.size() - 1].getInnovationNumber() });
		}
	}

	uint16_t Genome::getRandomGeneIdByType(GeneType type) const
	{
		std::vector<uint16_t> geneIds;
		for (const auto& gene : genes)
		{
			if (gene.getParameters().type == type)
			geneIds.push_back(gene.getParameters().id);
		}

		if (geneIds.empty())
			throw std::invalid_argument("There are no genes of the specified type.");

		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<uint16_t> dis(0, geneIds.size() - 1);

		return geneIds[dis(gen)];
	}
}