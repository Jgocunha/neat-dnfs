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
		// Connection weights mutate as in any NE system, with each connection either perturbed
		//or not at each generation.

		//In the add connection mutation, a single new connection gene
		// with a random weight is added connecting two previously unconnected nodes.

		//In the add node mutation, an existing connection is
		//	split and the new node placed where the old connection used to be.The old connection
		//	is disabled and two new connections are added to the genome.The new connection
		//	leading into the new node receives a weight of 1, and the new connection leading out
		//	receives the same weight as the old connection.

		static constexpr double addGeneProbability = 0.1;
		static constexpr double mutateGeneProbability = 0.2;
		static constexpr double addConnectionGeneProbability = 0.3;
		static constexpr double mutateConnectionGeneProbability = 0.35;
		static constexpr double toggleConnectionGeneProbability = 0.05;

		if (addGeneProbability > static_cast<double>(rand()) / RAND_MAX)
		{
			addGene();
			return;
		}

		if (mutateGeneProbability > static_cast<double>(rand()) / RAND_MAX)
		{
			mutateGene();
			return;
		}

		if (addConnectionGeneProbability > static_cast<double>(rand()) / RAND_MAX)
		{
			addConnectionGene();
			return;
		}

		if (mutateConnectionGeneProbability > static_cast<double>(rand()) / RAND_MAX)
		{
			mutateConnectionGene();
			return;
		}

		if (toggleConnectionGeneProbability > static_cast<double>(rand()) / RAND_MAX)
		{
			const auto connectionGeneId = rand() % connectionGenes.size();
			connectionGenes[connectionGeneId].toggle();
		}
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
			return 0;

		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<uint16_t> dis(0, geneIds.size() - 1);

		return geneIds[dis(gen)];
	}

	ConnectionGene Genome::getEnabledConnectionGene() const
	{
		std::vector<ConnectionGene> enabledConnectionGenes;
		for (const auto& connectionGene : connectionGenes)
		{
			if (connectionGene.isEnabled())
				enabledConnectionGenes.push_back(connectionGene);
		}

		if (enabledConnectionGenes.empty())
			return ConnectionGene{ 0, 0 };

		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<uint16_t> dis(0, enabledConnectionGenes.size() - 1);

		return enabledConnectionGenes[dis(gen)];
	}

	void Genome::addGene()
	{
		using namespace dnf_composer::element;

		auto randEnabledConnectionGene = getEnabledConnectionGene();
		if (randEnabledConnectionGene.getInGeneId() == 0)
			return;
		randEnabledConnectionGene.disable();
		const auto inGeneId = randEnabledConnectionGene.getParameters().inGeneId;
		const auto outGeneId = randEnabledConnectionGene.getParameters().outGeneId;
		const auto kernel = randEnabledConnectionGene.getKernel();
		const auto kernelParameters = std::dynamic_pointer_cast<GaussKernel>(kernel)->getParameters();

		const auto index = genes.size() + 1;
		genes.push_back(Gene({ GeneType::HIDDEN, static_cast<uint16_t>(index) }));

		// create two new connection genes
		const auto connectionGeneKernelParametersIn = GaussKernelParameters{ 5, 10, false, false };
		const auto connectionGeneKernelParametersOut = GaussKernelParameters{ kernelParameters.sigma, kernelParameters.amplitude, false, false };
		const ConnectionGene connectionGeneIn { inGeneId, static_cast<uint16_t>(index), connectionGeneKernelParametersIn };
		const ConnectionGene connectionGeneOut{ static_cast<uint16_t>(index), outGeneId, connectionGeneKernelParametersOut };
		connectionGenes.push_back(connectionGeneIn);
		connectionGenes.push_back(connectionGeneOut);
	}

	void Genome::mutateGene()
	{
		const auto geneId = getRandomGeneIdByType(GeneType::HIDDEN);
		if (geneId == 0)
			return;
		genes[geneId-1].mutate();
	}

	void Genome::addConnectionGene()
	{
		const std::tuple<uint16_t, uint16_t> geneTuple = getNewRandomGeneTuple();
		if (geneTuple == std::tuple<uint16_t, uint16_t>{})
			return;
		addConnectionGeneIfNewWithinGeneration(geneTuple);
	}

	void Genome::mutateConnectionGene()
	{
		if (connectionGenes.empty())
			return;

		const auto connectionGeneId = rand() % connectionGenes.size();
		connectionGenes[connectionGeneId].mutate();
	}
}