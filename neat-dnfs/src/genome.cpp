#include "genome.h"

namespace neat_dnfs
{
	Genome::Genome()
	{
		genes = std::vector<Gene>();
		connectionGenes = std::vector<ConnectionGene>();
		parameters = GenomeParameters();
	}

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

	void Genome::addConnectionGene()
	{
		if (genes.size() < 2)
			throw std::invalid_argument("There must be at least two genes to add a connection gene.");

		const auto inGeneId = getRandomGeneIdByType(GeneType::INPUT);
		const auto outGeneId = getRandomGeneIdByType(GeneType::OUTPUT);

		connectionGenes.emplace_back(inGeneId, outGeneId);
	}

	GenomeParameters Genome::getParameters() const
	{
		return parameters;
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