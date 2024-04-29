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

	void Genome::addConnectionGene()
	{
		if (genes.size() < 2)
			throw std::invalid_argument("There must be at least two genes to add a connection gene.");

		const auto inGeneId = getRandomGeneIdByType(GeneType::INPUT);
		const auto outGeneId = getRandomGeneIdByType(GeneType::OUTPUT);

		connectionGenes.emplace_back(inGeneId, outGeneId);
	}

	Genome Genome::clone()
	{
		Genome clone = *this;
		return *this;
	}


	//void Genome::evaluate()
	//{
	//	using namespace dnf_composer;
	//	static constexpr int numSteps = 100;
	//	static constexpr int xSize = 100;
	//	static constexpr double dx = 1.0;

	//	Simulation simulation{ "sim", 25, 0, 0 };

	//	for (const auto& gene : genes)
	//	{
	//		// add neural fields and self-excitation kernels
	//		const auto nf = gene.getNeuralField();
	//		simulation.addElement(nf);
	//		const auto kernel = gene.getKernel();
	//		simulation.addElement(kernel);
	//		simulation.createInteraction(nf->getUniqueName(), "output", kernel->getUniqueName());
	//		simulation.createInteraction(kernel->getUniqueName(), "output", nf->getUniqueName());

	//		const element::NormalNoiseParameters nnp = { 0.3 };
	//		const std::shared_ptr<element::NormalNoise> noise(new element::NormalNoise({ "noise " + nf->getUniqueName(), {100, 1.0} }, nnp));
	//		simulation.addElement(noise);
	//		simulation.createInteraction(noise->getUniqueName(), "output", nf->getUniqueName());
	//	}

	//	for (const auto& connectionGene : connectionGenes)
	//	{
	//		// add connection genes
	//		const auto kernel = connectionGene.getKernel();
	//		simulation.addElement(kernel);
	//		const ConnectionGeneParameters conGenParams = connectionGene.getParameters();

	//		Gene inputGene = genes[conGenParams.inGeneId - 1];
	//		const auto inputNf = inputGene.getNeuralField();
	//		simulation.createInteraction(inputNf->getUniqueName(), "output", kernel->getUniqueName());

	//		Gene outputGene = genes[conGenParams.outGeneId - 1];
	//		const auto outputNf = outputGene.getNeuralField();
	//		simulation.createInteraction(kernel->getUniqueName(), "output", outputNf->getUniqueName());
	//	}

	//	// add inputs
	//	const element::GaussStimulusParameters gcp_a = { 5, 10, 50, false, false };
	//	const std::shared_ptr<element::GaussStimulus> gauss_stimulus
	//	(new element::GaussStimulus({ "gauss stimulus", {100, 1.0 }}, gcp_a));
	//	simulation.addElement(gauss_stimulus);
	//	simulation.createInteraction(gauss_stimulus->getUniqueName(), "output", genes[0].getNeuralField()->getUniqueName());

	//	// Simulate behavior
	//	simulation.init();
	//	for(int i = 0; i < numSteps; i++)
	//		simulation.step();
	//	const auto ael = std::dynamic_pointer_cast<element::NeuralField>(simulation.getElement("nf 2"));
	//	const double centroid = ael->getCentroid();
	//		simulation.close();

	//	// Calculate fitness
	//	static constexpr double targetCentroid = 50;
	//	parameters.fitness = 1 / (1 + std::abs(centroid - targetCentroid));
	//}

	//GenomeParameters Genome::getParameters() const
	//{
	//	return parameters;
	//}

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