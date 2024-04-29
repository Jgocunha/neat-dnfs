#include "phenotype.h"

namespace neat_dnfs
{
	Phenotype::Phenotype()
	{
		architecture = dnf_composer::createSimulation("phenotype", 25.0, 0.0, 0.0);
	}

	std::shared_ptr<Architecture> Phenotype::getArchitecture() const
	{
		return architecture;
	}
	  
	void Phenotype::buildFromGenome(const Genome& genome)
	{
		// for each gene in the genome
		// add neural fields and self-excitation kernels
		for (auto const gene : genome.getGenes())
		{
			const auto nf = gene.getNeuralField();
			const auto kernel = gene.getKernel();
			architecture->addElement(nf);
			architecture->addElement(kernel);
			architecture->createInteraction(nf->getUniqueName(), "output", kernel->getUniqueName());
			architecture->createInteraction(kernel->getUniqueName(), "output", nf->getUniqueName());
		}

		// for each connection in the genome
		// add interaction kernels
		for (auto const connectionGene : genome.getConnectionGenes())
		{
			const auto kernel = connectionGene.getKernel();
			const auto sourceId = connectionGene.getInGeneId();
			const auto targetId = connectionGene.getOutGeneId();

			architecture->addElement(kernel);
			architecture->createInteraction("nf " + std::to_string(sourceId), "output", kernel->getUniqueName());
			architecture->createInteraction(kernel->getUniqueName(), "output", "nf " + std::to_string(targetId));
		}

	}

	double Phenotype::evaluateFitness()
	{
		using namespace dnf_composer;

		// add inputs to the neural fields
		const element::GaussStimulusParameters gcp_a = { 5, 25, 50, false, false };
		const std::shared_ptr<element::GaussStimulus> gauss_stimulus
		(new element::GaussStimulus({ "gauss stimulus", {100, 1.0 }}, gcp_a));
		architecture->addElement(gauss_stimulus);
		architecture->createInteraction(gauss_stimulus->getUniqueName(), "output", "nf 1");

		// simulate behavior
		static constexpr int numSteps = 1000;
		architecture->init();
		for(int i = 0; i < numSteps; i++)
			architecture->step();
		const auto ael = std::dynamic_pointer_cast<element::NeuralField>(architecture->getElement("nf 3"));
		const double centroid = ael->getCentroid();
		std::cout << "Centroid: " << centroid << std::endl;
		architecture->close();

		// Calculate fitness
		static constexpr double targetCentroid = 50;
		const double fitness = 1 / (1 + std::abs(centroid - targetCentroid));
		return fitness;
	}
}