#include "neat/solution.h"

namespace neat_dnfs
{
	Solution::Solution(const SolutionTopology& initialTopology)
		: initialTopology(initialTopology),
		parameters(),
		phenotype(SimulationConstants::name, SimulationConstants::deltaT),
		genome()
	{
		if (initialTopology.numInputGenes < SolutionConstants::minInitialInputGenes ||
			initialTopology.numOutputGenes < SolutionConstants::minInitialOutputGenes)
			throw std::invalid_argument("Number of input and output genes must be greater than 0");
	}

	void Solution::initialize()
	{
		createInputGenes();
		createOutputGenes();
		createHiddenGenes();
		createRandomInitialConnectionGenes();
	}

	void Solution::createRandomInitialConnectionGenes()
	{
		for (int i = 0; i < initialTopology.numInputGenes * initialTopology.numOutputGenes; ++i)
			if (tools::utils::generateRandomDouble(0.0, 1.0) < SolutionConstants::initialConnectionProbability)
				genome.addRandomInitialConnectionGene();
	}

	void Solution::mutate()
	{
		genome.mutate();
	}

	Phenotype Solution::getPhenotype() const
	{
		return phenotype;
	}

	Genome Solution::getGenome() const
	{
		return genome;
	}

	SolutionParameters Solution::getParameters() const
	{
		return parameters;
	}

	double Solution::getFitness() const
	{
		return parameters.fitness;
	}

	size_t Solution::getGenomeSize() const
	{
		return genome.getConnectionGenes().size();
	}

	void Solution::clearGenerationalInnovations() const
	{
		// static member accessed through instance - readability?
		genome.clearGenerationalInnovations();
	}

	std::vector<uint16_t> Solution::getInnovationNumbers() const
	{
		return genome.getInnovationNumbers();
	}

	void Solution::buildPhenotype()
	{
		translateGenesToPhenotype();
		translateConnectionGenesToPhenotype();
	}

	void Solution::createInputGenes()
	{
		for (int j = 0; j < initialTopology.numInputGenes; j++)
			genome.addInputGene();
	}

	void Solution::createOutputGenes()
	{
		for (int j = 0; j < initialTopology.numOutputGenes; j++)
			genome.addOutputGene();
	}

	void Solution::createHiddenGenes()
	{
		for (int j = 0; j < initialTopology.numHiddenGenes; j++)
			genome.addHiddenGene();
	}

	void Solution::translateGenesToPhenotype()
	{
		using namespace dnf_composer::element;

		for (auto const& gene : genome.getFieldGenes())
		{
			const auto nfcp = gene.getNeuralField()->getElementCommonParameters();
			const auto nfp = gene.getNeuralField()->getParameters();
			const auto nf = std::make_shared<NeuralField>(nfcp, nfp);
			phenotype.addElement(nf);


			const auto kcp = gene.getKernel()->getElementCommonParameters();
			switch (kcp.identifiers.label)
			{
				case ElementLabel::GAUSS_KERNEL:
				{
					const auto gkp = std::dynamic_pointer_cast<GaussKernel>(gene.getKernel())->getParameters();
					const auto kernel = std::make_shared<GaussKernel>(kcp, gkp);
					phenotype.addElement(kernel);
					phenotype.createInteraction(nf->getUniqueName(), "output", kernel->getUniqueName());
					phenotype.createInteraction(kernel->getUniqueName(), "output", nf->getUniqueName());
					break;
				}
				case ElementLabel::MEXICAN_HAT_KERNEL:
				{
					const auto mhkp = std::dynamic_pointer_cast<MexicanHatKernel>(gene.getKernel())->getParameters();
					const auto kernel = std::make_shared<MexicanHatKernel>(kcp, mhkp);
					phenotype.addElement(kernel);
					phenotype.createInteraction(nf->getUniqueName(), "output", kernel->getUniqueName());
					phenotype.createInteraction(kernel->getUniqueName(), "output", nf->getUniqueName());
					break;
				}
				case ElementLabel::LATERAL_INTERACTIONS:
				{
					const auto lip = std::dynamic_pointer_cast<LateralInteractions>(gene.getKernel())->getParameters();
					const auto kernel = std::make_shared<LateralInteractions>(kcp, lip);
					phenotype.addElement(kernel);
					phenotype.createInteraction(nf->getUniqueName(), "output", kernel->getUniqueName());
					phenotype.createInteraction(kernel->getUniqueName(), "output", nf->getUniqueName());
					break;
				}
				default:
					throw std::invalid_argument("Invalid kernel label");
			}
		}
	}

	void Solution::clearPhenotype()
	{
		for (auto const& element : phenotype.getElements())
			phenotype.removeElement(element->getUniqueName());
	}

	void Solution::translateConnectionGenesToPhenotype()
	{
		if (phenotype.getElement("nf 2")->getInputs().size() > 1)
			std::cout << "nf 2 inputs: " << phenotype.getElement("nf 2")->getInputs().size() << std::endl;

		for (auto const& connectionGene : genome.getConnectionGenes())
		{
			if (connectionGene.isEnabled())
			{
				const auto kernel = connectionGene.getKernel();
				const auto sourceId = connectionGene.getInFieldGeneId();
				const auto targetId = connectionGene.getOutFieldGeneId();

				phenotype.addElement(kernel);
				phenotype.createInteraction("nf " + std::to_string(sourceId),
					"output", kernel->getUniqueName());
				phenotype.createInteraction(kernel->getUniqueName(),
					"output", "nf " + std::to_string(targetId));
			}
		}
	}

	void Solution::incrementAge()
	{
		parameters.age++;
	}

	void Solution::setAdjustedFitness(double adjustedFitness)
	{
		parameters.adjustedFitness = adjustedFitness;
	}

	void Solution::addFieldGene(const FieldGene& gene)
	{
		genome.addFieldGene(gene);
	}

	void Solution::addConnectionGene(const ConnectionGene& gene)
	{
		genome.addConnectionGene(gene);
	}

	bool Solution::containsConnectionGene(const ConnectionGene& gene) const
	{
		return genome.containsConnectionGene(gene);
	}

}