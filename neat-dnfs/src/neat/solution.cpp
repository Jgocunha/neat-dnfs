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

	void Solution::evaluate()
	{
		buildPhenotype();
		testPhenotype();
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
		// log I am calling mutate() on solution with address: 0x7fffbf7f7f70
		std::cout << "I am calling mutate() on solution with address: " << this << std::endl;
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
		clearPhenotype();
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

			// check if neural field already exists
			for (const auto& element : phenotype.getElements())
			{
				if (element->getUniqueName() == nfcp.identifiers.uniqueName)
				{
					log(tools::logger::LogLevel::ERROR, "Neural field with unique name " + nfcp.identifiers.uniqueName + " already exists in phenotype.");
				}
			}

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
					throw std::invalid_argument("Invalid kernel label while translating genes to phenotype.");
			}
		}
	}

	void Solution::clearPhenotype()
	{
		for (auto const& element : phenotype.getElements())
			phenotype.removeElement(element->getUniqueName());
		if (!phenotype.getElements().empty())
			throw std::runtime_error("Phenotype elements were not cleared correctly.");
	}

	void Solution::translateConnectionGenesToPhenotype()
	{
		using namespace dnf_composer::element;

		for (auto const& connectionGene : genome.getConnectionGenes())
		{
			if (connectionGene.isEnabled())
			{
				const auto kernel = connectionGene.getKernel();
				const auto sourceId = connectionGene.getInFieldGeneId();
				const auto targetId = connectionGene.getOutFieldGeneId();

				/*const auto kcp = connectionGene.getKernel()->getElementCommonParameters();
				switch (kcp.identifiers.label)
				{
				case ElementLabel::GAUSS_KERNEL:
				{
					const auto gkp = std::dynamic_pointer_cast<GaussKernel>(connectionGene.getKernel())->getParameters();
					const auto kernel = std::make_shared<GaussKernel>(kcp, gkp);
					phenotype.addElement(kernel);
					phenotype.createInteraction("nf " + std::to_string(sourceId), "output", kernel->getUniqueName());
					phenotype.createInteraction(kernel->getUniqueName(), "output", "nf " + std::to_string(targetId));
					break;
				}
				case ElementLabel::MEXICAN_HAT_KERNEL:
				{
					const auto mhkp = std::dynamic_pointer_cast<MexicanHatKernel>(connectionGene.getKernel())->getParameters();
					const auto kernel = std::make_shared<MexicanHatKernel>(kcp, mhkp);
					phenotype.addElement(kernel);
					phenotype.createInteraction("nf " + std::to_string(sourceId), "output", kernel->getUniqueName());
					phenotype.createInteraction(kernel->getUniqueName(), "output", "nf " + std::to_string(targetId));
					break;
				}
				case ElementLabel::LATERAL_INTERACTIONS:
				{
					const auto lip = std::dynamic_pointer_cast<LateralInteractions>(connectionGene.getKernel())->getParameters();
					const auto kernel = std::make_shared<LateralInteractions>(kcp, lip);
					phenotype.addElement(kernel);
					phenotype.createInteraction("nf " + std::to_string(sourceId), "output", kernel->getUniqueName());
					phenotype.createInteraction(kernel->getUniqueName(), "output", "nf " + std::to_string(targetId));
					break;
				}
				default:
					throw std::invalid_argument("Invalid kernel label while translating genes to phenotype.");
				}*/


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

	SolutionPtr Solution::crossover(const SolutionPtr& other)
	{
		// if you are getting bad_weak_ptr exception when calling this function
		// you are probably calling it with a stack allocated object
		// use a heap allocated object instead
		// e.g. auto solution = std::make_shared<Solution>(topology);
		// instead of Solution solution(topology);
		// and then call solution->crossover(other);
		const SolutionPtr self = shared_from_this();

		const double fitnessDifference = std::abs(self->getFitness() - other->getFitness());
		const SolutionPtr moreFitParent = self->getFitness() > other->getFitness() ? self : other;
		const SolutionPtr lessFitParent = self->getFitness() > other->getFitness() ? other : self;

		SolutionPtr offspring = moreFitParent->clone();

		for (const auto& gene : moreFitParent->getGenome().getFieldGenes())
			offspring->addFieldGene(gene);

		const auto& parentConnectionGenes = moreFitParent->getGenome().getConnectionGenes();
		for (const auto& gene : parentConnectionGenes)
		{
			// Matching genes are inherited randomly from either parent
			if (lessFitParent->containsConnectionGene(gene))
			{
				const auto lessFitGene = lessFitParent->getGenome().getConnectionGeneByInnovationNumber(gene.getInnovationNumber());
				if (tools::utils::generateRandomInt(0, 1))
					offspring->addConnectionGene(gene);
				else
					offspring->addConnectionGene(lessFitGene);
			}
			else
			{
				// Disjoint and excess genes are inherited from the more fit parent
				// unless the fitness difference is 0, in which case the gene is inherited randomly
				// here we are only considering the most fit parent
				// later add randomly from the less fit parent
				if (fitnessDifference < 1e-6)
				{
					if (tools::utils::generateRandomInt(0, 1))
						offspring->addConnectionGene(gene);
				}
				else
					offspring->addConnectionGene(gene);
			}
		}

		// If the fitness is the same we still have to randomly inherit the excess and disjoint genes
		// from the less fit parent
		if (fitnessDifference < 1e-6)
		{
			const auto& lessFitParentConnectionGenes = lessFitParent->getGenome().getConnectionGenes();
			for (const auto& gene : lessFitParentConnectionGenes)
			{
				// Non-matching genes are inherited randomly from the less fit parent
				if (!moreFitParent->containsConnectionGene(gene))
				{
					if (tools::utils::generateRandomInt(0, 1))
					{
						offspring->addConnectionGene(gene);
						// make sure the field genes are also added
						const uint16_t inFieldGeneId = gene.getInFieldGeneId();
						const uint16_t outFieldGeneId = gene.getOutFieldGeneId();
						for (const auto& fieldGene : lessFitParent->getGenome().getFieldGenes())
						{
							if (fieldGene.getParameters().id == inFieldGeneId || fieldGene.getParameters().id == outFieldGeneId)
								offspring->addFieldGene(fieldGene);
						}

					}
				}
			}

		}

		return offspring;
	}

	void Solution::initSimulation()
	{
		phenotype.init();
	}

	void Solution::stopSimulation()
	{
		phenotype.close();
	}

	void Solution::runSimulation(const uint16_t iterations)
	{
		//const auto neuralField = std::dynamic_pointer_cast<dnf_composer::element::NeuralField>(phenotype.getElement("nf 2"));
		phenotype.step();
		//double highestActivation = neuralField->getHighestActivation();
		//std::cout << "Highest activation before input: " << highestActivation << std::endl;
		for (int i = 0; i < iterations; ++i)
			phenotype.step();
		//highestActivation = neuralField->getHighestActivation();
		//std::cout << "Highest activation after: " << highestActivation << std::endl << std::endl;
	}

	void Solution::runSimulationUntilFieldStable(const std::string& targetElement)
	{
		const auto neuralField = std::dynamic_pointer_cast<dnf_composer::element::NeuralField>(phenotype.getElement(targetElement));
		//neuralField->setThresholdForStability(0.0001);
		//double highestActivation = neuralField->getHighestActivation();
		//std::cout << "Highest activation before input: " << highestActivation << std::endl;
		//uint16_t steps = 0;
		do
		{
			//++steps;
			phenotype.step();
		} while (!neuralField->isStable());
		//highestActivation = neuralField->getHighestActivation();
		//std::cout << "Highest activation after: " << highestActivation << " in " << steps << " steps." << std::endl << std::endl;
	}


	void Solution::addGaussianStimulus(const std::string& targetElement, const dnf_composer::element::GaussStimulusParameters& parameters)
	{
		using namespace dnf_composer;
		using namespace dnf_composer::element;

		static const ElementSpatialDimensionParameters dimension = 
			phenotype.getElement(targetElement)->getElementCommonParameters().dimensionParameters;

		const std::string gsId = "gs " + targetElement + " " + std::to_string(parameters.position);
		const auto gaussStimulus = GaussStimulus{
			{gsId, dimension}, parameters };
		phenotype.addElement(std::make_shared<GaussStimulus>(gaussStimulus));
		phenotype.createInteraction(gsId, "output", targetElement);
	}

	void Solution::removeGaussianStimuli()
	{
		using namespace dnf_composer::element;

		for (const auto& element : phenotype.getElements())
		{
			if (element->getLabel() == GAUSS_STIMULUS)
				phenotype.removeElement(element->getUniqueName());
		}
	}

}