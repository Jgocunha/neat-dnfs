#include "neat/solution.h"

namespace neat_dnfs
{
	Solution::Solution(const SolutionTopology& initialTopology)
		: id(uniqueIdentifierCounter++),
		name("undefined"),
		initialTopology(initialTopology),
		parameters(),
		phenotype(SimulationConstants::name + std::to_string(id), SimulationConstants::deltaT),
		genome()
	{
		// views::keys can be used
		for(const auto& geneTypeAndDimension : initialTopology.geneTopology)
		{
			if (geneTypeAndDimension.first == FieldGeneType::INPUT)
				break;
		}
		for (const auto& geneTypeAndDimension : initialTopology.geneTopology)
		{
			if (geneTypeAndDimension.first == FieldGeneType::OUTPUT)
				return;
		}

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
		for (int i = 0; i < initialTopology.geneTopology.size(); ++i)
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
		clearPhenotype();
		translateGenesToPhenotype();
		translateConnectionGenesToPhenotype();
	}

	void Solution::createInputGenes()
	{
		for (const auto& gene : initialTopology.geneTopology)
			if (gene.first == FieldGeneType::INPUT)
				genome.addInputGene(gene.second);
	}

	void Solution::createOutputGenes()
	{
		for (const auto& gene : initialTopology.geneTopology)
			if (gene.first == FieldGeneType::OUTPUT)
				genome.addOutputGene(gene.second);
	}

	void Solution::createHiddenGenes()
	{
		for (const auto& gene : initialTopology.geneTopology)
			if (gene.first == FieldGeneType::HIDDEN)
				genome.addHiddenGene(gene.second);
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
				const auto coupling = connectionGene.getFieldCoupling();
				const auto sourceId = connectionGene.getInFieldGeneId();
				const auto targetId = connectionGene.getOutFieldGeneId();

				phenotype.addElement(coupling);
				phenotype.createInteraction("nf " + std::to_string(sourceId),
					"output", coupling->getUniqueName());
				phenotype.createInteraction(coupling->getUniqueName(),
					"output", "nf " + std::to_string(targetId));
			}
		}
	}

	bool Solution::containsConnectionGeneWithTheSameInputOutputPair(const ConnectionGene& gene) const
	{
		return genome.containsConnectionGeneWithTheSameInputOutputPair(gene);
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
			offspring->addFieldGene(gene.clone());

		const auto& parentConnectionGenes = moreFitParent->getGenome().getConnectionGenes();
		for (const auto& gene : parentConnectionGenes)
		{
			// Matching genes are inherited randomly from either parent
			if (lessFitParent->containsConnectionGene(gene))
			{
				const auto lessFitGene = lessFitParent->getGenome().getConnectionGeneByInnovationNumber(gene.getInnovationNumber());
				if (tools::utils::generateRandomInt(0, 1))
					offspring->addConnectionGene(gene.clone());
				else
					offspring->addConnectionGene(lessFitGene.clone());
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
						offspring->addConnectionGene(gene.clone());
				}
				else
					offspring->addConnectionGene(gene.clone());
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
					if (!moreFitParent->containsConnectionGeneWithTheSameInputOutputPair(gene))
					{
						if (tools::utils::generateRandomSignal())
						{
							offspring->addConnectionGene(gene.clone());
							// make sure the field genes are also added
							const uint16_t inFieldGeneId = gene.getInFieldGeneId();
							const uint16_t outFieldGeneId = gene.getOutFieldGeneId();
							for (const auto& fieldGene : lessFitParent->getGenome().getFieldGenes())
							{
								if (fieldGene.getParameters().id == inFieldGeneId || fieldGene.getParameters().id == outFieldGeneId)
									offspring->addFieldGene(fieldGene.clone());
							}

						}
					}
				}
			}

		}

		// check if there are connection genes with the same input and output field genes
		for (const auto& gene : offspring->getGenome().getConnectionGenes())
		{
			const auto inFieldGeneId = gene.getInFieldGeneId();
			const auto outFieldGeneId = gene.getOutFieldGeneId();
			for (const auto& otherGene : offspring->getGenome().getConnectionGenes())
			{
				if (gene.getInnovationNumber() != otherGene.getInnovationNumber() &&
					inFieldGeneId == otherGene.getInFieldGeneId() &&
					outFieldGeneId == otherGene.getOutFieldGeneId())
				{
					if (fitnessDifference < 1e-6)
						tools::logger::log(tools::logger::LogLevel::ERROR, "Crossover produced offspring with duplicate connection genes.");
					else
						tools::logger::log(tools::logger::LogLevel::WARNING, "Crossover produced offspring with duplicate connection genes.");
					break;
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
		for (int i = 0; i < iterations; ++i)
			phenotype.step();
	}

	bool Solution::runSimulationUntilFieldStable(const std::string& targetElement)
	{
		const auto neuralField = std::dynamic_pointer_cast<dnf_composer::element::NeuralField>(phenotype.getElement(targetElement));
		size_t counter = 0;
		do
		{
			counter++;
			phenotype.step();
			if (counter > SimulationConstants::maxSimulationSteps)
				return false;
		} while (!neuralField->isStable());
		return true;
	}

	void Solution::addGaussianStimulus(const std::string& targetElement, const dnf_composer::element::GaussStimulusParameters& stimulusParameters, 
		const dnf_composer::element::ElementDimensions& dimensions)
	{
		using namespace dnf_composer::element;

		const std::string gsId = "gs " + targetElement + " " + std::to_string(stimulusParameters.position);
		auto gaussStimulus = GaussStimulus{
			{gsId, dimensions}, stimulusParameters };
		phenotype.addElement(std::make_shared<GaussStimulus>(gaussStimulus));
		phenotype.createInteraction(gsId, "output", targetElement);
		gaussStimulus.init();
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

	bool Solution::hasTheSameTopology(const SolutionPtr& other) const
	{
		return initialTopology == other->initialTopology;
	}

	bool Solution::hasTheSameGenome(const SolutionPtr& other) const
	{
		return genome == other->genome;
	}

	bool Solution::hasTheSameParameters(const SolutionPtr& other) const
	{
		return parameters == other->parameters;
	}

	bool Solution::hasTheSamePhenotype(const SolutionPtr& other) const
	{
		log(tools::logger::LogLevel::FATAL, "Checking if phenotypes are the same. "
									  "Functionality still not implemented.");
		return false;
	}

	bool Solution::operator==(const SolutionPtr& other) const
	{
		return hasTheSameTopology(other) && hasTheSameGenome(other) && hasTheSameParameters(other);
	}

	std::string Solution::toString() const
	{
		std::stringstream address;
		address << this;
		std::string result = "Solution: \n";
		result += "Address: " + address.str() + "\n";
		result += "Topology: " + initialTopology.toString() + "\n";
		result += "Parameters: " + parameters.toString() + "\n";
		result += "Genome: " + genome.toString() + "\n";
		result += "Phenotype: \n";
		for(const auto& element : phenotype.getElements())
			result += element->toString() + "\n";
		return result;
	}

	void Solution::print() const
	{
		log(tools::logger::LogLevel::INFO, toString());
	}

}