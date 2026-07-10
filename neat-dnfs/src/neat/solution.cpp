#include <utility>

#include "neat/solution.h"
#include <format> 

namespace neat_dnfs
{
	Solution::Solution(const SolutionTopology& initialTopology)
		: id(uniqueIdentifierCounter++),
		name("undefined"),
		initialTopology(initialTopology),
		parameters(),
		phenotype(std::format("{}{}", SimulationConstants::name, id), SimulationConstants::deltaT), 
		genome(),
		parents(0,0)
	{
		bool hasInput = false, hasOutput = false;
		for (const auto& geneTypeAndDimension : initialTopology.geneTopology)
		{
			if (geneTypeAndDimension.first == FieldGeneType::INPUT)
			{
				hasInput = true;
			}
			if (geneTypeAndDimension.first == FieldGeneType::OUTPUT)
			{
				hasOutput = true;
			}
		}
		if (!hasInput || !hasOutput)
		{
			throw std::invalid_argument("Number of input and output genes must be greater than 0");
		}
	}

	Solution::Solution(SolutionTopology initialTopology, dnf_composer::Simulation  phenotype)
		: id(uniqueIdentifierCounter++),
		name("undefined"),
		initialTopology(std::move(initialTopology)),
		parameters(),
		phenotype(std::move(phenotype)),
		genome(),
		parents(0, 0)
	{
		translatePhenotypeToGenome();
		clearPhenotype();
		this->phenotype = dnf_composer::Simulation(std::format("{}{}", SimulationConstants::name, id), SimulationConstants::deltaT);
	}

	void Solution::evaluate()
	{
		buildPhenotype();
		testPhenotype();
		clearPhenotype();
	}

	void Solution::initialize()
	{
		if (genome.isEmpty())
		{
			createInputGenes();
			createOutputGenes();
		}
	}

	void Solution::mutate()
	{
		genome.mutate();
	}

	void Solution::setSpeciesId(int speciesId)
	{
		parameters.speciesId = speciesId;
	}

	void Solution::setParents(int parent1, int parent2)
	{
		parents = std::make_tuple(parent1, parent2);
	}

	dnf_composer::Simulation Solution::getPhenotype() const
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

	std::string Solution::getAddress() const
	{
		std::stringstream address;
		address << this;
		return address.str();
	}

	double Solution::getFitness() const
	{
		return parameters.fitness;
	}

	size_t Solution::getGenomeSize() const
	{
		return genome.getConnectionGenes().size();
	}

	void Solution::clearGenerationalInnovations()
	{
		Genome::clearGenerationalInnovations();
	}

	std::vector<int> Solution::getInnovationNumbers() const
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
		for (const auto& gene : initialTopology.geneTopology)
		{
			if (gene.first == FieldGeneType::INPUT)
			{
				genome.addInputGene(gene.second);
			}
		}
	}

	void Solution::createOutputGenes()
	{
		for (const auto& gene : initialTopology.geneTopology)
		{
			if (gene.first == FieldGeneType::OUTPUT)
			{
				genome.addOutputGene(gene.second);
			}
		}
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
					log(tools::logger::LogLevel::ERROR, std::format("Neural field with unique name {} already exists in phenotype.", nfcp.identifiers.uniqueName));	
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
			case ElementLabel::OSCILLATORY_KERNEL:
				{
					const auto okp = std::dynamic_pointer_cast<OscillatoryKernel>(gene.getKernel())->getParameters();
					const auto kernel = std::make_shared<OscillatoryKernel>(kcp, okp);
					phenotype.addElement(kernel);
					phenotype.createInteraction(nf->getUniqueName(), "output", kernel->getUniqueName());
					phenotype.createInteraction(kernel->getUniqueName(), "output", nf->getUniqueName());
					break;
				}
				default:
					throw std::invalid_argument("Invalid kernel label while translating genes to phenotype.");
			}

			const auto nncp = gene.getNoise()->getElementCommonParameters();
			const auto nnp = gene.getNoise()->getParameters();
			const auto nn = std::make_shared<NormalNoise>(nncp, nnp);
			phenotype.addElement(nn);
			phenotype.createInteraction(nn->getUniqueName(), "output", nf->getUniqueName());
		}
	}

	void Solution::clearPhenotype()
	{
		phenotype.close();
		// remove all interactions
		for (const auto& element : phenotype.getElements())
		{
			for (const auto& interaction : element->getInputs())
			{
				element->removeInputs();
				element->removeOutputs();
			}
		}
		// remove all elements
		for (const auto& element : phenotype.getElements())
		{
			phenotype.removeElement(element->getUniqueName());
		}
		// check if elements were removed
		phenotype.clean();
		if (!phenotype.getElements().empty())
		{
			throw std::runtime_error("Phenotype elements were not cleared correctly.");
		}
	}

	void Solution::translateConnectionGenesToPhenotype()
	{
		using namespace dnf_composer::element;

		for (auto const& connectionGene : genome.getConnectionGenes())
		{
			if (connectionGene.isEnabled())
			{
				auto coupling = connectionGene.getKernel();
				const auto sourceId = connectionGene.getInFieldGeneId();
				const auto targetId = connectionGene.getOutFieldGeneId();

				phenotype.addElement(coupling);
				phenotype.createInteraction(std::format("nf {}", sourceId), "output", coupling->getUniqueName());
				phenotype.createInteraction(coupling->getUniqueName(), "output", std::format("nf {}", targetId));
			}
		}
	}

	void Solution::translatePhenotypeToGenome()
	{
		// Clear the current genome before rebuilding it
		genome = Genome();

		if (phenotype.getElements().empty())
		{
			//tools::logger::log(tools::logger::LogLevel::WARNING, "Phenotype is empty. Cannot translate to genome.");
			return;
		}

		using namespace dnf_composer::element;

		// Map to track the field gene IDs by neural field names
		std::map<std::string, int> fieldNameToIdMap;
		int nextFieldId = 1;

		// First pass: identify all neural fields and create field genes
		for (const auto& element : phenotype.getElements())
		{
			if (element->getLabel() == ElementLabel::NEURAL_FIELD)
			{
				const auto neuralField = std::dynamic_pointer_cast<NeuralField>(element);
				const auto nfcp = neuralField->getElementCommonParameters();
				const auto nfp = neuralField->getParameters();

				// Determine a field gene type based on naming convention
				FieldGeneType fieldType;
				if (nfcp.identifiers.uniqueName.find(NeuralFieldConstants::namePrefix) == 0)
				{
					const std::string idStr = nfcp.identifiers.uniqueName.substr(NeuralFieldConstants::namePrefix.length());
					const int fieldId = std::stoi(idStr);

					// Store mapping for later use with connections
					fieldNameToIdMap[nfcp.identifiers.uniqueName] = fieldId;

					// Determine if this is input, output, or hidden based on connections
									// Logic based on connection patterns
					const size_t numInputs = element->getInputs().size();
					const size_t numOutputs = element->getOutputs().size();

					if (numInputs == 2 && numOutputs >= 1)
					{
						fieldType = FieldGeneType::INPUT;
					}
					else if (numInputs >= 3 && numOutputs == 1)
					{
						fieldType = FieldGeneType::OUTPUT;
					}
					else if (numInputs >= 3 && numOutputs >= 2)
					{
						fieldType = FieldGeneType::HIDDEN;
					}
					else
					{
						// Default to HIDDEN if the connection pattern doesn't match expected patterns
						fieldType = FieldGeneType::HIDDEN;
						tools::logger::log(tools::logger::LogLevel::WARNING,
							std::format("Unusual connection pattern for neural field: {} (inputs: {}, outputs: {})", 
								nfcp.identifiers.uniqueName, numInputs, numOutputs));
					}

					// Create field gene parameters
					FieldGeneParameters params(fieldType, fieldId);

					// Find associated kernel and noise for this neural field
					KernelPtr associatedKernel = nullptr;
					NormalNoisePtr associatedNoise = nullptr;

					for (const auto& outputInteraction : element->getOutputs())
					{
						//const auto targetElement = outputInteraction->;
						if (outputInteraction->getLabel() == ElementLabel::GAUSS_KERNEL ||
							outputInteraction->getLabel() == ElementLabel::MEXICAN_HAT_KERNEL ||
							outputInteraction->getLabel() == ElementLabel::OSCILLATORY_KERNEL)
						{
							if (outputInteraction->getInputs() ==  outputInteraction->getOutputs())
							{
								associatedKernel = std::dynamic_pointer_cast<Kernel>(outputInteraction);
								//std::cout << "Kernel: " << associatedKernel->toString() << std::endl;
								break;
							}
						}
					}

					for (const auto& inputInteraction : element->getInputs())
					{
						//const auto sourceElement = inputInteraction->getSource();
						if (inputInteraction->getLabel() == ElementLabel::NORMAL_NOISE)
						{
							associatedNoise = std::dynamic_pointer_cast<NormalNoise>(inputInteraction);
							break;
						} 
					}

					// If kernel and noise are found, add the field gene
					if (associatedKernel && associatedNoise)
					{
						FieldGene fieldGene(params, neuralField, associatedKernel);
						genome.addFieldGene(fieldGene);
					}
					else
					{
						tools::logger::log(tools::logger::LogLevel::WARNING,
							std::format("Could not find associated kernel or noise for neural field: {}", nfcp.identifiers.uniqueName));
					}

					// Update nextFieldId if necessary
					nextFieldId = std::max(nextFieldId, fieldId + 1);
				}
			}
		}

		int innovationCounter = 1;

		// Second pass: identify all connections between neural fields and create connection genes
		for (const auto& element : phenotype.getElements())
		{
			// Check if element is a kernel used for connection between neural fields
			if (element->getLabel() == ElementLabel::GAUSS_KERNEL ||
				element->getLabel() == ElementLabel::MEXICAN_HAT_KERNEL ||
				element->getLabel() == ElementLabel::OSCILLATORY_KERNEL)
			{
				// Skip self-connection kernels (which are part of field genes)
				bool isSelfConnection = false;
				std::string sourceName, targetName;

				// Find source and target of this connection
				for (const auto& inputInteraction : element->getInputs())
				{
					//const auto sourceElement = inputInteraction->getSource();
					if (inputInteraction->getLabel() == ElementLabel::NEURAL_FIELD)
					{
						sourceName = inputInteraction->getUniqueName();
					}
				}

				for (const auto& outputInteraction : element->getOutputs())
				{
					//const auto targetElement = outputInteraction->getTarget();
					if (outputInteraction->getLabel() == ElementLabel::NEURAL_FIELD)
					{
						targetName = outputInteraction->getUniqueName();
					}
				}

				// Skip if this is a self-connection (part of a field gene)
				if (sourceName == targetName)
				{
					continue;
				}

				// If we have valid source and target field names in our map
				if (fieldNameToIdMap.find(sourceName) != fieldNameToIdMap.end() &&
					fieldNameToIdMap.find(targetName) != fieldNameToIdMap.end())
				{
					int sourceId = fieldNameToIdMap[sourceName];
					int targetId = fieldNameToIdMap[targetName];

					// Create connection tuple
					ConnectionTuple connectionTuple(sourceId, targetId);

					// Create a connection gene with an appropriate innovation number
					// For reconstructing, we'll use a simple incremental approach
					//static int innovationCounter = 1;

					// Get the kernel parameters based on type
					switch (element->getLabel())
					{
					case ElementLabel::GAUSS_KERNEL:
					{
						auto gaussKernel = std::dynamic_pointer_cast<GaussKernel>(element);
						ConnectionGene connectionGene(connectionTuple, innovationCounter++, gaussKernel->getParameters());
						genome.addConnectionGene(connectionGene);
						break;
					}
					case ElementLabel::MEXICAN_HAT_KERNEL:
					{
						auto mexicanHatKernel = std::dynamic_pointer_cast<MexicanHatKernel>(element);
						ConnectionGene connectionGene(connectionTuple, innovationCounter++, mexicanHatKernel->getParameters());
						genome.addConnectionGene(connectionGene);
						break;
					}
					default:
						break;
					}

					//Genome::setNextInnovationNumber(innovationCounter); !!
				}
			}
		}

		// Ensure the genome is valid by adding any missing input or output genes from topology
		for (const auto& geneTypeAndDimension : initialTopology.geneTopology)
		{
			if (geneTypeAndDimension.first == FieldGeneType::INPUT)
			{
				bool found = false;
				for (const auto& fieldGene : genome.getFieldGenes())
				{
					if (fieldGene.getParameters().type == FieldGeneType::INPUT)
					{
						found = true;
						break;
					}
				}

				if (!found)
				{
					genome.addInputGene(geneTypeAndDimension.second);
				}
			}
			else if (geneTypeAndDimension.first == FieldGeneType::OUTPUT)
			{
				bool found = false;
				for (const auto& fieldGene : genome.getFieldGenes())
				{
					if (fieldGene.getParameters().type == FieldGeneType::OUTPUT)
					{
						found = true;
						break;
					}
				}

				if (!found)
				{
					genome.addOutputGene(geneTypeAndDimension.second);
				}
			}
		}

		// Final validation: check for duplicate connection genes
		genome.checkForDuplicateConnectionGenes();
	}

	void Solution::clearGenome()
	{
		genome = Genome();
	}

	void Solution::clearLastMutations()
	{
		genome.clearLastMutations();
	}

	void Solution::resetUniqueIdentifier()
	{
		uniqueIdentifierCounter = 0;
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
		offspring->setParents(moreFitParent->getId(), lessFitParent->getId());
		offspring->clearGenome();

		for (const auto& gene : moreFitParent->getGenome().getFieldGenes())
		{
			offspring->addFieldGene(gene.clone());
		}

		const auto& parentConnectionGenes = moreFitParent->getGenome().getConnectionGenes();
		for (const auto& gene : parentConnectionGenes)
		{
			// Matching genes are inherited randomly from either parent
			if (lessFitParent->containsConnectionGene(gene))
			{
				const auto lessFitGene = lessFitParent->getGenome().getConnectionGeneByInnovationNumber(gene.getInnovationNumber());
				if (tools::utils::generateRandomInt(0, 1))
				{
					offspring->addConnectionGene(gene.clone());
				}
				else
				{
					offspring->addConnectionGene(lessFitGene.clone());
				}
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
					{
						offspring->addConnectionGene(gene.clone());
					}
				}
				else
				{
					offspring->addConnectionGene(gene.clone());
				}
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
						const bool randomTrueOrFalse = tools::utils::generateRandomInt(0, 1);
						if (randomTrueOrFalse)
						{
							offspring->addConnectionGene(gene.clone());
							// make sure the field genes are also added
							const int inFieldGeneId = gene.getInFieldGeneId();
							const int outFieldGeneId = gene.getOutFieldGeneId();
							for (const auto& fieldGene : lessFitParent->getGenome().getFieldGenes())
							{
								if (fieldGene.getParameters().id == inFieldGeneId || fieldGene.getParameters().id == outFieldGeneId)
								{
									offspring->addFieldGene(fieldGene.clone());
								}
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
					{
						tools::logger::log(tools::logger::LogLevel::ERROR, "Crossover produced offspring with duplicate connection genes.");
					}
					else
					{
						tools::logger::log(tools::logger::LogLevel::WARNING, "Crossover produced offspring with duplicate connection genes.");
					}
					break;
				}
			}
		}


		return offspring;
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

	std::string Solution::toString() const
	{
		// solution id [ age, fit. spec., adj. fit., Parents, Genome, Mutations]
		return std::format("solution {} [{}, ({}, {}), {}, last mutations{{{}}}]",
			id,
			parameters.toString(),
			std::get<0>(parents),
			std::get<1>(parents),
			genome.toString(),
			genome.getMutationsInLastGeneration());
	}

	void Solution::print() const
	{
		log(tools::logger::LogLevel::INFO, toString()); 
	}

	// Solution evaluation specific functions

	void Solution::initSimulation()
	{
		phenotype.init();
	}

	void Solution::stopSimulation()
	{
		phenotype.close();
	}

	void Solution::runSimulation(const int iterations)
	{
		for (int i = 0; i < iterations; ++i)
		{
			phenotype.step();
		}
	}

	void Solution::addGaussianStimulus(const std::string& targetElement, const dnf_composer::element::GaussStimulusParameters& stimulusParameters, 
		const dnf_composer::element::ElementDimensions& dimensions)
	{
		using namespace dnf_composer::element;

		const std::string gsId = std::format("gs {} {}", targetElement, stimulusParameters.position);
		const auto gaussStimulus = std::make_shared<GaussStimulus>(GaussStimulus{ { gsId, dimensions }, stimulusParameters });
		phenotype.addElement(gaussStimulus);
		phenotype.createInteraction(gsId, "output", targetElement);
		gaussStimulus->init();
	}

	void Solution::removeGaussianStimuli()
	{
		using namespace dnf_composer::element;

		for (const auto& element : phenotype.getElements())
		{
			if (element->getLabel() == GAUSS_STIMULUS)
			{
				element->removeInputs();
				element->removeOutputs();
				phenotype.removeElement(element->getUniqueName());
			}
		}
	}

	void Solution::removeGaussianStimuliFromField(const std::string& fieldName)
	{
		using namespace dnf_composer::element;
		const auto neuralField = std::dynamic_pointer_cast<NeuralField>(phenotype.getElement(fieldName));

		for (const auto& input : neuralField->getInputs())
		{
			if (input->getLabel() == GAUSS_STIMULUS)
			{
				input->removeInputs();
				input->removeOutputs();
				phenotype.removeElement(input->getUniqueName());
			}
		}
	}

	void Solution::setGaussianStimulusParameters(const std::string& stimulusName, const dnf_composer::element::GaussStimulusParameters& parameters) const
	{
		using namespace dnf_composer::element;
		const auto gaussStimulus = std::dynamic_pointer_cast<GaussStimulus>(phenotype.getElement(stimulusName));
		gaussStimulus->setParameters(parameters);
	}

	double Solution::closenessToRestingLevel(const std::string& fieldName) const
	{
		// the highest value of activation should be equal to the resting level
		// the farther it is from the resting level, the lower the fitness (0.0)
		// the closer it is to the resting level, the higher the fitness (1.0)
		using namespace dnf_composer::element;
		const auto neuralField = std::dynamic_pointer_cast<NeuralField>(phenotype.getElement(fieldName));

		const double highestActivationValue = neuralField->getHighestActivation();
		const double restingLevel = neuralField->getParameters().startingRestingLevel;

		return 1.0 / (1.0 + std::abs(highestActivationValue - restingLevel));
	}

	double Solution::noBumps(const std::string& fieldName) const
	{
		using namespace dnf_composer::element;
		const auto neuralField = std::dynamic_pointer_cast<NeuralField>(phenotype.getElement(fieldName));
		const double highestActivation = neuralField->getHighestActivation();

		// If activation is below 0, return maximum fitness of 1.0
		if (highestActivation < 0.0)
		{
			return 1.0;
		}

		// For positive activations, apply exponential decay
		// The decay rate can be adjusted with the constant (5.0 here)
		// A larger value will make it decline more steeply
		static constexpr double decayRate = 10.0;
		return exp(-decayRate * highestActivation);
	}

	double Solution::iterationsUntilBump(const std::string& fieldName, const double targetIterations, const double maxIterations, const double tolerance)
	{
		using namespace dnf_composer::element;
		const auto neuralField = std::dynamic_pointer_cast<NeuralField>(phenotype.getElement(fieldName));
		int it = 0;
		do
		{
			phenotype.step();
			it++;
			if (!neuralField->getBumps().empty())
			{
				const double sigma = 6.0 * tolerance; // smoother shoulders; the higher the constant the smoother
				return tools::utils::normalizeWithFlatheadGaussian(
					it,
					targetIterations - tolerance,
					targetIterations + tolerance,
					sigma
				);
			}

		} while (it < maxIterations);

		return 0.0f;
	}

	double Solution::iterationsUntilNoBump(const std::string& fieldName, const double targetIterations, const double maxIterations, const double tolerance)
	{
		using namespace dnf_composer::element;
		const auto neuralField = std::dynamic_pointer_cast<NeuralField>(phenotype.getElement(fieldName));

		int it = 0;
		do
		{
			phenotype.step();
			it++;
			if (neuralField->getBumps().empty())
			{
				const double sigma = 6.0 * tolerance; // smoother shoulders; the higher the constant the smoother
				return tools::utils::normalizeWithFlatheadGaussian(
					it,
					targetIterations - tolerance,
					targetIterations + tolerance,
					sigma
				);
			}

		} while (it < maxIterations);

		return 0.0f;
	}

	double Solution::justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth(const std::string& fieldName, const std::vector<double>& positions, const double& amplitude, const double& width) const
	{
		using namespace dnf_composer::element;
		const auto neuralField = std::dynamic_pointer_cast<NeuralField>(phenotype.getElement(fieldName));

		static constexpr double wBumps  = 0.55;
		static constexpr double wPos    = 0.35;
		static constexpr double wAmp    = 0.05;
		static constexpr double wWidth  = 0.05;

		const int n = static_cast<int>(neuralField->getBumps().size());
		if (n == 0)
		{
			return 0.0;
		}
		if (n != 1)
		{
			return 0.2 * (wBumps / (1.0 + std::abs(1 - n)));
		}

		const NeuralFieldBump bump = neuralField->getBumps().front();

		// find distance to the closest valid position
		double minDistance = std::numeric_limits<double>::max();
		for (double p : positions)
		{
			minDistance = std::min(minDistance, std::abs(bump.centroid - p));
		}

		constexpr double epsilon = DimensionConstants::xSize / 20;

		// If the bump is not near any allowed position, do NOT give the big reward.
		if (minDistance >= epsilon) {
			// Return a small score that still prefers one bump vs. zero/two,
			// but is nowhere near "success".
			return 0.40 * (wBumps / (1.0 + std::abs(1 - n)));
		}

		// Otherwise, compute the full score.
		double fitness = 0.0;
		fitness += wBumps; // full credit for the correct bump count (since it's 1)
		fitness += wPos / (1.0 + minDistance);
		fitness += wAmp / (1.0 + std::abs(bump.amplitude - amplitude));
		fitness += wWidth / (1.0 + std::abs(bump.width - width));

		return fitness;
	}

	double Solution::oneBumpAtPositionWithAmplitudeAndWidth(const std::string& fieldName, const double& position, const double& 
		amplitude, const double& width) const
	{
		// if the field name is not in the phenotype, throw exception
		// ... .containsElement(name);
		static constexpr double weightBumps = 0.45;
		static constexpr double weightPos = 0.45;
		static constexpr double weightAmp = 0.05;
		static constexpr double weightWidth = 0.05;
		// if the sum of weights is not 1.0, throw exception
		if (std::abs(weightBumps + weightPos + weightAmp + weightWidth - 1.0) > 1e-6)
		{
			throw std::invalid_argument("Sum of weights must be 1.0");
		}

		static constexpr int targetNumberOfBumps = 1;
		double fitness = 0.0;

		using namespace dnf_composer::element;
		const auto neuralField = std::dynamic_pointer_cast<NeuralField>(phenotype.getElement(fieldName));
		const auto& bumps = neuralField->getBumps();
		const int numberOfBumps = static_cast<int>(neuralField->getBumps().size());
		if (numberOfBumps == 0)
		{
			return fitness;
		}

		fitness += weightBumps / ( 1.0 + std::abs(targetNumberOfBumps - numberOfBumps));

		NeuralFieldBump closestBump = bumps.front();
		for (const auto& bump : neuralField->getBumps())
		{
			if (std::abs(bump.centroid - position) < std::abs(closestBump.centroid - position))
			{
				closestBump = bump;
			}
		}

		fitness += weightPos / (1.0 + std::abs(closestBump.centroid - position));
		fitness += weightAmp / (1.0 + std::abs(closestBump.amplitude - amplitude));
		fitness += weightWidth / (1.0 + std::abs(closestBump.width - width));

		return fitness;
	}

	double Solution::twoBumpsAtPositionWithAmplitudeAndWidth(const std::string& fieldName, const double& position1, const double& amplitude1, const double& width1, const double& position2, const double& amplitude2, const double& width2) const
	{
		// can select the same bump twice (closest to position1 and closest to position2 might be
		// the same bump). That can inflate fitness even with only one real bump.

		static constexpr int targetNumberOfBumps = 2;
		static constexpr double weightBumps = 0.70;
		static constexpr double weightPos = 0.20 / targetNumberOfBumps;
		static constexpr double weightAmp = 0.05 / targetNumberOfBumps;
		static constexpr double weightWidth = 0.05 / targetNumberOfBumps;
		// if sum of weights is not 1.0, throw exception
		if (std::abs(weightBumps + (weightPos + weightAmp + weightWidth) * targetNumberOfBumps - 1.0) > 1e-6)
		{
			throw std::invalid_argument("Sum of weights must be 1.0");
		}
		double fitness = 0.0;

		using namespace dnf_composer::element;

		const auto neuralField = std::dynamic_pointer_cast<NeuralField>(phenotype.getElement(fieldName));
		const int numberOfBumps = static_cast<int>(neuralField->getBumps().size());
		fitness += weightBumps / (1.0 + std::abs(targetNumberOfBumps - numberOfBumps));

		NeuralFieldBump closestBump1;
		for (const auto& bump : neuralField->getBumps())
		{
			if (std::abs(bump.centroid - position1) < std::abs(closestBump1.centroid - position1))
			{
				closestBump1 = bump;
			}
		}
		fitness += weightPos / (1.0 + std::abs(closestBump1.centroid - position1));
		fitness += weightAmp / (1.0 + std::abs(closestBump1.amplitude - amplitude1));
		fitness += weightWidth / (1.0 + std::abs(closestBump1.width - width1));

		NeuralFieldBump closestBump2;
		for (const auto& bump : neuralField->getBumps())
		{
			if (std::abs(bump.centroid - position2) < std::abs(closestBump2.centroid - position2))
			{
				closestBump2 = bump;
			}
		}
		fitness += weightPos / (1.0 + std::abs(closestBump2.centroid - position2));
		fitness += weightAmp / (1.0 + std::abs(closestBump2.amplitude - amplitude2));
		fitness += weightWidth / (1.0 + std::abs(closestBump2.width - width2));

		return fitness;

	}

	double Solution::threeBumpsAtPositionWithAmplitudeAndWidth(const std::string& fieldName, const double& position1, const double& amplitude1, const double& width1, const double& position2, const double& amplitude2, const double& width2, const double& position3, const double& amplitude3, const double& width3) const
	{
		static constexpr int targetNumberOfBumps = 3;
		static constexpr double weightBumps = 0.40;
		static constexpr double weightPos = 0.20 / targetNumberOfBumps;
		static constexpr double weightAmp = 0.20 / targetNumberOfBumps;
		static constexpr double weightWidth = 0.20 / targetNumberOfBumps;
		// if sum of weights is not 1.0, throw exception
		if (std::abs(weightBumps + (weightPos + weightAmp + weightWidth) * targetNumberOfBumps - 1.0) > 1e-6)
		{
			tools::logger::log(tools::logger::LogLevel::ERROR, "Sum of weights must be 1.0 in three bump fitness evaluation.");
			throw std::invalid_argument("Sum of weights must be 1.0 in three bump fitness evaluation.");
		}
		double fitness = 0.0;

		using namespace dnf_composer::element;

		const auto neuralField = std::dynamic_pointer_cast<NeuralField>(phenotype.getElement(fieldName));
		const int numberOfBumps = static_cast<int>(neuralField->getBumps().size());
		fitness += weightBumps / (1.0 + std::abs(targetNumberOfBumps - numberOfBumps));

		NeuralFieldBump closestBump1;
		for (const auto& bump : neuralField->getBumps())
		{
			if (std::abs(bump.centroid - position1) < std::abs(closestBump1.centroid - position1))
			{
				closestBump1 = bump;
			}
		}
		fitness += weightPos / (1.0 + std::abs(closestBump1.centroid - position1));
		fitness += weightAmp / (1.0 + std::abs(closestBump1.amplitude - amplitude1));
		fitness += weightWidth / (1.0 + std::abs(closestBump1.width - width1));

		NeuralFieldBump closestBump2;
		for (const auto& bump : neuralField->getBumps())
		{
			if (std::abs(bump.centroid - position2) < std::abs(closestBump2.centroid - position2))
			{
				closestBump2 = bump;
			}
		}
		fitness += weightPos / (1.0 + std::abs(closestBump2.centroid - position2));
		fitness += weightAmp / (1.0 + std::abs(closestBump2.amplitude - amplitude2));
		fitness += weightWidth / (1.0 + std::abs(closestBump2.width - width2));

		NeuralFieldBump closestBump3;
		for (const auto& bump : neuralField->getBumps())
		{
			if (std::abs(bump.centroid - position3) < std::abs(closestBump3.centroid - position3))
			{
				closestBump3 = bump;
			}
		}
		fitness += weightPos / (1.0 + std::abs(closestBump3.centroid - position3));
		fitness += weightAmp / (1.0 + std::abs(closestBump3.amplitude - amplitude3));
		fitness += weightWidth / (1.0 + std::abs(closestBump3.width - width3));

		return fitness;
	}

	// double Solution::preShapedness(const std::string& fieldName) const
	// {
	// 	using namespace dnf_composer::element;
	// 	const auto neuralField = std::dynamic_pointer_cast<NeuralField>(phenotype.getElement(fieldName));
	//
	// 	const double highestActivationValue = neuralField->getHighestActivation();
	// 	const double restingLevel = neuralField->getParameters().startingRestingLevel;
	//
	// 	// target activation is between the resting level and 0.0 (supra-threshold)
	// 	const double targetActivation = restingLevel / 2.0;
	// 	const double width = std::abs(restingLevel / 1.05);
	//
	// 	return tools::utils::normalizeWithGaussian(highestActivationValue, targetActivation, width);
	// }
	//
	// double Solution::preShapedness(const std::string& fieldName, const std::vector<double>& positions)
	// {
	// 	using namespace dnf_composer::element;
	// 	const auto neuralField = std::dynamic_pointer_cast<NeuralField>(phenotype.getElement(fieldName));
	// 	const double restingLevel = neuralField->getParameters().startingRestingLevel;
	// 	// target activation is between the resting level and 0.0 (sub-threshold)
	// 	const double targetActivation = restingLevel / 2.0;
	// 	const double width = std::abs(restingLevel / 6.0);  // Makes both points 3 standard deviations away
	//
	// 	// If no positions specified, return 0.0
	// 	if (positions.empty()) {
	// 		return 0.0;
	// 	}
	//
	// 	// Calculate the score for each position
	// 	double totalScore = 0.0;
	// 	for (const auto& position : positions) {
	// 		const double activationAtPosition = neuralField->getComponent("activation")[position];
	// 		double positionScore = tools::utils::normalizeWithGaussian(activationAtPosition, targetActivation, width);
	// 		totalScore += positionScore;
	// 	}
	//
	// 	// Return average score (will be 1.0 if all positions have perfect sub-threshold peaks)
	// 	return totalScore / positions.size();
	// }

	double Solution::preShapednessAtPosition(const std::string& fieldName, double position) const
	{
		using namespace dnf_composer::element;
		const auto nf = std::dynamic_pointer_cast<NeuralField>(phenotype.getElement(fieldName));

		const int idx = static_cast<int>(position / nf->getElementCommonParameters().dimensionParameters.d_x);
		const double u = nf->getComponent("activation")[idx];
		const double h = nf->getParameters().startingRestingLevel;
		const double u_tar =  h / 2.0;
		constexpr double sigma = 10.0;

		// 1) enforce subthreshold
		if (u >= 0.0)
		{
			return 0.0;
		}

		// 2) enforce higher than resting level
		static constexpr double epsilon = 0.01;
		if (u <= h + epsilon)
		{
			return 0.0;
		}

		// 3) reward closeness to target subthreshold height
		const double score_height = tools::utils::normalizeWithGaussian(u, u_tar, sigma);

		return score_height;
	}

	double Solution::negativePreShapednessAtPosition(const std::string& fieldName, const double& position) const
	{
		using namespace dnf_composer::element;
		const auto neuralField = std::dynamic_pointer_cast<NeuralField>(phenotype.getElement(fieldName));

		const int pos = static_cast<int>(position/neuralField->getElementCommonParameters().dimensionParameters.d_x);
		const double u_pos = neuralField->getComponent("activation")[pos];

		// activation of field at position should be lower than the resting level
		// we need to be careful here because if the field is in the resting level, this still produces above 0.5 fitness
		static constexpr double epsilon = 0.15;
		if (u_pos >= neuralField->getParameters().startingRestingLevel - epsilon)
		{
			return 0.0;
		}

		// static constexpr double epsilon = 0.015;
		// // activation of field at position should be lower than the rest of the neighboring positions
		// // I thought this was necessary because of mhk shapes, but apparently it can self-correct
		// for(const auto& u: neuralField->getComponent("activation"))
		// {
		// 	if (u_pos >= u+ epsilon)
		// 		return 0.0;
		// }

		const double u_baseline = neuralField->getHighestActivation();
		// this should not be like this - I am hardcoding the position of the baseline activation
		//const double u_baseline = std::abs(neuralField->getComponent("activation")[0]);
		const double u_target = u_baseline + u_baseline / 2.0;
		constexpr double width = 10.0;// std::abs(u_baseline / 8.0);

		return tools::utils::normalizeWithGaussian(u_pos, u_target, width);

		// using namespace dnf_composer::element;
		// const auto neuralField = std::dynamic_pointer_cast<NeuralField>(phenotype.getElement(fieldName));
		//
		// const int pos = static_cast<int>(position / neuralField->getElementCommonParameters().dimensionParameters.d_x);
		// const double u_tar_pos = neuralField->getComponent("activation")[pos];
		//
		// // activation of field at position should be lower than the resting level
		// if (u_tar_pos >= neuralField->getParameters().startingRestingLevel)
		// 	return 0.0;
		//
		// static constexpr double epsilon = 0.015;
		// // activation of field at position should be lower than the rest of the neighboring positions
		// for (const auto& u_pos : neuralField->getComponent("activation"))
		// {
		// 	if (u_tar_pos >= u_pos + epsilon)
		// 		return 0.0;
		// }
		//
		// // this should not be like this - I am hardcoding the position of the baseline activation
		// const double u_baseline = std::abs(neuralField->getComponent("activation")[0]);
		// const double u_target = u_baseline + u_baseline / 2;
		// const double width = u_baseline / 2;
		//
		// return tools::utils::normalizeWithGaussian(std::abs(u_tar_pos), u_target, width);
	}



	void Solution::moveGaussianStimulusContinuously(const std::string& name, const double targetPosition, const double step)
	{
		constexpr double epsilon = 1e-6;
		double newPosition = 0.0;
		const auto gaussStimulus = std::dynamic_pointer_cast<dnf_composer::element::GaussStimulus>(phenotype.getElement(name));
		const double diff_x = std::abs(targetPosition - gaussStimulus->getParameters().position);
		const double steps_x = diff_x / step;
		const int steps_t = static_cast<int>(SimulationConstants::maxSimulationSteps / steps_x);

		do
		{
			const auto position = gaussStimulus->getParameters().position;
			newPosition = position + step;
			gaussStimulus->setParameters(dnf_composer::element::GaussStimulusParameters{ gaussStimulus->getParameters().width, gaussStimulus->getParameters().amplitude, newPosition });

			for (int i = 0; i < steps_t; i++)
			{
				phenotype.step();
			}
		} while (std::abs(newPosition - targetPosition) > epsilon);
	}

	double Solution::negativeBaseline(const std::string& fieldName) const
	{
		using namespace dnf_composer::element;
		const auto neuralField = std::dynamic_pointer_cast<NeuralField>(phenotype.getElement(fieldName));
		const double startingRestingLevel = neuralField->getParameters().startingRestingLevel;
		const double maxActivation = neuralField->getHighestActivation();

		const double targetBaseline = startingRestingLevel * 2;
		const double width = std::abs(maxActivation / 8);

		return tools::utils::normalizeWithGaussian(maxActivation, targetBaseline, width);
	}

}