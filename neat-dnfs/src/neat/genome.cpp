#include "neat/genome.h"

namespace neat_dnfs
{
	std::map<ConnectionTuple, int> Genome::connectionTupleAndInnovationNumberWithinGeneration;

	Genome::~Genome()
	{
		fieldGenes.clear();
		connectionGenes.clear();
	}

	void Genome::addInputGene(const dnf_composer::element::ElementDimensions& dimensions)
	{
		const auto index = fieldGenes.size() + 1;
		fieldGenes.push_back(FieldGene({ FieldGeneType::INPUT, static_cast<int>(index)}, dimensions ));
	}

	void Genome::addOutputGene(const dnf_composer::element::ElementDimensions& dimensions)
	{
		const auto index = fieldGenes.size() + 1;
		fieldGenes.push_back(FieldGene({ FieldGeneType::OUTPUT,
			static_cast<int>(index)}, dimensions ));
	}

	void Genome::addHiddenGene(const FieldGene& gene)
	{
		const auto index = fieldGenes.size() + 1;
		fieldGenes.push_back(FieldGene({ FieldGeneType::HIDDEN,
					static_cast<int>(index) }, gene));
	}

	void Genome::mutate()
	{
		constexpr double totalProbability = GenomeMutationConstants::addFieldGeneProbability +
			GenomeMutationConstants::mutateFieldGeneProbability +
			GenomeMutationConstants::addConnectionGeneProbability +
			GenomeMutationConstants::mutateConnectionGeneProbability +
			GenomeMutationConstants::toggleConnectionGeneProbability;

		constexpr double epsilon = 1e-6;
		if ( std::abs(totalProbability - 1.0) > epsilon )
			throw std::runtime_error("Mutation probabilities in genome mutation must sum up to 1.");

		const double randomValue = tools::utils::generateRandomDouble(0.0, 1.0);

		if (randomValue < GenomeMutationConstants::addFieldGeneProbability) {
			addGene();
		}
		else if (randomValue < GenomeMutationConstants::addFieldGeneProbability +
			GenomeMutationConstants::mutateFieldGeneProbability) {
			mutateGene();
		}
		else if (randomValue < GenomeMutationConstants::addFieldGeneProbability +
			GenomeMutationConstants::mutateFieldGeneProbability +
			GenomeMutationConstants::addConnectionGeneProbability) {
			addConnectionGene();
		}
		else if (randomValue < GenomeMutationConstants::addFieldGeneProbability +
			GenomeMutationConstants::mutateFieldGeneProbability +
			GenomeMutationConstants::addConnectionGeneProbability +
			GenomeMutationConstants::mutateConnectionGeneProbability) {
			mutateConnectionGene();
		}
		else {
			toggleConnectionGene();
		}

		checkForDuplicateConnectionGenes();
	}

	void Genome::checkForDuplicateConnectionGenes() const
	{
		if (GenomeMutationConstants::checkForDuplicateConnectionGenesInGenome)
		{
			for (const auto& gene : connectionGenes)
			{
				const auto inFieldGeneId = gene.getInFieldGeneId();
				const auto outFieldGeneId = gene.getOutFieldGeneId();
				for (const auto& otherGene : connectionGenes)
				{
					if (gene.getInnovationNumber() != otherGene.getInnovationNumber() &&
						inFieldGeneId == otherGene.getInFieldGeneId() &&
						outFieldGeneId == otherGene.getOutFieldGeneId())
					{
						tools::logger::log(tools::logger::LogLevel::ERROR, "Mutation produced offspring with duplicate connection genes.");
						break;
					}
				}
			}
		}
	}

	void Genome::clearGenerationalInnovations()
	{
		connectionTupleAndInnovationNumberWithinGeneration.clear();
	}

	void Genome::resetGlobalInnovationNumber()
	{
		globalInnovationNumber = 0;
	}

	void Genome::removeConnectionGene(int innov)
	{
		const auto it = std::ranges::find_if(connectionGenes, [innov](const ConnectionGene& connectionGene)
		{
				return connectionGene.getInnovationNumber() == innov;
		});

		if (it == connectionGenes.end())
			throw std::invalid_argument("Connection gene with the specified innovation number " +
				std::to_string(innov) + " does not exist.");

		connectionGenes.erase(it);
	}

	std::vector<FieldGene> Genome::getFieldGenes() const
	{
		return fieldGenes;
	}

	std::vector<ConnectionGene> Genome::getConnectionGenes() const
	{
		return connectionGenes;
	}

	std::vector<int> Genome::getInnovationNumbers() const
	{
		std::vector<int> innovationNumbers;
		for (const auto& connectionGene : connectionGenes)
			innovationNumbers.push_back(connectionGene.getInnovationNumber());

		return innovationNumbers;
	}

	int Genome::getGlobalInnovationNumber()
	{
		return globalInnovationNumber;
	}

	int Genome::excessGenes(const Genome& other) const
	{
		const auto thisInnovationNumbers = getInnovationNumbers();
		const auto otherInnovationNumbers = other.getInnovationNumbers();
		int thisMaxInnovationNumber = 0;
		int otherMaxInnovationNumber = 0;

		if (thisInnovationNumbers.empty() && otherInnovationNumbers.empty())
			return 0;

		if (!thisInnovationNumbers.empty())
			thisMaxInnovationNumber = *std::ranges::max_element(thisInnovationNumbers);

		if (!otherInnovationNumbers.empty())
			otherMaxInnovationNumber = *std::ranges::max_element(otherInnovationNumbers);

		const int thisExcessCount = static_cast<int>(std::ranges::count_if(thisInnovationNumbers,
			[otherMaxInnovationNumber](const int innovationNumber)
			{
				return innovationNumber > otherMaxInnovationNumber;
			}));

		const int otherExcessCount = static_cast<int>(std::ranges::count_if(otherInnovationNumbers,
			[thisMaxInnovationNumber](const int innovationNumber)
			{
				return innovationNumber > thisMaxInnovationNumber;
			}));

		return thisExcessCount + otherExcessCount;
	}

	int Genome::disjointGenes(const Genome& other) const
	{
		const auto thisInnovationNumbers = getInnovationNumbers();
		const auto otherInnovationNumbers = other.getInnovationNumbers();

		if (thisInnovationNumbers.empty() && otherInnovationNumbers.empty())
			return 0;

		std::vector<int> sortedThisInnovationNumbers = thisInnovationNumbers;
		std::vector<int> sortedOtherInnovationNumbers = otherInnovationNumbers;

		std::ranges::sort(sortedThisInnovationNumbers);
		std::ranges::sort(sortedOtherInnovationNumbers);

		std::vector<int> thisDisjointInnovationNumbers, otherDisjointInnovationNumbers;

		std::ranges::set_difference(sortedThisInnovationNumbers, sortedOtherInnovationNumbers,
			std::back_inserter(thisDisjointInnovationNumbers));
		std::ranges::set_difference(sortedOtherInnovationNumbers, sortedThisInnovationNumbers,
			std::back_inserter(otherDisjointInnovationNumbers));

		const int thisMaxInnovationNumber = !thisInnovationNumbers.empty() ? *std::ranges::max_element(thisInnovationNumbers) : 0;
		const int otherMaxInnovationNumber = !otherInnovationNumbers.empty() ? *std::ranges::max_element(otherInnovationNumbers) : 0;
		int minMaxInnovationNumber = std::min(thisMaxInnovationNumber, otherMaxInnovationNumber);

		const int thisDisjointCount = static_cast<int>(std::ranges::count_if(thisDisjointInnovationNumbers,
			[minMaxInnovationNumber](const int innovationNumber)
			{
				return innovationNumber <= minMaxInnovationNumber;
			}));

		const int otherDisjointCount = static_cast<int>(std::ranges::count_if(otherDisjointInnovationNumbers,
			[minMaxInnovationNumber](const int innovationNumber)
			{
				return innovationNumber <= minMaxInnovationNumber;
			}));

		return thisDisjointCount + otherDisjointCount;
	}

	double Genome::averageConnectionDifference(const Genome& other) const
	{
		const auto thisConnectionGenes = getConnectionGenes();
		const auto otherConnectionGenes = other.getConnectionGenes();

		if (thisConnectionGenes.empty() && otherConnectionGenes.empty())
			return 0.0;

		double sumStrengthDiff = 0.0;
		static constexpr double strengthDiffCoeff = 1.0;

		for (const auto& thisConnectionGene : thisConnectionGenes)
		{
			for (const auto& otherConnectionGene : otherConnectionGenes)
			{
				if (thisConnectionGene.getInnovationNumber() == otherConnectionGene.getInnovationNumber())
				{
					const double strengthDiff =
						std::abs(thisConnectionGene.getCouplingStrength() - otherConnectionGene.getCouplingStrength());
					sumStrengthDiff += strengthDiff;
				}
			}
		}
		const double totalDiff = sumStrengthDiff * strengthDiffCoeff;
		return totalDiff;
	}

	void Genome::addFieldGene(const FieldGene& fieldGene)
	{
		if (containsFieldGene(fieldGene))
			return;
		fieldGenes.push_back(fieldGene);
	}

	void Genome::addConnectionGene(const ConnectionGene& connectionGene)
	{
		if (containsConnectionGene(connectionGene))
			return;
		connectionGenes.push_back(connectionGene);
	}

	bool Genome::containsConnectionGene(const ConnectionGene& connectionGene) const
	{
		return std::ranges::find(connectionGenes, connectionGene) != connectionGenes.end();
	}

	bool Genome::containsFieldGene(const FieldGene& fieldGene) const
	{
		return std::ranges::find(fieldGenes, fieldGene) != fieldGenes.end();
	}

	bool Genome::containsConnectionGeneWithTheSameInputOutputPair(const ConnectionGene& gene) const
	{
		return std::ranges::any_of(connectionGenes, [&gene](const auto& connectionGene)
			{
				return gene.getInFieldGeneId() == connectionGene.getInFieldGeneId() &&
					gene.getOutFieldGeneId() == connectionGene.getOutFieldGeneId();
			});
	}

	ConnectionGene Genome::getConnectionGeneByInnovationNumber(int innovationNumber) const
	{
		const auto it = std::ranges::find_if(connectionGenes,
			[innovationNumber](const ConnectionGene& connectionGene)
			{
				return connectionGene.getInnovationNumber() == innovationNumber;
			});

		if (it == connectionGenes.end())
			throw std::invalid_argument("Connection gene with the specified innovation number " +
				std::to_string(innovationNumber) + " does not exist.");

		return *it;
	}

	FieldGene Genome::getFieldGeneById(int id) const
	{
		const auto it = std::ranges::find_if(fieldGenes,
			[id](const FieldGene& fieldGene)
			{
				return fieldGene.getParameters().id == id;
			});

		if (it == fieldGenes.end())
			throw std::invalid_argument("Field gene with the specified id " +
				std::to_string(id) + " does not exist.");

		return *it;
	}

	bool Genome::isEmpty() const
	{
		return fieldGenes.empty() && connectionGenes.empty();
	}

	bool Genome::operator==(const Genome& other) const
	{
		return fieldGenes == other.fieldGenes && connectionGenes == other.connectionGenes;
	}

	std::string Genome::toString() const
	{
		std::string genomeString = "genome ( " + std::to_string(fieldGenes.size()) + " field genes, "
		+ std::to_string(connectionGenes.size()) + " connection genes )";
		genomeString += " field genes {";
		for (const auto& fieldGene : fieldGenes)
			genomeString += fieldGene.toString() + ", ";
		genomeString += "}";

		genomeString += " connection genes {";
		for (const auto& connectionGene : connectionGenes)
			genomeString += connectionGene.toString() + ", ";
		genomeString += "}";

		return genomeString;
	}

	void Genome::print() const
	{
		tools::logger::log(tools::logger::INFO, toString());
	}

	ConnectionTuple Genome::getNewRandomConnectionGeneTuple() const
	{
		// if there aren't enough genes to create a connection gene
		if (fieldGenes.size() < 2)
			return { 0, 0 };

		int geneIndex1 = getRandomGeneIdByTypes({ FieldGeneType::HIDDEN, FieldGeneType::INPUT });
		if (geneIndex1 == -1)
			return { 0, 0 };

		int geneIndex2;
		do {
			geneIndex2 = getRandomGeneIdByTypes({ FieldGeneType::HIDDEN, FieldGeneType::OUTPUT });
			if (geneIndex2 == -1)
				return { 0, 0 };
		} while (geneIndex2 == geneIndex1);

		if (std::ranges::find_if(connectionGenes, [geneIndex1, geneIndex2](const ConnectionGene& connectionGene)
			{
				return connectionGene.getParameters().connectionTuple.inFieldGeneId == geneIndex1
					&& connectionGene.getParameters().connectionTuple.outFieldGeneId == geneIndex2;
			}) != connectionGenes.end())
		{
			return { 0, 0 };
		}

		return { static_cast<int>(geneIndex1), static_cast<int>(geneIndex2) };
	}

	int Genome::getRandomGeneId() const
	{
		if (fieldGenes.empty())
			return -1;

		const int randomValue = tools::utils::generateRandomInt(0, static_cast<int>(fieldGenes.size()) - 1);

		return fieldGenes[randomValue].getParameters().id;
	}

	int Genome::getRandomGeneIdByType(FieldGeneType type) const
	{
		std::vector<int> geneIds;
		for (const auto& gene : fieldGenes)
		{
			if (gene.getParameters().type == type)
				geneIds.push_back(gene.getParameters().id);
		}

		if (geneIds.empty())
			return -1;

		const int randomValue = tools::utils::generateRandomInt(0, static_cast<int>(geneIds.size()) - 1);

		return geneIds[randomValue];
	}

	int Genome::getRandomGeneIdByTypes(const std::vector<FieldGeneType>& types) const
	{
		std::vector<int> geneIds;
		for (const auto& gene : fieldGenes)
		{
			if (std::ranges::find(types, gene.getParameters().type) != types.end())
				geneIds.push_back(gene.getParameters().id);
		}

		if (geneIds.empty())
			return -1;

		const int randomValue = tools::utils::generateRandomInt(0, static_cast<int>(geneIds.size()) - 1);

		return geneIds[randomValue];
	}

	ConnectionGene* Genome::getEnabledConnectionGene() const
	{
		std::vector<ConnectionGene*> enabledConnectionGenes;
		for (const auto& connectionGene : connectionGenes)
		{
			if (connectionGene.isEnabled())
				enabledConnectionGenes.push_back(const_cast<ConnectionGene*>(&connectionGene));
		}

		if (enabledConnectionGenes.empty())
			return nullptr; // Return nullptr instead of a temporary object

		// Generate a random valid index
		const int randomIndex = tools::utils::generateRandomInt(0, static_cast<int>(enabledConnectionGenes.size()) - 1);
		return enabledConnectionGenes[randomIndex];
	}

	void Genome::addConnectionGene(ConnectionTuple connectionTuple)
	{
		const FieldGene gene1 = getFieldGeneById(connectionTuple.inFieldGeneId);
		const dnf_composer::element::ElementDimensions dimensions1 =
			gene1.getNeuralField()->getElementCommonParameters().dimensionParameters;
		const FieldGene gene2 = getFieldGeneById(connectionTuple.outFieldGeneId);
		const dnf_composer::element::ElementDimensions dimensions2 =
			gene2.getNeuralField()->getElementCommonParameters().dimensionParameters;

		const int innov = getInnovationNumberOfTupleWithinGeneration(connectionTuple);
		if (innov > -1)
			// exists in the current generation
			// use the same innovation number
		{
			connectionGenes.emplace_back(connectionTuple, innov, dimensions1, dimensions2);
		}
		else
			// does not exist in the current generation
			// create new innovation number
		{
			connectionGenes.emplace_back(connectionTuple, globalInnovationNumber, dimensions1, dimensions2);
			connectionTupleAndInnovationNumberWithinGeneration[connectionTuple] = globalInnovationNumber;
			globalInnovationNumber++;
		}
	}

	void Genome::addGene()
	{
		tools::logger::log(tools::logger::FATAL, "addGene() function should not be called.");

		// using namespace dnf_composer::element;
		//
		// ConnectionGene* randEnabledConnectionGene = getEnabledConnectionGene();
		// if (randEnabledConnectionGene == nullptr)
		// 	return;
		// randEnabledConnectionGene->disable();
		//
		// const auto inGeneId = randEnabledConnectionGene->getParameters().connectionTuple.inFieldGeneId;
		// const auto outGeneId = randEnabledConnectionGene->getParameters().connectionTuple.outFieldGeneId;
		// const auto coupling = randEnabledConnectionGene->getFieldCoupling();
		// const auto couplingParameters = std::dynamic_pointer_cast<FieldCoupling>(coupling)->getParameters();
		// const auto couplingInputDimensions = couplingParameters.inputFieldDimensions;
		// const auto couplingOutputDimensions = coupling->getElementCommonParameters().dimensionParameters;
		//
		// //addHiddenGene({ DimensionConstants::xSize, DimensionConstants::dx });
		// addHiddenGene(getFieldGeneById(inGeneId));
		// //addHiddenGene(couplingInputDimensions);
		//
		// // create two new connection genes
		// // when creating the two new connection genes we have to obey the same rules in add connection gene mutation
		//
		// const ConnectionTuple connectionTupleIn{ inGeneId, fieldGenes.back().getParameters().id };
		// const ConnectionTuple connectionTupleOut{ fieldGenes.back().getParameters().id, outGeneId };
		// int innovIn = getInnovationNumberOfTupleWithinGeneration(connectionTupleIn);
		// int innovOut = getInnovationNumberOfTupleWithinGeneration(connectionTupleOut);
		//
		// if (innovIn == -1) // if this mutation has not been performed in the current generation
		// {
		// 	connectionTupleAndInnovationNumberWithinGeneration[connectionTupleIn] = globalInnovationNumber;
		// 	innovIn = globalInnovationNumber;
		// 	globalInnovationNumber++;
		// }
		//
		// if (innovOut == -1) // if this mutation has not been performed in the current generation
		// {
		// 	connectionTupleAndInnovationNumberWithinGeneration[connectionTupleOut] = globalInnovationNumber;
		// 	innovOut = globalInnovationNumber;
		// 	globalInnovationNumber++;
		// }
		//
		// const auto in_kernel_p = GaussKernelParameters{ GaussKernelConstants::width, GaussKernelConstants::amplitude, GaussKernelConstants::amplitudeGlobal };
		// const ConnectionGene connectionGeneIn{ ConnectionTuple{inGeneId, fieldGenes.back().getParameters().id}, static_cast<int>(innovIn), in_kernel_p };
		//
		//
		// const auto gkp = std::dynamic_pointer_cast<GaussKernel>(kernel)->getParameters();
		// //const ConnectionGene connectionGeneIn{ ConnectionTuple{inGeneId, fieldGenes.back().getParameters().id}, static_cast<int>(innovIn), gkp };
		// const ConnectionGene connectionGeneOut{ ConnectionTuple{fieldGenes.back().getParameters().id, outGeneId}, static_cast<int>(innovOut), gkp };
		// connectionGenes.emplace_back(connectionGeneIn);
		// connectionGenes.emplace_back(connectionGeneOut);

	}

	void Genome::mutateGene() const
	{
		const int geneId = getRandomGeneId();
		if (geneId == -1)
		{
			return;
		}
		auto gene = getFieldGeneById(static_cast<int>(geneId));
		gene.mutate();
	}

	void Genome::addConnectionGene()
	{
		const ConnectionTuple connectionTuple = getNewRandomConnectionGeneTuple();
		if (connectionTuple == ConnectionTuple{ 0, 0 })
		{
			return;
		}
		addConnectionGene(connectionTuple);
	}

	void Genome::mutateConnectionGene() const
	{
		if (connectionGenes.empty())
		{
			return;
		}

		const auto connectionGeneId = tools::utils::generateRandomInt(0, static_cast<int>(connectionGenes.size()) - 1);
		connectionGenes[connectionGeneId].mutate();
	}

	void Genome::toggleConnectionGene()
	{
		if (connectionGenes.empty())
		{
			return;
		}
		const auto connectionGeneId = tools::utils::generateRandomInt(0, static_cast<int>(connectionGenes.size()) - 1);
		connectionGenes[connectionGeneId].toggle();
	}

	int Genome::getInnovationNumberOfTupleWithinGeneration(const ConnectionTuple& tuple)
	{
		if (connectionTupleAndInnovationNumberWithinGeneration.contains(tuple))
		{
			return connectionTupleAndInnovationNumberWithinGeneration[tuple];
		}
		return -1;
	}
}