#include "neat/genome.h"

namespace neat_dnfs
{
	int Genome::globalInnovationNumber = 0;
	std::map<ConnectionTuple, int> Genome::connectionTupleAndInnovationNumberWithinGeneration;
	std::mutex Genome::innovationMutex;

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
		if (tools::utils::generateRandomDouble(0.0, 1.0) <
			GenomeMutationConstants::toggleConnectionGeneProbability)
		{
			toggleConnectionGene();
		}

		if (tools::utils::generateRandomDouble(0.0, 1.0) <
			GenomeMutationConstants::addFieldGeneProbability)
		{
			addGene();
		}

		if (tools::utils::generateRandomDouble(0.0, 1.0) <
			GenomeMutationConstants::mutateFieldGenesProbability)
		{
			mutateGene();
		}

		if (tools::utils::generateRandomDouble(0.0, 1.0) <
			GenomeMutationConstants::addConnectionGeneProbability)
		{
			addConnectionGene();
		}

		if (tools::utils::generateRandomDouble(0.0, 1.0) <
			GenomeMutationConstants::mutateConnectionGenesProbability)
		{
			mutateConnectionGene();
		}

		mutationsInLastGeneration += " (mutated)";
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
						tools::logger::log(tools::logger::LogLevel::ERROR,
							"Mutation produced offspring with duplicate connection genes.");
						break;
					}
				}
			}
		}
	}

	void Genome::clearGenerationalInnovations()
	{
		std::scoped_lock lock(innovationMutex);
		connectionTupleAndInnovationNumberWithinGeneration.clear();
	}

	void Genome::resetGlobalInnovationNumber()
	{
		std::scoped_lock lock(innovationMutex);
		globalInnovationNumber = 0;
	}

	void Genome::clearLastMutations()
	{
		mutationsInLastGeneration = "";
		for (auto& gene : fieldGenes)
		{
			gene.clearLastMutations();
		}
		for (auto& gene : connectionGenes)
		{
			gene.clearLastMutations();
		}
	}

	void Genome::removeConnectionGene(int innov)
	{
		const auto it = std::ranges::find_if(connectionGenes,
			[innov](const ConnectionGene& connectionGene)
		{
				return connectionGene.getInnovationNumber() == innov;
		});

		if (it == connectionGenes.end())
		{
			throw std::invalid_argument("Connection gene with the specified innovation number " +
				std::to_string(innov) + " does not exist.");
		}

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
		innovationNumbers.reserve(connectionGenes.size());
		for (const auto& connectionGene : connectionGenes)
		{
			innovationNumbers.push_back(connectionGene.getInnovationNumber());
		}

		return innovationNumbers;
	}

	int Genome::getGlobalInnovationNumber()
	{
		std::scoped_lock lock(innovationMutex);
		return globalInnovationNumber;
	}

	std::string Genome::getMutationsInLastGeneration() const
	{
		return mutationsInLastGeneration;
	}

	int Genome::excessGenes(const Genome& other) const
	{
		const auto thisInnovationNumbers = getInnovationNumbers();
		const auto otherInnovationNumbers = other.getInnovationNumbers();
		int thisMaxInnovationNumber = 0;
		int otherMaxInnovationNumber = 0;

		if (thisInnovationNumbers.empty() && otherInnovationNumbers.empty())
		{
			return 0;
		}

		if (!thisInnovationNumbers.empty())
		{
			thisMaxInnovationNumber = *std::ranges::max_element(thisInnovationNumbers);
		}

		if (!otherInnovationNumbers.empty())
		{
			otherMaxInnovationNumber = *std::ranges::max_element(otherInnovationNumbers);
		}

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
		{
			return 0;
		}

		std::vector<int> sortedThisInnovationNumbers = thisInnovationNumbers;
		std::vector<int> sortedOtherInnovationNumbers = otherInnovationNumbers;

		std::ranges::sort(sortedThisInnovationNumbers);
		std::ranges::sort(sortedOtherInnovationNumbers);

		std::vector<int> thisDisjointInnovationNumbers;
		std::vector<int> otherDisjointInnovationNumbers;

		std::ranges::set_difference(sortedThisInnovationNumbers,
			sortedOtherInnovationNumbers, std::back_inserter(thisDisjointInnovationNumbers));
		std::ranges::set_difference(sortedOtherInnovationNumbers,
			sortedThisInnovationNumbers, std::back_inserter(otherDisjointInnovationNumbers));

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
		{
			return 0.0;
		}

		double sumAmpDiff = 0.0;
		double sumWidthDiff = 0.0;

		for (const auto& thisConnectionGene : thisConnectionGenes)
		{
			for (const auto& otherConnectionGene : otherConnectionGenes)
			{
				if (thisConnectionGene.getInnovationNumber() == otherConnectionGene.getInnovationNumber())
				{
					const double ampDiff =
						std::abs(thisConnectionGene.getKernelAmplitude() - otherConnectionGene.getKernelAmplitude());
					const double widthDiff =
						std::abs(thisConnectionGene.getKernelWidth() - otherConnectionGene.getKernelWidth());
					sumAmpDiff += ampDiff;
					sumWidthDiff += widthDiff;
				}
			}
		}
		const double totalDiff = CompatibilityCoefficients::amplitudeDifferenceCoefficient * sumAmpDiff
								+ CompatibilityCoefficients::widthDifferenceCoefficient * sumWidthDiff;
		return totalDiff;
	}

	void Genome::addFieldGene(const FieldGene& fieldGene)
	{
		if (containsFieldGene(fieldGene))
		{
			return;
		}
		fieldGenes.push_back(fieldGene);
	}

	void Genome::addConnectionGene(const ConnectionGene& connectionGene)
	{
		if (containsConnectionGene(connectionGene))
		{
			return;
		}
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
		const auto it =
			std::ranges::find_if(connectionGenes, [innovationNumber](const ConnectionGene& connectionGene)
			{
				return connectionGene.getInnovationNumber() == innovationNumber;
			});

		if (it == connectionGenes.end())
		{
			throw std::invalid_argument("Connection gene with the specified innovation number " +
				std::to_string(innovationNumber) + " does not exist.");
		}

		return *it;
	}

	FieldGene Genome::getFieldGeneById(int id) const
	{
		const auto it =
			std::ranges::find_if(fieldGenes, [id](const FieldGene& fieldGene)
			{
				return fieldGene.getParameters().id == id;
			});

		if (it == fieldGenes.end())
		{
			throw std::invalid_argument("Field gene with the specified id " +
				std::to_string(id) + " does not exist.");
		}

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
		{
			genomeString += fieldGene.toString() + ", ";
		}
		genomeString += "}";

		genomeString += " connection genes {";
		for (const auto& connectionGene : connectionGenes)
		{
			genomeString += connectionGene.toString() + ", ";
		}
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
		{
			return { 0, 0 };
		}

		int geneIndex1 = getRandomGeneIdByTypes({ FieldGeneType::HIDDEN, FieldGeneType::INPUT });
		if (geneIndex1 == -1)
		{
			return { 0, 0 };
		}

		// Bounded retry: if the only {HIDDEN, OUTPUT} candidate happens to be
		// geneIndex1 itself, the draw below would keep returning geneIndex1
		// forever. Cap the attempts and fall back to the sentinel instead of
		// spinning indefinitely (see issue #57).
		constexpr int maxGeneIndex2Attempts = 8;
		int geneIndex2 = -1;
		bool foundDistinctGeneIndex2 = false;
		for (int attempt = 0; attempt < maxGeneIndex2Attempts; ++attempt)
		{
			geneIndex2 = getRandomGeneIdByTypes({ FieldGeneType::HIDDEN, FieldGeneType::OUTPUT });
			if (geneIndex2 == -1)
			{
				return { 0, 0 };
			}
			if (geneIndex2 != geneIndex1)
			{
				foundDistinctGeneIndex2 = true;
				break;
			}
		}
		if (!foundDistinctGeneIndex2)
		{
			return { 0, 0 };
		}

		if (std::ranges::find_if(connectionGenes, [geneIndex1, geneIndex2](const ConnectionGene& connectionGene)
			{
				return connectionGene.getParameters().connectionTuple.inFieldGeneId == geneIndex1
					&& connectionGene.getParameters().connectionTuple.outFieldGeneId == geneIndex2;
			}) != connectionGenes.end())
		{
			return { 0, 0 };
		}

		return { geneIndex1, geneIndex2 };
	}

	int Genome::getRandomGeneId() const
	{
		if (fieldGenes.empty())
		{
			return -1;
		}

		const int randomValue = tools::utils::generateRandomInt(0, static_cast<int>(fieldGenes.size()) - 1);

		return fieldGenes[randomValue].getParameters().id;
	}

	int Genome::getRandomGeneIdByType(FieldGeneType type) const
	{
		std::vector<int> geneIds;
		for (const auto& gene : fieldGenes)
		{
			if (gene.getParameters().type == type)
			{
				geneIds.push_back(gene.getParameters().id);
			}
		}

		if (geneIds.empty())
		{
			return -1;
		}

		const int randomValue = tools::utils::generateRandomInt(0, static_cast<int>(geneIds.size()) - 1);

		return geneIds[randomValue];
	}

	int Genome::getRandomGeneIdByTypes(const std::vector<FieldGeneType>& types) const
	{
		std::vector<int> geneIds;
		for (const auto& gene : fieldGenes)
		{
			if (std::ranges::find(types, gene.getParameters().type) != types.end())
			{
				geneIds.push_back(gene.getParameters().id);
			}
		}

		if (geneIds.empty())
		{
			return -1;
		}

		const int randomValue = tools::utils::generateRandomInt(0, static_cast<int>(geneIds.size()) - 1);

		return geneIds[randomValue];
	}

	ConnectionGene* Genome::getEnabledConnectionGene()
	{
		std::vector<ConnectionGene*> enabledConnectionGenes;
		for (auto& connectionGene : connectionGenes)
		{
			if (connectionGene.isEnabled())
			{
				enabledConnectionGenes.push_back(&connectionGene);
			}
		}

		if (enabledConnectionGenes.empty())
		{
			return nullptr; // Return nullptr instead of a temporary object
		}

		// Generate a random valid index
		const int randomIndex = tools::utils::generateRandomInt(0, static_cast<int>(enabledConnectionGenes.size()) - 1);
		return enabledConnectionGenes[randomIndex];
	}

	void Genome::addConnectionGene(ConnectionTuple connectionTuple)
	{
		std::scoped_lock lock(innovationMutex);
		const int innov = getInnovationNumberOfTupleWithinGenerationUnlocked(connectionTuple);
		if (innov > -1)
			// exists in the current generation
			// use the same innovation number
		{
			connectionGenes.emplace_back(connectionTuple, innov);
			mutationsInLastGeneration += "(added cg " + connectionTuple.toString()
			+ " innov." + std::to_string(innov) + ")";
		}
		else
			// does not exist in the current generation
			// create new innovation number
		{
			connectionGenes.emplace_back(connectionTuple, globalInnovationNumber);
			connectionTupleAndInnovationNumberWithinGeneration[connectionTuple] = globalInnovationNumber;
			mutationsInLastGeneration += "(added cg " + connectionTuple.toString()
			+ " innov." + std::to_string(globalInnovationNumber) + ")";
			globalInnovationNumber++;
		}
	}

	void Genome::addGene()
	{
		using namespace dnf_composer::element;

		ConnectionGene* randEnabledConnectionGene = getEnabledConnectionGene();
		if (randEnabledConnectionGene == nullptr)
		{
			return;
		}
		randEnabledConnectionGene->disable();

		const auto inGeneId = randEnabledConnectionGene->getParameters().connectionTuple.inFieldGeneId;
		const auto outGeneId = randEnabledConnectionGene->getParameters().connectionTuple.outFieldGeneId;
		const auto kernel = randEnabledConnectionGene->getKernel();

		//addHiddenGene(dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		addHiddenGene(getFieldGeneById(inGeneId));

		// create two new connection genes
		// when creating the two new connection genes we have to obey the same rules in add connection gene mutation

		const ConnectionTuple connectionTupleIn{ inGeneId, fieldGenes.back().getParameters().id };
		const ConnectionTuple connectionTupleOut{ fieldGenes.back().getParameters().id, outGeneId };
		int innovIn = -1;
		int innovOut = -1;
		{
			std::scoped_lock lock(innovationMutex);
			innovIn = getInnovationNumberOfTupleWithinGenerationUnlocked(connectionTupleIn);
			innovOut = getInnovationNumberOfTupleWithinGenerationUnlocked(connectionTupleOut);

			if (innovIn == -1) // if this mutation has not been performed in the current generation
			{
				connectionTupleAndInnovationNumberWithinGeneration[connectionTupleIn] = globalInnovationNumber;
				innovIn = globalInnovationNumber;
				globalInnovationNumber++;
			}

			if (innovOut == -1) // if this mutation has not been performed in the current generation
			{
				connectionTupleAndInnovationNumberWithinGeneration[connectionTupleOut] = globalInnovationNumber;
				innovOut = globalInnovationNumber;
				globalInnovationNumber++;
			}
		}

		const auto in_kernel_p = GaussKernelParameters{
			GaussKernelConstants::width, GaussKernelConstants::amplitude, GaussKernelConstants::amplitudeGlobal };
		const ConnectionGene connectionGeneIn{
			ConnectionTuple{inGeneId, fieldGenes.back().getParameters().id},
			innovIn,
			in_kernel_p };

		switch (kernel->getLabel())
		{
		case GAUSS_KERNEL:
			{
				const auto gkp = std::dynamic_pointer_cast<GaussKernel>(kernel)->getParameters();
				const ConnectionGene connectionGeneOut{
					ConnectionTuple{fieldGenes.back().getParameters().id, outGeneId},
					innovOut,
					gkp };
				connectionGenes.emplace_back(connectionGeneIn);
				connectionGenes.emplace_back(connectionGeneOut);
			}
			break;
		case MEXICAN_HAT_KERNEL:
			{
				const auto mhkp = std::dynamic_pointer_cast<MexicanHatKernel>(kernel)->getParameters();
				const ConnectionGene connectionGeneOut{
					ConnectionTuple{fieldGenes.back().getParameters().id, outGeneId},
						innovOut,
						mhkp };
				connectionGenes.emplace_back(connectionGeneIn);
				connectionGenes.emplace_back(connectionGeneOut);

			}
			break;
		default:
			throw std::invalid_argument("Invalid kernel type.");
		}
		mutationsInLastGeneration += "(added fg " + std::to_string(fieldGenes.back().getParameters().id) +
			" and cgs innov's " + std::to_string(innovIn) + ", " + std::to_string(innovOut) + ")";
	}

	void Genome::mutateGene()
	{
		for (auto& gene : fieldGenes)
		{
			if (tools::utils::generateRandomDouble(0.0, 1.0) < GenomeMutationConstants::mutateFieldGeneProbability)
			{
				gene.mutate();
				mutationsInLastGeneration +=  "[fg " + std::to_string(gene.getParameters().id) + " " +
					gene.getMutationsInLastGeneration() + "] ";
			}
		}
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

	void Genome::mutateConnectionGene()
	{
		if (connectionGenes.empty())
		{
			return;
		}

		for (auto& connectionGene : connectionGenes)
		{
			if (tools::utils::generateRandomDouble(0.0, 1.0) <
				GenomeMutationConstants::mutateConnectionGeneProbability)
			{
				connectionGene.mutate();
				const std::string cg_id = connectionGene.getParameters().connectionTuple.toString();
				mutationsInLastGeneration += "[cg " + cg_id + " " + connectionGene.getMutationsInLastGeneration() + "] ";
			}
		}
	}

	void Genome::toggleConnectionGene()
	{
		if (connectionGenes.empty())
		{
			return;
		}
		const auto connectionGeneId = tools::utils::generateRandomInt(0, static_cast<int>(connectionGenes.size()) - 1);
		connectionGenes[connectionGeneId].toggle();
		mutationsInLastGeneration += "toggle cg " + std::to_string(connectionGeneId) + " to " +
			(connectionGenes[connectionGeneId].isEnabled() ? "enabled." : "disabled.");
	}

	int Genome::getInnovationNumberOfTupleWithinGenerationUnlocked(const ConnectionTuple& tuple)
	{
		if (connectionTupleAndInnovationNumberWithinGeneration.contains(tuple))
		{
			return connectionTupleAndInnovationNumberWithinGeneration[tuple];
		}
		return -1;
	}
}