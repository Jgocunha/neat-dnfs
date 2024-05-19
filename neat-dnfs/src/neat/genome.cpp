#include "neat/genome.h"

namespace neat_dnfs
{
	std::map<ConnectionTuple, uint16_t> Genome::connectionToInnovationNumberMap;

	void Genome::addInputGene()
	{
		fieldGenes.emplace_back(FieldGeneType::INPUT);
	}

	void Genome::addOutputGene()
	{
		fieldGenes.emplace_back(FieldGeneType::OUTPUT);
	}

	void Genome::addHiddenGene()
	{
		fieldGenes.emplace_back(FieldGeneType::HIDDEN);
	}

	void Genome::addRandomInitialConnectionGene()
	{
		if (fieldGenes.size() < 2)
			throw std::invalid_argument(
			"There must be at least two genes to add a connection gene.");

		const auto inGeneId = getRandomGeneIdByType(FieldGeneType::INPUT);
		const auto outGeneId = getRandomGeneIdByType(FieldGeneType::OUTPUT);

		connectionGenes.push_back(ConnectionTuple{ inGeneId, outGeneId });
	}

	void Genome::mutate()
	{
		constexpr double totalProbability = MutationConstants::addGeneProbability + 
											MutationConstants::mutateGeneProbability + 
											MutationConstants::addConnectionGeneProbability +
											MutationConstants::mutateConnectionGeneProbability + 
											MutationConstants::toggleConnectionGeneProbability;

		if constexpr (totalProbability != 1.0)
			throw std::runtime_error("Mutation probabilities must sum up to 1.");

		const double randomValue = dnf_composer::tools::utils::generateRandomNumber(0.0, 1.0);

		if (randomValue < MutationConstants::addGeneProbability) {
			addGene();
		}
		else if (randomValue < MutationConstants::addGeneProbability + 
								MutationConstants::mutateGeneProbability) {
			mutateGene();
		}
		else if (randomValue < MutationConstants::addGeneProbability + 
								MutationConstants::mutateGeneProbability + 
								MutationConstants::addConnectionGeneProbability) {
			addConnectionGene();
		}
		else if (randomValue < MutationConstants::addGeneProbability + 
								MutationConstants::mutateGeneProbability + 
								MutationConstants::addConnectionGeneProbability + 
								MutationConstants::mutateConnectionGeneProbability) {
			mutateConnectionGene();
		}
		else {
			toggleConnectionGene();
		}
	}

	void Genome::clearGenerationalInnovations()
	{
		connectionToInnovationNumberMap.clear();
	}
	
	std::vector<FieldGene> Genome::getFieldGenes() const
	{
		return fieldGenes;
	}

	std::vector<ConnectionGene> Genome::getConnectionGenes() const
	{
		return connectionGenes;
	}

	std::vector<uint16_t> Genome::getInnovationNumbers() const
	{
		std::vector<uint16_t> innovationNumbers;
		for (const auto& connectionGene : connectionGenes)
			innovationNumbers.push_back(connectionGene.getInnovationNumber());

		return innovationNumbers;
	}

	ConnectionTuple Genome::getNewRandomConnectionGeneTuple() const
	{
		uint16_t geneIndex1 = getRandomGeneIdByTypes({ FieldGeneType::HIDDEN, FieldGeneType::INPUT });

		uint16_t geneIndex2;
		do {
			geneIndex2 = getRandomGeneIdByTypes({ FieldGeneType::HIDDEN, FieldGeneType::OUTPUT });
		} while (geneIndex2 == geneIndex1);

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

	uint16_t Genome::getRandomGeneIdByType(FieldGeneType type) const
	{
		std::vector<uint16_t> geneIds;
		for (const auto& gene : fieldGenes)
		{
			if (gene.getParameters().type == type)
				geneIds.push_back(gene.getParameters().id);
		}

		if (geneIds.empty())
			return 0;

		const double randomValue = dnf_composer::tools::utils::generateRandomNumber(0, static_cast<int>(geneIds.size()));

		return geneIds[static_cast<int>(randomValue)];
	}

	uint16_t Genome::getRandomGeneIdByTypes(const std::vector<FieldGeneType>& types) const
	{
		std::vector<uint16_t> geneIds;
		for (const auto& gene : fieldGenes)
		{
			if (std::ranges::find(types, gene.getParameters().type) != types.end())
				geneIds.push_back(gene.getParameters().id);
		}

		if (geneIds.empty())
			return 0;

		const double randomValue = dnf_composer::tools::utils::generateRandomNumber(0, static_cast<int>(geneIds.size()));

		return geneIds[static_cast<int>(randomValue)];
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
			return ConnectionGene{ {0, 0}};

		const double randomValue = dnf_composer::tools::utils::generateRandomNumber(0, static_cast<int>(enabledConnectionGenes.size()) + 1);

		return enabledConnectionGenes[static_cast<int>(randomValue)];
	}

	void Genome::addConnectionGeneIfNewWithinGeneration(ConnectionTuple connectionTuple)
	{
		if (connectionToInnovationNumberMap.contains(connectionTuple))
		{
			const uint16_t innovationNumber = connectionToInnovationNumberMap[connectionTuple];
			connectionGenes.emplace_back(connectionTuple);
			connectionGenes.back().setInnovationNumber(innovationNumber);
		}
		else
		{
			connectionGenes.emplace_back(connectionTuple);
			connectionToInnovationNumberMap.insert({ connectionTuple,
				connectionGenes.back().getInnovationNumber() });
		}
	}

	void Genome::addGene()
	{
		using namespace dnf_composer::element;

		auto randEnabledConnectionGene = getEnabledConnectionGene();
		randEnabledConnectionGene.disable();

		const auto inGeneId= randEnabledConnectionGene.getParameters().connectionTuple.inFieldGeneId;
		const auto outGeneId= randEnabledConnectionGene.getParameters().connectionTuple.outFieldGeneId;
		const auto kernel = randEnabledConnectionGene.getKernel();
		const auto kernelParameters = std::dynamic_pointer_cast<GaussKernel>(kernel)->getParameters();

		addHiddenGene();

		// create two new connection genes
		const auto connectionGeneKernelParametersIn = GaussKernelParameters{ GaussKernelConstants::sigma,
														GaussKernelConstants::amplitude,
														KernelConstants::circularity,
														KernelConstants::normalization};
		const auto connectionGeneKernelParametersOut = GaussKernelParameters{ kernelParameters.sigma,
													kernelParameters.amplitude,
														KernelConstants::circularity,
												KernelConstants::normalization };

		const ConnectionGene connectionGeneIn{ {inGeneId,
			fieldGenes.back().getParameters().id}, connectionGeneKernelParametersIn};
		const ConnectionGene connectionGeneOut{ {fieldGenes.back().getParameters().id,
			outGeneId}, connectionGeneKernelParametersOut };
		connectionGenes.push_back(connectionGeneIn);
		connectionGenes.push_back(connectionGeneOut);
	}

	void Genome::mutateGene() const
	{
		const auto geneId = getRandomGeneIdByType(FieldGeneType::HIDDEN);
		fieldGenes[geneId].mutate();
	}

	void Genome::addConnectionGene()
	{
		const ConnectionTuple connectionTuple = getNewRandomConnectionGeneTuple();
		if (connectionTuple == ConnectionTuple{0, 0})
			return;
		addConnectionGeneIfNewWithinGeneration(connectionTuple);
	}

	void Genome::mutateConnectionGene() const
	{
		if (connectionGenes.empty())
			return;

		const auto connectionGeneId = dnf_composer::tools::utils::generateRandomNumber(0, static_cast<int>(connectionGenes.size()) - 1);
		connectionGenes[connectionGeneId].mutate();
	}

	void Genome::toggleConnectionGene()
	{
		if (connectionGenes.empty())
			return;
		const auto connectionGeneId = dnf_composer::tools::utils::generateRandomNumber(0, static_cast<int>(connectionGenes.size()) - 1);
		connectionGenes[connectionGeneId].toggle();
	}

	int Genome::excessGenes(const Genome& other) const
	{
		const auto thisInnovationNumbers = getInnovationNumbers();
		const auto otherInnovationNumbers = other.getInnovationNumbers();
		int thisMaxInnovationNumber = 0;
		int otherMaxInnovationNumber = 0;

		if (thisInnovationNumbers.empty() && otherInnovationNumbers.empty())
			return 0;

		// max_element can be replaced with ranges::max_element
		if (!thisInnovationNumbers.empty())
			thisMaxInnovationNumber = *std::max_element(thisInnovationNumbers.begin(),
				thisInnovationNumbers.end());

		if(!otherInnovationNumbers.empty())
			otherMaxInnovationNumber = *std::max_element(otherInnovationNumbers.begin(),
									otherInnovationNumbers.end());

		// call to count_if can be replaced with ranges::count_if
		// also narrowing conversion from int to uint16_t
		const int thisExcessCount = std::count_if(thisInnovationNumbers.begin(),
		thisInnovationNumbers.end(), [otherMaxInnovationNumber](const uint16_t innovationNumber)
		{return innovationNumber > otherMaxInnovationNumber;});

		const int otherExcessCount = std::count_if(otherInnovationNumbers.begin(), otherInnovationNumbers.end(),
			[thisMaxInnovationNumber](const uint16_t innovationNumber)
			{return innovationNumber > thisMaxInnovationNumber;});

		return thisExcessCount + otherExcessCount;
	}

	int Genome::disjointGenes(const Genome& other) const
	{
		const auto thisInnovationNumbers = getInnovationNumbers();
		const auto otherInnovationNumbers = other.getInnovationNumbers();

		if (thisInnovationNumbers.empty() && otherInnovationNumbers.empty())
			return 0;

		std::vector<uint16_t> sortedThisInnovationNumbers = thisInnovationNumbers;
		std::vector<uint16_t> sortedOtherInnovationNumbers = otherInnovationNumbers;

		// call to sort can be replaced with ranges::sort
		std::sort(sortedThisInnovationNumbers.begin(), sortedThisInnovationNumbers.end());
		std::sort(sortedOtherInnovationNumbers.begin(), sortedOtherInnovationNumbers.end());

		std::vector<uint16_t> thisDisjointInnovationNumbers, otherDisjointInnovationNumbers;

		// Get the elements in sortedThisInnovationNumbers that are not in sortedOtherInnovationNumbers
		// call to set_difference can be replaced with ranges::set_difference
		std::set_difference(sortedThisInnovationNumbers.begin(), sortedThisInnovationNumbers.end(),
						sortedOtherInnovationNumbers.begin(), sortedOtherInnovationNumbers.end(),
						std::inserter(thisDisjointInnovationNumbers, thisDisjointInnovationNumbers.begin()));
		// Get the elements in sortedOtherInnovationNumbers that are not in sortedThisInnovationNumbers
		std::set_difference(sortedOtherInnovationNumbers.begin(), sortedOtherInnovationNumbers.end(),
									sortedThisInnovationNumbers.begin(), sortedThisInnovationNumbers.end(),
									std::inserter(otherDisjointInnovationNumbers, otherDisjointInnovationNumbers.begin()));

		// Calculate the maximum and minimum innovation numbers
		// call to max_element can be replaced with ranges::max_element
		const int thisMaxInnovationNumber = !thisInnovationNumbers.empty() ? *std::max_element(thisInnovationNumbers.begin(),
									thisInnovationNumbers.end()) : 0;
		const int otherMaxInnovationNumber = !otherInnovationNumbers.empty() ? *std::max_element(otherInnovationNumbers.begin(),
												otherInnovationNumbers.end()) : 0;
		int minMaxInnovationNumber = std::min(thisMaxInnovationNumber, otherMaxInnovationNumber);

		// Filter out the disjoint genes that are beyond the range of the other genome
		// call to count_if can be replaced with ranges::count_if
		// also narrowing conversion from int to uint16_t
		const int thisDisjointCount = std::count_if(thisDisjointInnovationNumbers.begin(),
						thisDisjointInnovationNumbers.end(), [minMaxInnovationNumber](const uint16_t innovationNumber)
					{return innovationNumber <= minMaxInnovationNumber; });
		const int otherDisjointCount = std::count_if(otherDisjointInnovationNumbers.begin(),
									otherDisjointInnovationNumbers.end(), [minMaxInnovationNumber](const uint16_t innovationNumber)
							{return innovationNumber <= minMaxInnovationNumber; });

		return thisDisjointCount + otherDisjointCount;
	}

	double Genome::averageConnectionDifference(const Genome& other) const
	{
		const auto thisConnectionGenes = getConnectionGenes();
		const auto otherConnectionGenes = other.getConnectionGenes();

		if (thisConnectionGenes.empty() && otherConnectionGenes.empty())
			return 0.0;

		double sumAmpDiff = 0.0;
		double sumWidthDiff = 0.0;
		static constexpr double ampDiffCoeff = 0.8;
		static constexpr double widthDiffCoeff = 0.2;

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
		const double totalDiff = ampDiffCoeff * sumAmpDiff + widthDiffCoeff * sumWidthDiff;
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
		if(containsConnectionGene(connectionGene))
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

	ConnectionGene Genome::getConnectionGeneByInnovationNumber(uint16_t innovationNumber) const
	{
		const auto it = std::ranges::find_if(connectionGenes, [innovationNumber](const ConnectionGene& connectionGene)
			{
				return connectionGene.getInnovationNumber() == innovationNumber;
			});

		if (it == connectionGenes.end())
			throw std::invalid_argument("Connection gene with the specified innovation number " +
				std::to_string(innovationNumber) + " does not exist.");

		return *it;
	}

}