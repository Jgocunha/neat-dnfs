#include "neat/genome.h"

namespace neat_dnfs
{
	std::map<ConnectionTuple, uint16_t> Genome::connectionToInnovationNumberMap;

	void Genome::addInputGene()
	{
		const auto index = fieldGenes.size() + 1;
		fieldGenes.push_back(FieldGene({FieldGeneType::INPUT,
			static_cast<uint16_t>(index) }));
	}

	void Genome::addOutputGene()
	{
		const auto index = fieldGenes.size() + 1;
		fieldGenes.push_back(FieldGene({ FieldGeneType::OUTPUT,
			static_cast<uint16_t>(index) }));
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
		static constexpr double addGeneProbability = 0.1;
		static constexpr double mutateGeneProbability = 0.2;
		static constexpr double addConnectionGeneProbability = 0.3;
		static constexpr double mutateConnectionGeneProbability = 0.35;
		static constexpr double toggleConnectionGeneProbability = 0.05;

		if (addGeneProbability > static_cast<double>(rand()) / RAND_MAX)
			addGene();

		if (mutateGeneProbability > static_cast<double>(rand()) / RAND_MAX)
			mutateGene();

		if (addConnectionGeneProbability > static_cast<double>(rand()) / RAND_MAX)
			addConnectionGene();

		if (mutateConnectionGeneProbability > static_cast<double>(rand()) / RAND_MAX)
			mutateConnectionGene();

		if (toggleConnectionGeneProbability > static_cast<double>(rand()) / RAND_MAX)
			toggleConnectionGene();
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
		// Select two different random genes
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<uint16_t> dis(1, fieldGenes.size());
		uint16_t geneIndex1 = dis(gen);

		// Ensure geneIndex2 is different from geneIndex1
		uint16_t geneIndex2;
		do {
			geneIndex2 = dis(gen);
		} while (geneIndex2 == geneIndex1);

		// if gene with index 2 is an input gene return
		if (fieldGenes[geneIndex2 - 1].getParameters().type == FieldGeneType::INPUT)
			return {0,0};

		// If a connection gene between the two genes already exists, return
		if (std::find_if(connectionGenes.begin(), connectionGenes.end(),
			[geneIndex1, geneIndex2](const ConnectionGene& connectionGene) 
			{
				return connectionGene.getParameters().connectionTuple.inFieldGeneId == geneIndex1
				&& connectionGene.getParameters().connectionTuple.outFieldGeneId == geneIndex2;
			} ) != connectionGenes.end())
			return {0,0};

		return {geneIndex1, geneIndex2};
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

		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<uint16_t> dis(0, geneIds.size() - 1);

		return geneIds[dis(gen)];
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

		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<uint16_t> dis(0, enabledConnectionGenes.size() - 1);

		return enabledConnectionGenes[dis(gen)];
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
				connectionGenes[connectionGenes.size() - 1].getInnovationNumber() });
		}
	}

	void Genome::addGene()
	{
		using namespace dnf_composer::element;

		auto randEnabledConnectionGene = getEnabledConnectionGene();
		if (randEnabledConnectionGene.getInFieldGeneId() == 0)
			return;
		randEnabledConnectionGene.disable();
		const auto inGeneId
			= randEnabledConnectionGene.getParameters().connectionTuple.inFieldGeneId;
		const auto outGeneId
			= randEnabledConnectionGene.getParameters().connectionTuple.outFieldGeneId;
		const auto kernel = randEnabledConnectionGene.getKernel();
		const auto kernelParameters = 
			std::dynamic_pointer_cast<GaussKernel>(kernel)->getParameters();

		const auto index = fieldGenes.size() + 1;
		fieldGenes.push_back(FieldGene({ FieldGeneType::HIDDEN,
			static_cast<uint16_t>(index) }));

		// create two new connection genes
		const auto connectionGeneKernelParametersIn = 
			GaussKernelParameters{ 5, 10, false, false };
		const auto connectionGeneKernelParametersOut = 
			GaussKernelParameters{ kernelParameters.sigma, kernelParameters.amplitude,
				false, false };
		const ConnectionGene connectionGeneIn{ {inGeneId,
			static_cast<uint16_t>(index)}, connectionGeneKernelParametersIn };
		const ConnectionGene connectionGeneOut{ {static_cast<uint16_t>(index),
			outGeneId}, connectionGeneKernelParametersOut };
		connectionGenes.push_back(connectionGeneIn);
		connectionGenes.push_back(connectionGeneOut);
	}

	void Genome::mutateGene() const
	{
		const auto geneId = getRandomGeneIdByType(FieldGeneType::HIDDEN);
		if (geneId == 0)
			return;
		fieldGenes[geneId-1].mutate();
	}

	void Genome::addConnectionGene()
	{
		const ConnectionTuple connectionTuple = getNewRandomConnectionGeneTuple();
		if (connectionTuple == ConnectionTuple{0, 0})
			return;
		addConnectionGeneIfNewWithinGeneration(connectionTuple);
	}

	void Genome::mutateConnectionGene()
	{
		if (connectionGenes.empty())
			return;

		const auto connectionGeneId = rand() % connectionGenes.size();
		connectionGenes[connectionGeneId].mutate();
	}

	void Genome::toggleConnectionGene()
	{
		if (connectionGenes.empty())
			return;
		const auto connectionGeneId = rand() % connectionGenes.size();
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

		if (!thisInnovationNumbers.empty())
			thisMaxInnovationNumber = *std::max_element(thisInnovationNumbers.begin(),
				thisInnovationNumbers.end());

		if(!otherInnovationNumbers.empty())
			otherMaxInnovationNumber = *std::max_element(otherInnovationNumbers.begin(),
									otherInnovationNumbers.end());

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

		std::sort(sortedThisInnovationNumbers.begin(), sortedThisInnovationNumbers.end());
		std::sort(sortedOtherInnovationNumbers.begin(), sortedOtherInnovationNumbers.end());

		std::vector<uint16_t> thisDisjointInnovationNumbers, otherDisjointInnovationNumbers;

		// Get the elements in sortedThisInnovationNumbers that are not in sortedOtherInnovationNumbers
		std::set_difference(sortedThisInnovationNumbers.begin(), sortedThisInnovationNumbers.end(),
						sortedOtherInnovationNumbers.begin(), sortedOtherInnovationNumbers.end(),
						std::inserter(thisDisjointInnovationNumbers, thisDisjointInnovationNumbers.begin()));
		// Get the elements in sortedOtherInnovationNumbers that are not in sortedThisInnovationNumbers
		std::set_difference(sortedOtherInnovationNumbers.begin(), sortedOtherInnovationNumbers.end(),
									sortedThisInnovationNumbers.begin(), sortedThisInnovationNumbers.end(),
									std::inserter(otherDisjointInnovationNumbers, otherDisjointInnovationNumbers.begin()));

		// Calculate the maximum and minimum innovation numbers
		const int thisMaxInnovationNumber = !thisInnovationNumbers.empty() ? *std::max_element(thisInnovationNumbers.begin(),
									thisInnovationNumbers.end()) : 0;
		const int otherMaxInnovationNumber = !otherInnovationNumbers.empty() ? *std::max_element(otherInnovationNumbers.begin(),
												otherInnovationNumbers.end()) : 0;
		int minMaxInnovationNumber = std::min(thisMaxInnovationNumber, otherMaxInnovationNumber);

		// Filter out the disjoint genes that are beyond the range of the other genome
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
		return std::find(connectionGenes.begin(), connectionGenes.end(), connectionGene) != connectionGenes.end();
	}

	bool Genome::containsFieldGene(const FieldGene& fieldGene) const
	{
		return std::find(fieldGenes.begin(), fieldGenes.end(), fieldGene) != fieldGenes.end();
	}

	ConnectionGene Genome::getConnectionGeneByInnovationNumber(uint16_t innovationNumber) const
	{
		const auto it = std::find_if(connectionGenes.begin(), connectionGenes.end(),
		[innovationNumber](const ConnectionGene& connectionGene)
		{
			return connectionGene.getInnovationNumber() == innovationNumber;
		});

		//if (it == connectionGenes.end())
			//throw std::invalid_argument("Connection gene with the specified innovation number does not exist.");

		return *it;
	}

}