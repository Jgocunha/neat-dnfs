#pragma once

#include <elements/element_factory.h>

namespace neat_dnfs
{
	typedef std::shared_ptr<dnf_composer::element::NeuralField> NeuralFieldPtr;
	typedef std::shared_ptr<dnf_composer::element::Kernel> KernelPtr;
	typedef std::shared_ptr<dnf_composer::element::FieldCoupling> FieldCouplingPtr;
	typedef std::shared_ptr<dnf_composer::element::NormalNoise> NormalNoisePtr;

	struct SimulationConstants
	{
		inline static std::string name = "solution ";
		static constexpr double deltaT = 25.0;
		static constexpr size_t maxSimulationSteps = 200;
	};

	struct DimensionConstants
	{
		static constexpr int xSize = 100;
		static constexpr double dx = 1.0;
	};

	struct NoiseConstants
	{
		inline static std::string namePrefix = "nn ";
		static constexpr double amplitude = 0.2; //0.2 (if selection is required), null otherwise
	};

	struct GaussStimulusConstants
	{
		inline static std::string namePrefix = "gs ";
		static constexpr double width = 5.0;
		static constexpr double amplitude = 20.0;
		static constexpr bool circularity = true;
		static constexpr bool normalization = false;
	};

	struct NeuralFieldConstants
	{
		inline static std::string namePrefix = "nf ";
		static constexpr double stabilityThreshold = 0.9; // 0.90 if amp noise is 0.2, 0.035 otherwise
		static constexpr double tau = 200;
		static constexpr double restingLevel = -5;
		inline static dnf_composer::element::HeavisideFunction activationFunction{ 0.0 };
		static constexpr double tauMinVal = 15.0;
		static constexpr double tauMaxVal = 200.0;
		static constexpr double tauStep = 5.0;
		static constexpr double restingLevelMinVal = -20.0;
		static constexpr double restingLevelMaxVal = -1.0;
		static constexpr double restingLevelStep = 0.5;
	};

	struct KernelConstants
	{
		static constexpr bool circularity = true;
		static constexpr bool normalization = true;
	};

	struct GaussKernelConstants
	{
		inline static std::string namePrefix = "gk ";
		inline static std::string namePrefixConnectionGene = "gk cg ";

		static constexpr double width = 6;
		static constexpr double amplitude = 10;
		static constexpr double amplitudeGlobal = -0.01;

		static constexpr double widthMinVal = 1.0; // 3
		static constexpr double widthMaxVal = 30.0; // 5.0
		static constexpr double widthStep = 0.5; // 0.5
		static constexpr double ampMinVal = 1.0; // 0.5
		static constexpr double ampMaxVal = 60.0; // 8.0
		static constexpr double ampStep = 0.5; // 0.5
		static constexpr double ampGlobalMinVal = -5.0; // -0.5
		static constexpr double ampGlobalMaxVal = 0.0; // -0.01
		static constexpr double ampGlobalStep = 0.05; // 0.01	 
	};

	struct MexicanHatKernelConstants
	{
		inline static std::string namePrefix = "mhk ";
		inline static std::string namePrefixConnectionGene = "mhk cg ";

		static constexpr double widthExc = 2.5; // 2.5
		static constexpr double widthInh = 5.0; // 5.0
		static constexpr double amplitudeExc = 11.0;  // 11.0
		static constexpr double amplitudeInh = 15.0;  // 15.0
		static constexpr double amplitudeGlobal = -0.01; // -0.1

		static constexpr double widthExcMinVal = 1.0; // 2.0
		static constexpr double widthExcMaxVal = 30.0; // 9.0
		static constexpr double widthExcStep = 0.5; // 0.5

		static constexpr double widthInhMinVal = 1.0; // 5.0
		static constexpr double widthInhMaxVal = 30.0; // 30.0
		static constexpr double widthInhStep = 0.5; // 0.5

		static constexpr double ampExcMinVal = 1.0; // 8.0
		static constexpr double ampExcMaxVal = 65.0; // 25.0
		static constexpr double ampExcStep = 0.5; // 0.5

		static constexpr double ampInhMinVal = 1.0; // 12.0
		static constexpr double ampInhMaxVal = 30.0; // 30.0
		static constexpr double ampInhStep = 0.5; // 0.5

		static constexpr double ampGlobMin = -5.0; // -0.5
		static constexpr double ampGlobMax = 0.0; // -0.01
		static constexpr double ampGlobStep = 0.05; // 0.01
	};

	struct OscillatoryKernelConstants
	{
		inline static std::string namePrefix = "osck ";
		inline static std::string namePrefixConnectionGene = "osck cg ";

		static constexpr double amplitude = 1.0;
		static constexpr double decay = 0.08;
		static constexpr double zeroCrossings = 0.3;
		static constexpr double amplitudeGlobal = -0.01;

		static constexpr double amplitudeMinVal = 1;
		static constexpr double amplitudeMaxVal = 25.0;
		static constexpr double amplitudeStep = 0.5;

		static constexpr double decayMinVal = 0.01;
		static constexpr double decayMaxVal = 1.0;
		static constexpr double decayStep = 0.05;

		static constexpr double zeroCrossingsMinVal = 0.1;
		static constexpr double zeroCrossingsMaxVal = 1.0;
		static constexpr double zeroCrossingsStep = 0.5;

		static constexpr double ampGlobMin = -5.0;
		static constexpr double ampGlobMax = 0.0;
		static constexpr double ampGlobStep = 0.05;
	};

	struct CompatibilityCoefficients
	{
		static constexpr double compatibilityThreshold = 5.0;
		static constexpr double excessGenesCompatibilityWeight = 1.0;
		static constexpr double disjointGenesCompatibilityWeight = 0.5;
		static constexpr double averageConnectionDifferenceCompatibilityWeight = 0.1;

		static constexpr double amplitudeDifferenceCoefficient = 0.05;
		static constexpr double widthDifferenceCoefficient = 0.05;
	};

	struct GenomeMutationConstants
	{
		// genome mutation probabilities (sum must be 1.0)
		static constexpr double addFieldGeneProbability = 0.0025; // 0.0025
		static constexpr double mutateFieldGeneProbability = 0.35; // 0.35
		static constexpr double addConnectionGeneProbability = 0.20; // 0.20
		static constexpr double mutateConnectionGeneProbability = 0.44; // 0.44
		static constexpr double toggleConnectionGeneProbability = 0.0075; // 0.0075

		static constexpr bool checkForDuplicateConnectionGenesInGenome = false;
	};

	struct FieldGeneConstants
	{
		static constexpr bool variableParameters = true;

		static constexpr double gaussKernelProbability = 0.5;
		static constexpr double mexicanHatKernelProbability = 0.5;
		static constexpr double oscillatoryKernelProbability = 0.0;

		// field gene mutation probabilities (sum must be 1.0)
		static constexpr double mutateFielGeneKernelProbability = 0.50;
		static constexpr double mutateFielGeneKernelTypeProbability = 0.10;
		static constexpr double mutateFieldGeneNeuralFieldProbability = 0.40;
		// field gene gauss kernel mutation probabilities (sum must be 1.0)
		static constexpr double mutateFieldGeneGaussKernelAmplitudeProbability = 1.0f / 3.0f;
		static constexpr double mutateFieldGeneGaussKernelWidthProbability = 1.0f / 3.0f;
		static constexpr double mutateFieldGeneGaussKernelGlobalAmplitudeProbability = 1.0f / 3.0f;
		// field gene mexican hat kernel mutation probabilities (sum must be 1.0)
		static constexpr double mutateFieldGeneMexicanHatKernelAmplitudeExcProbability = 1.0f / 5.0f;
		static constexpr double mutateFieldGeneMexicanHatKernelAmplitudeInhProbability = 1.0f / 5.0f;
		static constexpr double mutateFieldGeneMexicanHatKernelWidthExcProbability = 1.0f / 5.0f;
		static constexpr double mutateFieldGeneMexicanHatKernelWidthInhProbability = 1.0f / 5.0f;
		static constexpr double mutateFieldGeneMexicanHatKernelGlobalAmplitudeProbability = 1.0f / 5.0f;
		// field gene oscillatory kernel mutation probabilities (sum must be 1.0)
		static constexpr double mutateFieldGeneOscillatoryKernelAmplitudeProbability = 1.0f / 4.0f;
		static constexpr double mutateFieldGeneOscillatoryKernelDecayProbability = 1.0f / 4.0f;
		static constexpr double mutateFieldGeneOscillatoryKernelZeroCrossingsProbability = 1.0f / 4.0f;
		static constexpr double mutateFieldGeneOscillatoryKernelGlobalAmplitudeProbability = 1.0f / 4.0f;
		// field gene neural field mutation probabilities (sum must be 1.0)
		static constexpr double mutateFieldGeneNeuralFieldTauProbability = 0.45;
		static constexpr double mutateFieldGeneNeuralFieldRestingLevelProbability = 0.45;
		static constexpr double mutateFieldGeneNeuralFieldRandomlyProbability = 0.10;
	};

	struct ConnectionGeneConstants
	{
		static constexpr int allowInhibitoryConnections = true;

		static constexpr double gaussKernelProbability = 0.5;
		static constexpr double mexicanHatKernelProbability = 0.5;
		static constexpr double oscillatoryKernelProbability = 0.0;

		// connection gene mutation probabilities (sum must be 1.0)
		static constexpr double mutateConnectionGeneKernelProbability = 0.80;
		static constexpr double mutateConnectionGeneKernelTypeProbability = 0.10;
		static constexpr double mutateConnectionGeneConnectionSignalProbability = 0.10;

		// field gene gauss kernel mutation probabilities (sum must be 1.0)
		static constexpr double mutateConnectionGeneGaussKernelAmplitudeProbability = 1.0f / 2.0f;
		static constexpr double mutateConnectionGeneGaussKernelWidthProbability = 1.0f / 2.0f;
		static constexpr double mutateConnectionGeneGaussKernelGlobalAmplitudeProbability = 0.0f;
		// field gene mexican hat kernel mutation probabilities (sum must be 1.0)
		static constexpr double mutateConnectionGeneMexicanHatKernelAmplitudeExcProbability = 1.0f / 4.0f;
		static constexpr double mutateConnectionGeneMexicanHatKernelAmplitudeInhProbability = 1.0f / 4.0f;
		static constexpr double mutateConnectionGeneMexicanHatKernelWidthExcProbability = 1.0f / 4.0f;
		static constexpr double mutateConnectionGeneMexicanHatKernelWidthInhProbability = 1.0f / 4.0f;
		static constexpr double mutateConnectionGeneMexicanHatKernelGlobalAmplitudeProbability = 0.0f;
		// field gene oscillatory kernel mutation probabilities (sum must be 1.0)
		static constexpr double mutateConnectionGeneOscillatoryKernelAmplitudeProbability = 1.0f / 3.0f;
		static constexpr double mutateConnectionGeneOscillatoryKernelDecayProbability = 1.0f / 3.0f;
		static constexpr double mutateConnectionGeneOscillatoryKernelZeroCrossingsProbability = 1.0f / 3.0f;
		static constexpr double mutateConnectionGeneOscillatoryKernelGlobalAmplitudeProbability = 0.0f;
	};

	struct SolutionConstants
	{
		static constexpr uint8_t minInitialInputGenes = 1;
		static constexpr uint8_t minInitialOutputGenes = 1;
	};

	struct PopulationConstants
	{
		static constexpr bool parallelEvolution = true;
		static constexpr double pruneRatio = 0.2;
		static constexpr int generationsWithoutImprovementThresholdInPopulation = 10;
		static constexpr int generationsWithoutImprovementThresholdInSpecies = 5;
		static constexpr bool elitism = true;

		static constexpr bool validateUniqueSolutions					= false;
		static constexpr bool validatePopulationSize					= false;
		static constexpr bool validateElitism							= false;
		static constexpr bool validateUniqueGenesInGenomes				= false;
		static constexpr bool validateUniqueKernelAndNeuralFieldPtrs	= false;
		static constexpr bool validateIfSpeciesHaveUniqueRepresentative = false;
		static constexpr bool validateAssignmentIntoSpecies				= false;

		static constexpr bool logSolutions = false;
		static constexpr bool logOverview = true;
		static constexpr bool logSpecies = false;
		static constexpr bool logMutationStatistics = false;

		static constexpr bool saveChampions = true;
		static constexpr bool saveStatistics = true;
	};
}