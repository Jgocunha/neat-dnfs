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
		inline static std::string name				= "solution ";
		static constexpr double deltaT				= 25;
		static constexpr size_t maxSimulationSteps	= 2500;
	};

	struct DimensionConstants
	{
		static constexpr int xSize = 100;
		static constexpr double dx = 1.0;
	};

	struct NoiseConstants
	{
		inline static std::string namePrefix	= "nn ";
		static constexpr double amplitude		= 0.05;
	};

	struct GaussStimulusConstants
	{
		inline static std::string namePrefix	= "gs ";
		static constexpr double width			= 5.0;
		static constexpr double amplitude		= 20.0;
		static constexpr bool circularity		= true;
		static constexpr bool normalization		= false;
	};

	struct NeuralFieldConstants
	{
		inline static std::string namePrefix		= "nf ";
		static constexpr double stabilityThreshold	= 0.9;
		static constexpr double tau					= 200;
		static constexpr double restingLevel		= -5;
		inline static dnf_composer::element::HeavisideFunction activationFunction{ 0.0 };

		static constexpr double tauMinVal			= 220.0;
		static constexpr double tauMaxVal			= 300.0;
		static constexpr double tauStep				= 5.0;

		static constexpr double restingLevelMinVal	= -20.0;
		static constexpr double restingLevelMaxVal	= -1.0;
		static constexpr double restingLevelStep	= 0.5;
	};

	struct KernelConstants
	{
		static constexpr bool circularity	= true;
		static constexpr bool normalization = true;
	};

	struct GaussKernelConstants
	{
		inline static std::string namePrefix				= "gk ";
		inline static std::string namePrefixConnectionGene	= "gk cg ";

		static constexpr double width			= 6;
		static constexpr double amplitude		= 10;
		static constexpr double amplitudeGlobal = -0.01;

		static constexpr double widthMinVal		= 3.0; //3 // 3 // 1
		static constexpr double widthMaxVal		= 10.0; //25 // 5.0 // 50
		static constexpr double widthStep		= 0.5; // 0.5 // 1

		static constexpr double ampMinVal		= 3.0; // 0.5 // 1
		static constexpr double ampMaxVal		= 50.0; // 10.0 // 8.0 // 80
		static constexpr double ampStep			= 0.5; // 0.5 // 1

		static constexpr double ampGlobalMinVal = -3.5; // -0.5 // -5
		static constexpr double ampGlobalMaxVal = 0.0; // -0.01 // 0
		static constexpr double ampGlobalStep	= 0.1; // 0.01	 // 0.05
	};

	struct MexicanHatKernelConstants
	{
		inline static std::string namePrefix				= "mhk ";
		inline static std::string namePrefixConnectionGene	= "mhk cg ";

		static constexpr double widthExc		= 2.5; // 2.5
		static constexpr double widthInh		= 5.0; // 5.0
		static constexpr double amplitudeExc	= 11.0;  // 11.0
		static constexpr double amplitudeInh	= 15.0;  // 15.0
		static constexpr double amplitudeGlobal = -0.01; // -0.1

		static constexpr double widthExcMinVal	= 5.0; // 2.0 // 1
		static constexpr double widthExcMaxVal	= 30.0; // 9.0 // 50
		static constexpr double widthExcStep	= 0.5; // 0.5

		static constexpr double widthInhMinVal	= 5.0; // 5.0 // 1
		static constexpr double widthInhMaxVal	= 30.0; // 30.0 // 60
		static constexpr double widthInhStep	= 0.5; // 0.5

		static constexpr double ampExcMinVal	= 5.0; // 8.0
		static constexpr double ampExcMaxVal	= 45.0; // 25.0 //85
		static constexpr double ampExcStep		= 0.5; // 0.5

		static constexpr double ampInhMinVal	= 10.0; // 12.0 // 1
		static constexpr double ampInhMaxVal	= 25.0; // 30.0 // 60
		static constexpr double ampInhStep		= 0.5; // 0.5

		static constexpr double ampGlobMin	= -3.5; // -0.5 // -5
		static constexpr double ampGlobMax	= 0.0; // -0.01
		static constexpr double ampGlobStep = 0.1; // 0.01
	};

	struct OscillatoryKernelConstants
	{
		inline static std::string namePrefix				= "osck ";
		inline static std::string namePrefixConnectionGene	= "osck cg ";

		static constexpr double amplitude		= 1.0;
		static constexpr double decay			= 0.08;
		static constexpr double zeroCrossings	= 0.3;
		static constexpr double amplitudeGlobal = -0.01;

		static constexpr double amplitudeMinVal = 1;
		static constexpr double amplitudeMaxVal = 25.0;
		static constexpr double amplitudeStep	= 0.5;

		static constexpr double decayMinVal = 0.01;
		static constexpr double decayMaxVal = 1.0;
		static constexpr double decayStep	= 0.05;

		static constexpr double zeroCrossingsMinVal = 0.1;
		static constexpr double zeroCrossingsMaxVal = 1.0;
		static constexpr double zeroCrossingsStep	= 0.5;

		static constexpr double ampGlobMin		= -5.0;
		static constexpr double ampGlobMax		= 0.0;
		static constexpr double ampGlobStep		= 0.05;
	};

	struct CompatibilityCoefficients
	{
		static constexpr double compatibilityThreshold							= 3.0;
		static constexpr double excessGenesCompatibilityWeight					= 1.0;
		static constexpr double disjointGenesCompatibilityWeight				= 0.5;
		static constexpr double averageConnectionDifferenceCompatibilityWeight	= 1.0; //0.5

		static constexpr double amplitudeDifferenceCoefficient	= 0.05;
		static constexpr double widthDifferenceCoefficient		= 0.05;
	};

	struct GenomeMutationConstants
	{
		// genome mutation probabilities (sum does not have to be 1.0)
		// structural mutations
		static constexpr double toggleConnectionGeneProbability = 0.01;
		static constexpr double addFieldGeneProbability			= 0.005;
		static constexpr double addConnectionGeneProbability	= 0.20;
		// parametrical mutations
		static constexpr double mutateFieldGenesProbability		= 0.50;
		static constexpr double mutateConnectionGenesProbability = 0.50;
		// per gene mutation probabilities
		static constexpr double mutateFieldGeneProbability		= 0.20;
		static constexpr double mutateConnectionGeneProbability = 0.20;

		static constexpr bool checkForDuplicateConnectionGenesInGenome = false;
	};

	struct FieldGeneConstants
	{
		static constexpr bool variableParameters = true;

		// (sum must be 1.0)
		static constexpr double gaussKernelProbability			= 0.5;
		static constexpr double mexicanHatKernelProbability		= 0.5;

		// field gene mutation probabilities (sum must be 1.0)
		static constexpr double mutateFielGeneKernelProbability			= 0.70;
		static constexpr double mutateFielGeneKernelTypeProbability		= 0.10;
		static constexpr double mutateFieldGeneNeuralFieldProbability	= 0.20;
		// field gene gauss kernel mutation probabilities (sum does not have to be 1.0)
		static constexpr double mutateFieldGeneGaussKernelAmplitudeProbability			= 0.80;
		static constexpr double mutateFieldGeneGaussKernelWidthProbability				= 0.60;
		static constexpr double mutateFieldGeneGaussKernelGlobalAmplitudeProbability	= 0.10;
		// field gene mexican hat kernel mutation probabilities (sum does not have to be 1.0)
		static constexpr double mutateFieldGeneMexicanHatKernelAmplitudeExcProbability		= 0.80;
		static constexpr double mutateFieldGeneMexicanHatKernelAmplitudeInhProbability		= 0.80;
		static constexpr double mutateFieldGeneMexicanHatKernelWidthExcProbability			= 0.60;
		static constexpr double mutateFieldGeneMexicanHatKernelWidthInhProbability			= 0.60;
		static constexpr double mutateFieldGeneMexicanHatKernelGlobalAmplitudeProbability	= 0.10;

		// field gene neural field mutation probabilities (sum must be 1.0)
		static constexpr double mutateFieldGeneNeuralFieldParametersProbability	= 0.90;
		static constexpr double mutateFieldGeneNeuralFieldGenerateRandomParametersProbability	= 0.10;
		// field gene neural field parameters mutation probabilities (sum does not have to be 1.0)
		static constexpr double mutateFieldGeneNeuralFieldParametersTauProbability			= 0.20;
		static constexpr double mutateFieldGeneNeuralFieldParametersRestingLevelProbability	= 0.80;
	};

	struct ConnectionGeneConstants
	{
		static constexpr int allowInhibitoryConnections = true;

		// connection gene kernel type probability (sum must be 1.0)
		static constexpr double gaussKernelProbability			= 0.5;
		static constexpr double mexicanHatKernelProbability		= 0.5;
		static constexpr double oscillatoryKernelProbability	= 0.0;

		// connection gene mutation probabilities (sum must be 1.0)
		static constexpr double mutateConnectionGeneKernelProbability			= 0.70;
		static constexpr double mutateConnectionGeneKernelTypeProbability		= 0.10;
		static constexpr double mutateConnectionGeneConnectionSignalProbability = 0.20;

		// field gene gauss kernel mutation probabilities (sum does not have to be 1.0)
		static constexpr double mutateConnectionGeneGaussKernelAmplitudeProbability			= 0.80;
		static constexpr double mutateConnectionGeneGaussKernelWidthProbability				= 0.60;
		static constexpr double mutateConnectionGeneGaussKernelGlobalAmplitudeProbability	= 0.01;
		// field gene mexican hat kernel mutation probabilities (sum does not have to be 1.0)
		static constexpr double mutateConnectionGeneMexicanHatKernelAmplitudeExcProbability		= 0.80;
		static constexpr double mutateConnectionGeneMexicanHatKernelAmplitudeInhProbability		= 0.80;
		static constexpr double mutateConnectionGeneMexicanHatKernelWidthExcProbability			= 0.60;
		static constexpr double mutateConnectionGeneMexicanHatKernelWidthInhProbability			= 0.60;
		static constexpr double mutateConnectionGeneMexicanHatKernelGlobalAmplitudeProbability	= 0.01;
	};

	struct SolutionConstants
	{
		static constexpr uint8_t minInitialInputGenes	= 1;
		static constexpr uint8_t minInitialOutputGenes	= 1;
	};

	struct PopulationConstants
	{
		static constexpr bool parallelEvolution									= true;
		static constexpr double pruneRatio										= 0.2;
		static constexpr int generationsWithoutImprovementThresholdInPopulation = 20;
		static constexpr int generationsWithoutImprovementThresholdInSpecies	= 10;
		static constexpr bool elitism											= true;

		static constexpr bool validateUniqueSolutions					= true;
		static constexpr bool validatePopulationSize					= true;
		static constexpr bool validateElitism							= true;
		static constexpr bool validateUniqueGenesInGenomes				= true;
		static constexpr bool validateUniqueKernelAndNeuralFieldPtrs	= true;
		static constexpr bool validateIfSpeciesHaveUniqueRepresentative = true;
		static constexpr bool validateAssignmentIntoSpecies				= true;

		static constexpr bool logSolutions				= false;
		static constexpr bool logOverview				= true;
		static constexpr bool logSpecies				= false;
		static constexpr bool logMutationStatistics		= false;

		static constexpr bool saveChampions		= true;
		static constexpr bool saveStatistics	= true;
	};
}