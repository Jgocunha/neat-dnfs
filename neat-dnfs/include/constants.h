#pragma once

#include <dnf_composer/tools/logger.h>
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
		static constexpr double deltaT				= 5; // checks
		static constexpr size_t maxSimulationSteps	= 500; // checks
	};

	struct DimensionConstants
	{
		static constexpr int xSize = 60; // checks
		static constexpr double dx = 1.0; // checks
	};

	struct NoiseConstants
	{
		inline static std::string namePrefix	= "nn ";
		static constexpr double amplitude		= 0.015; // checks
	};

	struct GaussStimulusConstants
	{
		inline static std::string namePrefix	= "gs ";
		static constexpr double width			= 3.0; // checks
		static constexpr double amplitude		= 6.0; // checks
		static constexpr bool circularity		= true; // checks
		static constexpr bool normalization		= false; // checks
	};

	struct NeuralFieldConstants
	{
		inline static std::string namePrefix		= "nf ";
		static constexpr double tau					= 25.0;
		static constexpr double restingLevel		= -5.0;
		inline static dnf_composer::element::SigmoidFunction activationFunction{0.0f, 5.0f}; // checks with .pdf // different in paper!

		static constexpr double tauMinVal			= 5.0; // checks
		static constexpr double tauMaxVal			= 50.0; // checks
		static constexpr double tauStep				= 1.0; // checks

		static constexpr double restingLevelMinVal	= -10.0; // checks
		static constexpr double restingLevelMaxVal	= -1.0; // checks
		static constexpr double restingLevelStep	= 0.5; // checks
	};

	struct KernelConstants
	{
		static constexpr bool circularity	= true; // checks
		static constexpr bool normalization = true; // checks
	};

	struct GaussKernelConstants
	{
		inline static std::string namePrefix				= "gk ";
		inline static std::string namePrefixConnectionGene	= "gk cg ";

		static constexpr double amplitude		= 5.00;
		static constexpr double width			= 2.00;
		static constexpr double amplitudeGlobal = -0.01;

		static constexpr double widthMinVal		= 1.00; // checks
		static constexpr double widthMaxVal		= 5.00; // checks
		static constexpr double widthStep		= 0.25; // checks

		static constexpr double ampMinVal		= 1.00; // checks
		static constexpr double ampMaxVal		= 30.0; // checks
		static constexpr double ampStep			= 0.50; // checks

		static constexpr double ampGlobalMinVal = -1.0; // checks
		static constexpr double ampGlobalMaxVal = 0.00; // checks
		static constexpr double ampGlobalStep	= 0.10; // checks
	};

	struct MexicanHatKernelConstants
	{
		inline static std::string namePrefix				= "mhk ";
		inline static std::string namePrefixConnectionGene	= "mhk cg ";

		static constexpr double widthExc		= 2.00;
		static constexpr double widthInh		= 5.00;
		static constexpr double amplitudeExc	= 10.0;
		static constexpr double amplitudeInh	= 10.0;
		static constexpr double amplitudeGlobal = -0.01;

		static constexpr double widthExcMinVal	= 1.00; // checks
		static constexpr double widthExcMaxVal	= 5.00; // checks
		static constexpr double widthExcStep	= 0.25; // checks

		static constexpr double widthInhMinVal	= 3.00; // checks
		static constexpr double widthInhMaxVal	= 10.0; // checks
		static constexpr double widthInhStep	= 0.50; // checks

		static constexpr double ampExcMinVal	= 5.00; // checks
		static constexpr double ampExcMaxVal	= 20.0; // checks
		static constexpr double ampExcStep		= 0.50; // checks

		static constexpr double ampInhMinVal	= 5.00; // checks
		static constexpr double ampInhMaxVal	= 20.0; // checks
		static constexpr double ampInhStep		= 0.50; // checks

		static constexpr double ampGlobMin		= -0.20; // checks
		static constexpr double ampGlobMax		= 0.00; // checks
		static constexpr double ampGlobStep 	= 0.05; // checks
	};

	struct CompatibilityCoefficients
	{
		static constexpr double compatibilityThreshold							= 5.0; // checks with .pdf // different in paper!
		static constexpr double excessGenesCompatibilityWeight					= 1.0; // checks
		static constexpr double disjointGenesCompatibilityWeight				= 0.5; // checks
		static constexpr double averageConnectionDifferenceCompatibilityWeight	= 0.5; // checks

		static constexpr double amplitudeDifferenceCoefficient	= 0.05; // checks
		static constexpr double widthDifferenceCoefficient		= 0.05; // checks
	};

	struct GenomeMutationConstants
	{
		// genome mutation probabilities (the sum does not have to be 1.0)
		// structural mutations
		static constexpr double toggleConnectionGeneProbability  = 0.005; // checks
		static constexpr double addFieldGeneProbability			 = 0.0005;// checks with .pdf // different in paper!
		static constexpr double addConnectionGeneProbability	 = 0.15; // checks
		// parametrical mutations
		static constexpr double mutateFieldGenesProbability		 = 0.800; // checks with .pdf // different in paper!
		static constexpr double mutateConnectionGenesProbability = 0.800; // checks with .pdf // different in paper!
		// per gene mutation probabilities
		static constexpr double mutateFieldGeneProbability		 = 0.800; // checks with .pdf // different in paper!
		static constexpr double mutateConnectionGeneProbability  = 0.800; // checks with .pdf // different in paper!

		static constexpr bool checkForDuplicateConnectionGenesInGenome = false;
	};

	struct FieldGeneConstants
	{
		static constexpr bool variableParameters = true;

		// (sum must be 1.0)
		static constexpr double gaussKernelProbability			= 0.8; // checks
		static constexpr double mexicanHatKernelProbability		= 0.2; // checks

		// field gene mutation probabilities (sum must be 1.0)
		static constexpr double mutateFieldGeneKernelProbability			= 0.70; // checks
		static constexpr double mutateFieldGeneKernelTypeProbability		= 0.10; // checks
		static constexpr double mutateFieldGeneNeuralFieldProbability		= 0.20; // checks
		// field gene gauss kernel mutation probabilities (sum does not have to be 1.0)
		static constexpr double mutateFieldGeneGaussKernelAmplitudeProbability			= 0.80; // checks
		static constexpr double mutateFieldGeneGaussKernelWidthProbability				= 0.60; // checks
		static constexpr double mutateFieldGeneGaussKernelGlobalAmplitudeProbability	= 0.20; // checks
		// field gene mexican hat kernel mutation probabilities (sum does not have to be 1.0)
		static constexpr double mutateFieldGeneMexicanHatKernelAmplitudeExcProbability		= 0.80; // checks
		static constexpr double mutateFieldGeneMexicanHatKernelAmplitudeInhProbability		= 0.80; // checks
		static constexpr double mutateFieldGeneMexicanHatKernelWidthExcProbability			= 0.60; // checks
		static constexpr double mutateFieldGeneMexicanHatKernelWidthInhProbability			= 0.60; // checks
		static constexpr double mutateFieldGeneMexicanHatKernelGlobalAmplitudeProbability	= 0.20; // checks

		// field gene neural field mutation probabilities (sum must be 1.0)
		static constexpr double mutateFieldGeneNeuralFieldParametersProbability					= 0.90; // checks
		static constexpr double mutateFieldGeneNeuralFieldGenerateRandomParametersProbability	= 0.10; // checks
		// field gene neural field parameters mutation probabilities (sum does not have to be 1.0)
		static constexpr double mutateFieldGeneNeuralFieldParametersTauProbability			= 0.50; // checks
		static constexpr double mutateFieldGeneNeuralFieldParametersRestingLevelProbability	= 0.80; // checks
	};

	struct ConnectionGeneConstants
	{
		static constexpr int allowInhibitoryConnections = true; // true

		// connection gene kernel type probability (sum must be 1.0)
		static constexpr double gaussKernelProbability			= 0.8; // checks
		static constexpr double mexicanHatKernelProbability		= 0.2; // checks

		// connection gene mutation probabilities (sum must be 1.0)
		static constexpr double mutateConnectionGeneKernelProbability			= 0.70; // checks
		static constexpr double mutateConnectionGeneKernelTypeProbability		= 0.10;  // checks
		static constexpr double mutateConnectionGeneConnectionSignalProbability = 0.20;  // checks

		// field gene gauss kernel mutation probabilities (sum does not have to be 1.0)
		static constexpr double mutateConnectionGeneGaussKernelAmplitudeProbability			= 0.80; // checks
		static constexpr double mutateConnectionGeneGaussKernelWidthProbability				= 0.60; // checks
		static constexpr double mutateConnectionGeneGaussKernelGlobalAmplitudeProbability	= 0.20; // checks
		// field gene mexican hat kernel mutation probabilities (sum does not have to be 1.0)
		static constexpr double mutateConnectionGeneMexicanHatKernelAmplitudeExcProbability		= 0.80; // checks
		static constexpr double mutateConnectionGeneMexicanHatKernelAmplitudeInhProbability		= 0.80; // checks
		static constexpr double mutateConnectionGeneMexicanHatKernelWidthExcProbability			= 0.60; // checks
		static constexpr double mutateConnectionGeneMexicanHatKernelWidthInhProbability			= 0.60; // checks
		static constexpr double mutateConnectionGeneMexicanHatKernelGlobalAmplitudeProbability	= 0.20; // checks
	};

	struct SolutionConstants
	{
		static constexpr uint8_t minInitialInputGenes	= 1;
		static constexpr uint8_t minInitialOutputGenes	= 1;
	};

	struct AblationConstants
	{
		inline static std::string label = "";   // appended to solution name -> data/<name>/ folder

		// structural freezes (No Growth IO Only, No Growth One Hidden)
		inline static bool disableAddFieldGene      = false;
		inline static bool disableAddConnectionGene = false;
		inline static bool disableToggleConnectionGene = false;

		// seeding (No Growth IO Only, No Growth One Hidden, Random Initial Topology)
		inline static bool seedAllLegalConnections  = false;   // No Growth IO Only, No Growth One Hidden
		inline static bool seedRandomHiddenFields   = false;   // No Growth One Hidden (min=max=1), Random Initial Topology (0..5)
		inline static int  seedHiddenFieldsMin      = 0;
		inline static int  seedHiddenFieldsMax      = 0;
		inline static bool seedRandomConnections    = false;   // Random Initial Topology
		inline static int  seedConnectionsMin       = 1;
		inline static int  seedConnectionsMax       = 8;

		// mechanism ablations (No Speciation, No Crossover)
		inline static bool disableSpeciation = false;
		inline static bool disableCrossover  = false;
	};

	struct PopulationConstants
	{
		static constexpr bool parallelEvolution									= true; // checks
		static constexpr double pruneRatio										= 0.8; // checks
		static constexpr int generationsWithoutImprovementThresholdInPopulation = 15; // checks
		static constexpr int generationsWithoutImprovementThresholdInSpecies	= 5; // checks
		static constexpr bool elitism											= true; // checks

		static constexpr bool validateUniqueSolutions					= false;
		static constexpr bool validatePopulationSize					= false;
		static constexpr bool validateElitism							= false; // true //WARNING  Fitness decreased and previous best solution is not in the population.
		static constexpr bool validateUniqueGenesInGenomes				= false;
		static constexpr bool validateUniqueKernelAndNeuralFieldPtrs	= false;
		static constexpr bool validateIfSpeciesHaveUniqueRepresentative = false; // true //FATAL Species have the same representative
		static constexpr bool validateAssignmentIntoSpecies				= false;

		static constexpr bool logSolutions				= false;
		static constexpr bool logOverview				= true;
		static constexpr bool logSpecies				= false;

		static constexpr bool saveOverview				= true;
		static constexpr bool savePerGenerationOverview	= true;
		static constexpr bool saveChampions				= true;
		static constexpr bool saveBestSolutions			= true;
		static constexpr bool saveSolutions				= true;
		static constexpr bool saveSpecies				= true;
	};
}