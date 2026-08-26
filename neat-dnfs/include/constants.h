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
		static constexpr double deltaT				= 5;
		static constexpr size_t maxSimulationSteps	= 500;
	};

	struct DimensionConstants
	{
		static constexpr int xSize = 60;
		static constexpr double dx = 1.0;
	};

	struct NoiseConstants
	{
		inline static std::string namePrefix	= "nn ";
		static constexpr double amplitude		= 0.015;
	};

	struct GaussStimulusConstants
	{
		inline static std::string namePrefix	= "gs ";
		static constexpr double width			= 3.0; 
		static constexpr double amplitude		= 6.0; 
		static constexpr bool circularity		= true; 
		static constexpr bool normalization		= false;
	};

	struct NeuralFieldConstants
	{
		inline static std::string namePrefix		= "nf ";
		static constexpr double tau					= 25.0;
		static constexpr double restingLevel		= -5.0;
		inline static dnf_composer::element::SigmoidFunction activationFunction{0.0f, 5.0f};

		static constexpr double tauMinVal			= 5.0;
		static constexpr double tauMaxVal			= 50.0;
		static constexpr double tauStep				= 1.0;

		static constexpr double restingLevelMinVal	= -10.0;
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

		static constexpr double amplitude		= 5.00;
		static constexpr double width			= 2.00;
		static constexpr double amplitudeGlobal = -0.01;

		static constexpr double widthMinVal		= 1.00; 
		static constexpr double widthMaxVal		= 5.00; 
		static constexpr double widthStep		= 0.25; 

		static constexpr double ampMinVal		= 1.00; 
		static constexpr double ampMaxVal		= 30.0; 
		static constexpr double ampStep			= 1.00; // updated

		static constexpr double ampGlobalMinVal = -1.0;
		static constexpr double ampGlobalMaxVal = 0.00;
		static constexpr double ampGlobalStep	= 0.10;
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

		static constexpr double widthExcMinVal	= 1.00; 
		static constexpr double widthExcMaxVal	= 5.00; 
		static constexpr double widthExcStep	= 0.25; 

		static constexpr double widthInhMinVal	= 3.00; 
		static constexpr double widthInhMaxVal	= 10.0; 
		static constexpr double widthInhStep	= 0.50; 

		static constexpr double ampExcMinVal	= 5.00; 
		static constexpr double ampExcMaxVal	= 20.0; 
		static constexpr double ampExcStep		= 1.00; // updated

		static constexpr double ampInhMinVal	= 5.00; 
		static constexpr double ampInhMaxVal	= 20.0; 
		static constexpr double ampInhStep		= 1.00; 

		static constexpr double ampGlobMin		= -0.20;
		static constexpr double ampGlobMax		= 0.00; 		
		static constexpr double ampGlobStep 	= 0.05; 	};

	struct CompatibilityCoefficients
	{
		static constexpr double compatibilityThreshold							= 4.0; // updated
		static constexpr double excessGenesCompatibilityWeight					= 1.0; 
		static constexpr double disjointGenesCompatibilityWeight				= 0.5; 
		static constexpr double averageConnectionDifferenceCompatibilityWeight	= 0.5; 

		static constexpr double amplitudeDifferenceCoefficient	= 0.05; 
		static constexpr double widthDifferenceCoefficient		= 0.05; 
	};

	struct GenomeMutationConstants
	{
		// genome mutation probabilities (the sum does not have to be 1.0)
		// structural mutations
		static constexpr double toggleConnectionGeneProbability  = 0.005;
		static constexpr double addFieldGeneProbability			 = 0.001;// updated
		static constexpr double addConnectionGeneProbability	 = 0.15;
		// parametrical mutations
		static constexpr double mutateFieldGenesProbability		 = 0.500; // updated
		static constexpr double mutateConnectionGenesProbability = 0.500; // updated
		// per gene mutation probabilities
		static constexpr double mutateFieldGeneProbability		 = 0.400; // updated
		static constexpr double mutateConnectionGeneProbability  = 0.400; // updated

		static constexpr bool checkForDuplicateConnectionGenesInGenome = false;
	};

	struct FieldGeneConstants
	{
		static constexpr bool variableParameters = true;

		// (sum must be 1.0)
		static constexpr double gaussKernelProbability			= 0.8; 		
		static constexpr double mexicanHatKernelProbability		= 0.2; 
		// field gene mutation probabilities (sum must be 1.0)
		static constexpr double mutateFieldGeneKernelProbability			= 0.70;
		static constexpr double mutateFieldGeneKernelTypeProbability		= 0.10;
		static constexpr double mutateFieldGeneNeuralFieldProbability		= 0.20;
		// field gene gauss kernel mutation probabilities (sum does not have to be 1.0)
		static constexpr double mutateFieldGeneGaussKernelAmplitudeProbability			= 0.50; // updated
		static constexpr double mutateFieldGeneGaussKernelWidthProbability				= 0.30; // updated
		static constexpr double mutateFieldGeneGaussKernelGlobalAmplitudeProbability	= 0.10; // updated
		// field gene mexican hat kernel mutation probabilities (sum does not have to be 1.0)
		static constexpr double mutateFieldGeneMexicanHatKernelAmplitudeExcProbability		= 0.50; // updated
		static constexpr double mutateFieldGeneMexicanHatKernelAmplitudeInhProbability		= 0.50; // updated
		static constexpr double mutateFieldGeneMexicanHatKernelWidthExcProbability			= 0.30; //  updated	
		static constexpr double mutateFieldGeneMexicanHatKernelWidthInhProbability			= 0.30; // updated	
		static constexpr double mutateFieldGeneMexicanHatKernelGlobalAmplitudeProbability	= 0.10; // updated	

		// field gene neural field mutation probabilities (sum must be 1.0)
		static constexpr double mutateFieldGeneNeuralFieldParametersProbability					= 0.90;
		static constexpr double mutateFieldGeneNeuralFieldGenerateRandomParametersProbability	= 0.10;
		// field gene neural field parameters mutation probabilities (sum does not have to be 1.0)
		static constexpr double mutateFieldGeneNeuralFieldParametersTauProbability			= 0.15; // updated
		static constexpr double mutateFieldGeneNeuralFieldParametersRestingLevelProbability	= 0.50; // updated
	};

	struct ConnectionGeneConstants
	{
		static constexpr int allowInhibitoryConnections = true;

		// connection gene kernel type probability (sum must be 1.0)
		static constexpr double gaussKernelProbability			= 0.8; 
		static constexpr double mexicanHatKernelProbability		= 0.2; 

		// connection gene mutation probabilities (sum must be 1.0)
		static constexpr double mutateConnectionGeneKernelProbability			= 0.70; 
		static constexpr double mutateConnectionGeneKernelTypeProbability		= 0.10;  
		static constexpr double mutateConnectionGeneConnectionSignalProbability = 0.20; 

		// field gene gauss kernel mutation probabilities (sum does not have to be 1.0)
		static constexpr double mutateConnectionGeneGaussKernelAmplitudeProbability			= 0.50; // updated
		static constexpr double mutateConnectionGeneGaussKernelWidthProbability				= 0.30; // updated
		static constexpr double mutateConnectionGeneGaussKernelGlobalAmplitudeProbability	= 0.10; // updated
		// field gene mexican hat kernel mutation probabilities (sum does not have to be 1.0)
		static constexpr double mutateConnectionGeneMexicanHatKernelAmplitudeExcProbability		= 0.50; // updated
		static constexpr double mutateConnectionGeneMexicanHatKernelAmplitudeInhProbability		= 0.50; // updated
		static constexpr double mutateConnectionGeneMexicanHatKernelWidthExcProbability			= 0.30; // updated
		static constexpr double mutateConnectionGeneMexicanHatKernelWidthInhProbability			= 0.30; // updated
		static constexpr double mutateConnectionGeneMexicanHatKernelGlobalAmplitudeProbability	= 0.10; // updated
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
		inline static bool seedRandomConnections    = false;   // Random Initial Topology; count drawn as U[1, maxLegalConnectionCount()]

		// mechanism ablations (No Speciation, No Crossover)
		inline static bool disableSpeciation = false;
		inline static bool disableCrossover  = false;

		static void reset()
		{
			label = "";
			disableAddFieldGene = false;
			disableAddConnectionGene = false;
			disableToggleConnectionGene = false;
			seedAllLegalConnections = false;
			seedRandomHiddenFields = false;
			seedHiddenFieldsMin = 0;
			seedHiddenFieldsMax = 0;
			seedRandomConnections = false;
			disableSpeciation = false;
			disableCrossover = false;
		}
	};

	struct PopulationConstants
	{
		static constexpr bool parallelEvolution									= true;
		static constexpr double pruneRatio										= 0.8; 
		static constexpr int generationsWithoutImprovementThresholdInPopulation = 15; 
		static constexpr int generationsWithoutImprovementThresholdInSpecies	= 5; 
		static constexpr bool elitism											= true; 

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