#pragma once

#include <elements/element_factory.h>

#include "neat_tools/logger.h"
#include "neat_tools/utils.h"

namespace neat_dnfs
{
	using NeuralFieldPtr = std::shared_ptr<dnf_composer::element::NeuralField>;
	using KernelPtr = std::shared_ptr<dnf_composer::element::Kernel>;
	using FieldCouplingPtr = std::shared_ptr<dnf_composer::element::FieldCoupling>;
	using NormalNoisePtr = std::shared_ptr<dnf_composer::element::NormalNoise>;

	/// @brief Global simulation timing and step-count limits.
	struct SimulationConstants
	{
		inline static std::string name				= "solution ";
		static constexpr double deltaT				= 1;
		static constexpr size_t maxSimulationSteps	= 500;
	};

	struct DimensionConstants
	{
		static constexpr int xSize = 100;
		static constexpr double dx = 1.0;
	};

	struct NoiseConstants
	{
		inline static std::string namePrefix	= "nn ";
		static constexpr double amplitude		= 0.010;
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
		static constexpr double tau					= 100;
		static constexpr double restingLevel		= -10;
		inline static dnf_composer::element::SigmoidFunction activationFunction{0.0F, 5.0F};

		static constexpr double tauMinVal			= 1.0;
		static constexpr double tauMaxVal			= 200.0;
		static constexpr double tauStep				= 15.0;

		static constexpr double restingLevelMinVal	= -15.0;
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

		static constexpr double width			= 2.00;
		static constexpr double amplitude		= 8.00;
		static constexpr double amplitudeGlobal = -0.01;

		static constexpr double widthMinVal		= 1.00;
		static constexpr double widthMaxVal		= 10.0;
		static constexpr double widthStep		= 0.50;

		static constexpr double ampMinVal		= 3.00;
		static constexpr double ampMaxVal		= 30.0;
		static constexpr double ampStep			= 0.50;

		static constexpr double ampGlobalMinVal = -0.2;
		static constexpr double ampGlobalMaxVal = 0.00;
		static constexpr double ampGlobalStep	= 0.05;
	};

	struct MexicanHatKernelConstants
	{
		inline static std::string namePrefix				= "mhk ";
		inline static std::string namePrefixConnectionGene	= "mhk cg ";

		static constexpr double widthExc		= 2.50;
		static constexpr double widthInh		= 5.00;
		static constexpr double amplitudeExc	= 11.0;
		static constexpr double amplitudeInh	= 15.0;
		static constexpr double amplitudeGlobal = -0.01;

		static constexpr double widthExcMinVal	= 5.00;
		static constexpr double widthExcMaxVal	= 30.0;
		static constexpr double widthExcStep	= 0.50;

		static constexpr double widthInhMinVal	= 5.00;
		static constexpr double widthInhMaxVal	= 35.0;
		static constexpr double widthInhStep	= 0.50;

		static constexpr double ampExcMinVal	= 15.0;
		static constexpr double ampExcMaxVal	= 25.0;
		static constexpr double ampExcStep		= 0.50;

		static constexpr double ampInhMinVal	= 1.00;
		static constexpr double ampInhMaxVal	= 35.0;
		static constexpr double ampInhStep		= 0.50;

		static constexpr double ampGlobMin		= -0.20;
		static constexpr double ampGlobMax		= 0.000;
		static constexpr double ampGlobStep 	= 0.05;
	};

	/// @brief Weights used to compute the NEAT compatibility distance between two genomes.
	/// The distance determines whether two solutions belong to the same species.
	struct CompatibilityCoefficients
	{
		static constexpr double compatibilityThreshold							= 3.5;
		static constexpr double excessGenesCompatibilityWeight					= 1.0;
		static constexpr double disjointGenesCompatibilityWeight				= 0.5;
		static constexpr double averageConnectionDifferenceCompatibilityWeight	= 1.5;

		static constexpr double amplitudeDifferenceCoefficient	= 0.05;
		static constexpr double widthDifferenceCoefficient		= 0.05;
	};

	/// @brief Probabilities governing structural and parametric genome mutations.
	struct GenomeMutationConstants
	{
		// genome mutation probabilities (the sum does not have to be 1.0)
		// structural mutations
		static constexpr double toggleConnectionGeneProbability  = 0.01;//0.010;
		static constexpr double addFieldGeneProbability			 = 0.0005;//0.0005;
		static constexpr double addConnectionGeneProbability	 = 0.15;//0.250;
		// parametrical mutations
		static constexpr double mutateFieldGenesProbability		 = 0.800;
		static constexpr double mutateConnectionGenesProbability = 0.800;
		// per gene mutation probabilities
		static constexpr double mutateFieldGeneProbability		 = 0.800;
		static constexpr double mutateConnectionGeneProbability  = 0.800;

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
		static constexpr double mutateFieldGeneGaussKernelAmplitudeProbability			= 0.80;
		static constexpr double mutateFieldGeneGaussKernelWidthProbability				= 0.60;
		static constexpr double mutateFieldGeneGaussKernelGlobalAmplitudeProbability	= 0.20;
		// field gene mexican hat kernel mutation probabilities (sum does not have to be 1.0)
		static constexpr double mutateFieldGeneMexicanHatKernelAmplitudeExcProbability		= 0.80;
		static constexpr double mutateFieldGeneMexicanHatKernelAmplitudeInhProbability		= 0.80;
		static constexpr double mutateFieldGeneMexicanHatKernelWidthExcProbability			= 0.60;
		static constexpr double mutateFieldGeneMexicanHatKernelWidthInhProbability			= 0.60;
		static constexpr double mutateFieldGeneMexicanHatKernelGlobalAmplitudeProbability	= 0.20;

		// field gene neural field mutation probabilities (sum must be 1.0)
		static constexpr double mutateFieldGeneNeuralFieldParametersProbability					= 0.90;
		static constexpr double mutateFieldGeneNeuralFieldGenerateRandomParametersProbability	= 0.10;
		// field gene neural field parameters mutation probabilities (sum does not have to be 1.0)
		static constexpr double mutateFieldGeneNeuralFieldParametersTauProbability			= 0.50;
		static constexpr double mutateFieldGeneNeuralFieldParametersRestingLevelProbability	= 0.80;
	};

	struct ConnectionGeneConstants
	{
		static constexpr bool allowInhibitoryConnections = true;

		// connection gene kernel type probability (sum must be 1.0)
		static constexpr double gaussKernelProbability			= 0.8;
		static constexpr double mexicanHatKernelProbability		= 0.2;

		// connection gene mutation probabilities (sum must be 1.0)
		static constexpr double mutateConnectionGeneKernelProbability			= 0.70;
		static constexpr double mutateConnectionGeneKernelTypeProbability		= 0.05;
		static constexpr double mutateConnectionGeneConnectionSignalProbability = 0.25;

		// field gene gauss kernel mutation probabilities (sum does not have to be 1.0)
		static constexpr double mutateConnectionGeneGaussKernelAmplitudeProbability			= 0.80;
		static constexpr double mutateConnectionGeneGaussKernelWidthProbability				= 0.60;
		static constexpr double mutateConnectionGeneGaussKernelGlobalAmplitudeProbability	= 0.20;
		// field gene mexican hat kernel mutation probabilities (sum does not have to be 1.0)
		static constexpr double mutateConnectionGeneMexicanHatKernelAmplitudeExcProbability		= 0.80;
		static constexpr double mutateConnectionGeneMexicanHatKernelAmplitudeInhProbability		= 0.80;
		static constexpr double mutateConnectionGeneMexicanHatKernelWidthExcProbability			= 0.60;
		static constexpr double mutateConnectionGeneMexicanHatKernelWidthInhProbability			= 0.60;
		static constexpr double mutateConnectionGeneMexicanHatKernelGlobalAmplitudeProbability	= 0.20;
	};

	struct SolutionConstants
	{
		static constexpr uint8_t minInitialInputGenes	= 1;
		static constexpr uint8_t minInitialOutputGenes	= 1;
	};

	struct PopulationConstants
	{
		static constexpr double pruneRatio										= 0.8;
		static constexpr int generationsWithoutImprovementThresholdInPopulation = 10;
		static constexpr int generationsWithoutImprovementThresholdInSpecies	= 7;
		static constexpr bool elitism											= true;
		// Tolerance for the elitism validation check. The DNF simulation is
		// stochastic (see NoiseConstants::amplitude), so re-evaluating the same
		// preserved elite yields a fitness that drifts slightly rather than being
		// bit-identical; this covers that drift without masking a real regression.
		static constexpr double elitismFitnessEpsilon							= 2e-3;

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