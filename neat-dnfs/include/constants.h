#pragma once

#include <string>
#include <string_view>
#include <vector>

#include <elements/element_factory.h>

#include "neat_tools/logger.h"
#include "neat_tools/utils.h"

namespace neat_dnfs
{
	using NeuralFieldPtr = std::shared_ptr<dnf_composer::element::NeuralField>;
	using KernelPtr = std::shared_ptr<dnf_composer::element::Kernel>;
	using FieldCouplingPtr = std::shared_ptr<dnf_composer::element::FieldCoupling>;
	using NormalNoisePtr = std::shared_ptr<dnf_composer::element::NormalNoise>;

	// Every tunable value below is declared without an initializer and is
	// populated at startup by ConfigLoader from config/neat_dnfs.json, with any
	// per-experiment overrides from config/solutions/<task>.json merged over it
	// (see include/neat_tools/config_loader.h). There are
	// deliberately no compiled-in fallbacks: a missing config file or key is a
	// hard startup error, so a run can never silently use a value that is not
	// recorded in the config it was launched with.
	//
	// The name/namePrefix string_views are the exception and stay compile-time.
	// They are not tuning knobs: they build element unique names via
	// std::format and are parsed back with starts_with/substr in solution.cpp,
	// and every file in templates/ and data/ hardcodes the strings they
	// produce ("nf 1", "gk 1", ...). Making them configurable would let a
	// config desync from that corpus with nothing to catch it.

	/// @brief Global simulation timing and step-count limits.
	struct SimulationConstants
	{
		static constexpr std::string_view name		= "solution ";
		inline static double deltaT;
		inline static size_t maxSimulationSteps;
	};

	struct DimensionConstants
	{
		inline static int xSize;
		inline static double dx;
	};

	struct NoiseConstants
	{
		static constexpr std::string_view namePrefix	= "nn ";
		inline static double amplitude;
	};

	struct GaussStimulusConstants
	{
		static constexpr std::string_view namePrefix	= "gs ";
		inline static double width;
		inline static double amplitude;
		inline static bool circularity;
		inline static bool normalization;
	};

	struct NeuralFieldConstants
	{
		static constexpr std::string_view namePrefix	= "nf ";
		inline static double tau;
		inline static double restingLevel;
		// SigmoidFunction has no default constructor, so an initialiser is
		// required here. These placeholder numbers are not defaults: both
		// members are overwritten from config/neat_dnfs.json before first use,
		// and a failed load throws rather than leaving them in place.
		inline static dnf_composer::element::SigmoidFunction activationFunction{ 0.0, 0.0 };

		inline static double tauMinVal;
		inline static double tauMaxVal;
		inline static double tauStep;

		inline static double restingLevelMinVal;
		inline static double restingLevelMaxVal;
		inline static double restingLevelStep;
	};

	struct KernelConstants
	{
		inline static bool circularity;
		inline static bool normalization;
	};

	struct GaussKernelConstants
	{
		static constexpr std::string_view namePrefix				= "gk ";
		static constexpr std::string_view namePrefixConnectionGene	= "gk cg ";

		inline static double width;
		inline static double amplitude;
		inline static double amplitudeGlobal;

		inline static double widthMinVal;
		inline static double widthMaxVal;
		inline static double widthStep;

		inline static double ampMinVal;
		inline static double ampMaxVal;
		inline static double ampStep;

		inline static double ampGlobalMinVal;
		inline static double ampGlobalMaxVal;
		inline static double ampGlobalStep;
	};

	struct MexicanHatKernelConstants
	{
		static constexpr std::string_view namePrefix				= "mhk ";
		static constexpr std::string_view namePrefixConnectionGene	= "mhk cg ";

		inline static double widthExc;
		inline static double widthInh;
		inline static double amplitudeExc;
		inline static double amplitudeInh;
		inline static double amplitudeGlobal;

		inline static double widthExcMinVal;
		inline static double widthExcMaxVal;
		inline static double widthExcStep;

		inline static double widthInhMinVal;
		inline static double widthInhMaxVal;
		inline static double widthInhStep;

		inline static double ampExcMinVal;
		inline static double ampExcMaxVal;
		inline static double ampExcStep;

		inline static double ampInhMinVal;
		inline static double ampInhMaxVal;
		inline static double ampInhStep;

		inline static double ampGlobMin;
		inline static double ampGlobMax;
		inline static double ampGlobStep;
	};

	/// @brief Per-mechanism overrides for ablation studies (issue #76). Unlike the
	/// other constant blocks in this file, these are runtime-mutable: a preset in
	/// AblationPresets sets them once before Population::initialize(), and they are
	/// only read (never written) during Population::evolve(), so parallel
	/// evaluation is unaffected. AblationConstants::reset() restores every preset-set
	/// field to its "no ablation" default; the config-loaded reference counts at the
	/// bottom of the struct are not preset state and are deliberately left alone.
	struct AblationConstants
	{
		inline static std::string label;   // appended to solution name -> data/<name>/ folder

		// structural freezes (No Growth IO Only, No Growth Reference Hidden Field Count)
		inline static bool disableAddFieldGene      = false;
		inline static bool disableAddConnectionGene = false;
		inline static bool disableToggleConnectionGene = false;

		// seeding (No Growth IO Only, No Growth Reference Hidden Field Count, Random Initial Topology)
		inline static bool seedAllLegalConnections  = false;   // No Growth IO Only, No Growth Reference Hidden Field Count
		inline static bool seedRandomHiddenFields   = false;   // No Growth Reference Hidden Field Count (min=max=reference min), Random Initial Topology (reference range)
		inline static int  seedHiddenFieldsMin      = 0;
		inline static int  seedHiddenFieldsMax      = 0;
		inline static bool seedRandomConnections    = false;   // Random Initial Topology; count drawn as U[1, maxLegalConnectionCount()]

		// mechanism ablations (No Speciation, No Crossover)
		inline static bool disableSpeciation = false;
		inline static bool disableCrossover  = false;

		// Unlike the flags above -- which default to "no ablation active" and are
		// only set by a preset -- these two are loaded from config/neat_dnfs.json
		// and are experiment-specific: the hidden-field count a task needs to
		// succeed is a property of that task, not of NEAT. Presets refer to them
		// by name rather than hardcoding a number (see config/ablations/).
		inline static int referenceHiddenFieldsMin;
		inline static int referenceHiddenFieldsMax;

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

	/// @brief Weights used to compute the NEAT compatibility distance between two genomes.
	/// The distance determines whether two solutions belong to the same species.
	struct CompatibilityCoefficients
	{
		inline static double compatibilityThreshold;
		inline static double excessGenesCompatibilityWeight;
		inline static double disjointGenesCompatibilityWeight;
		inline static double averageConnectionDifferenceCompatibilityWeight;

		inline static double amplitudeDifferenceCoefficient;
		inline static double widthDifferenceCoefficient;
	};

	/// @brief Probabilities governing structural and parametric genome mutations.
	struct GenomeMutationConstants
	{
		// genome mutation probabilities (the sum does not have to be 1.0)
		// structural mutations
		inline static double toggleConnectionGeneProbability;
		inline static double addFieldGeneProbability;
		inline static double addConnectionGeneProbability;
		// parametrical mutations
		inline static double mutateFieldGenesPerGenomeProbability;
		inline static double mutateConnectionGenesProbability;
		// per gene mutation probabilities
		inline static double mutateFieldGenePerGeneProbability;
		inline static double mutateConnectionGeneProbability;

		inline static bool checkForDuplicateConnectionGenesInGenome;
	};

	struct FieldGeneConstants
	{
		inline static bool variableParameters;

		// (sum must be 1.0)
		inline static double gaussKernelProbability;
		inline static double mexicanHatKernelProbability;

		// field gene mutation probabilities (sum must be 1.0)
		inline static double mutateFieldGeneKernelProbability;
		inline static double mutateFieldGeneKernelTypeProbability;
		inline static double mutateFieldGeneNeuralFieldProbability;
		// field gene gauss kernel mutation probabilities (sum does not have to be 1.0)
		inline static double mutateFieldGeneGaussKernelAmplitudeProbability;
		inline static double mutateFieldGeneGaussKernelWidthProbability;
		inline static double mutateFieldGeneGaussKernelGlobalAmplitudeProbability;
		// field gene mexican hat kernel mutation probabilities (sum does not have to be 1.0)
		inline static double mutateFieldGeneMexicanHatKernelAmplitudeExcProbability;
		inline static double mutateFieldGeneMexicanHatKernelAmplitudeInhProbability;
		inline static double mutateFieldGeneMexicanHatKernelWidthExcProbability;
		inline static double mutateFieldGeneMexicanHatKernelWidthInhProbability;
		inline static double mutateFieldGeneMexicanHatKernelGlobalAmplitudeProbability;

		// field gene neural field mutation probabilities (sum must be 1.0)
		inline static double mutateFieldGeneNeuralFieldParametersProbability;
		inline static double mutateFieldGeneNeuralFieldGenerateRandomParametersProbability;
		// field gene neural field parameters mutation probabilities (sum does not have to be 1.0)
		inline static double mutateFieldGeneNeuralFieldParametersTauProbability;
		inline static double mutateFieldGeneNeuralFieldParametersRestingLevelProbability;
	};

	struct ConnectionGeneConstants
	{
		// connection gene kernel type probability (sum must be 1.0)
		inline static double gaussKernelProbability;
		inline static double mexicanHatKernelProbability;

		// connection gene mutation probabilities (sum must be 1.0)
		inline static double mutateConnectionGeneKernelProbability;
		inline static double mutateConnectionGeneKernelTypeProbability;
		inline static double mutateConnectionGeneConnectionSignalProbability;

		// field gene gauss kernel mutation probabilities (sum does not have to be 1.0)
		inline static double mutateConnectionGeneGaussKernelAmplitudeProbability;
		inline static double mutateConnectionGeneGaussKernelWidthProbability;
		inline static double mutateConnectionGeneGaussKernelGlobalAmplitudeProbability;
		// field gene mexican hat kernel mutation probabilities (sum does not have to be 1.0)
		inline static double mutateConnectionGeneMexicanHatKernelAmplitudeExcProbability;
		inline static double mutateConnectionGeneMexicanHatKernelAmplitudeInhProbability;
		inline static double mutateConnectionGeneMexicanHatKernelWidthExcProbability;
		inline static double mutateConnectionGeneMexicanHatKernelWidthInhProbability;
		inline static double mutateConnectionGeneMexicanHatKernelGlobalAmplitudeProbability;
	};

	struct SolutionConstants
	{
		inline static uint8_t minInitialInputGenes;
		inline static uint8_t minInitialOutputGenes;

		// The reference weighting of a solution's partial-fitness terms. Every
		// task overrides this from its own config/solutions/<slug>.json, since
		// the number of terms differs per task. Each Solution copies the value
		// into an instance member at construction (see Solution::fitnessWeights)
		// so two solution types alive at once cannot read each other's weights.
		inline static std::vector<double> fitnessWeights;

		// The run protocol. These are task properties -- how large a population
		// and how many generations a task needs to solve, and what fitness counts
		// as solved -- so a task overrides them in its own config. An ablation
		// preset may override the first three to standardise the protocol across
		// conditions; targetFitness stays a task property either way. CLI flags
		// (--runs/--pop/--gens/--target) win over all of them.
		inline static int populationSize;
		inline static int numberGenerations;
		inline static int numberRuns;
		inline static double targetFitness;
	};

	struct PopulationConstants
	{
		inline static double pruneRatio;
		inline static int generationsWithoutImprovementThresholdInPopulation;
		inline static int generationsWithoutImprovementThresholdInSpecies;
		inline static bool elitism;
		// Tolerance for the elitism validation check. The DNF simulation is
		// stochastic (see NoiseConstants::amplitude), so re-evaluating the same
		// preserved elite yields a fitness that drifts rather than being
		// bit-identical -- and near a solution's bump-formation boundary, that
		// drift can be much larger than typical jitter (a bump forming or not
		// swings a partial-fitness term by ~0.1). This covers that drift for
		// the common case; a drop beyond it is still fine as long as the
		// previous best solution itself is still present (see validateElitism).
		inline static double elitismFitnessEpsilon;

		inline static bool logSolutions;
		inline static bool logOverview;
		inline static bool logSpecies;

		inline static bool saveOverview;
		inline static bool savePerGenerationOverview;
		inline static bool saveChampions;
		inline static bool saveBestSolutions;
		inline static bool saveSolutions;
		inline static bool saveSpecies;
	};
}