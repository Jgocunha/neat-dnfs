#pragma once

#include <elements/element_factory.h>

namespace neat_dnfs
{
	typedef std::shared_ptr<dnf_composer::element::NeuralField> NeuralFieldPtr;
	typedef std::shared_ptr<dnf_composer::element::Kernel> KernelPtr;

	struct DimensionConstants
	{
		static constexpr int xSize = 100;
		static constexpr double dx = 1.0;
	};

	struct NeuralFieldConstants
	{
		inline static std::string namePrefix = "nf ";
		static constexpr double tau = 25;
		static constexpr double restingLevel = -10;
		inline static dnf_composer::element::HeavisideFunction activationFunction{ 0.0 };
	};

	struct KernelConstants
	{
		static constexpr bool circularity = false;
		static constexpr bool normalization = false;
	};

	struct GaussKernelConstants
	{
		inline static std::string namePrefix = "gk ";
		inline static std::string namePrefixConnectionGene = "gk cg ";
		static constexpr double sigma = 2;
		static constexpr double amplitude = 1;
		static constexpr double initialSigmaMin = 0.0;
		static constexpr double initialSigmaMax = 10.0;
		static constexpr double initialAmplitudeMin = 0.0;
		static constexpr double initialAmplitudeMax = 10.0;
	};

	struct LateralInteractionsConstants
	{
		inline static std::string namePrefix = "lik ";
		static constexpr double sigmaExc = 5.3;
		static constexpr double sigmaInh = 7.4;
		static constexpr double amplitudeExc = 6;
		static constexpr double amplitudeInh = 6;
		static constexpr double amplitudeGlobal = -0.55;
	};

	struct MutationConstants
	{
		static constexpr double mutationStep = 0.5;
		static constexpr double minSigma = 0.0;
		static constexpr double maxSigma = 10.0;
		static constexpr double minAmplitude = -10.0;
		static constexpr double maxAmplitude = 10.0;

		static constexpr double addGeneProbability = 0.1;
		static constexpr double mutateGeneProbability = 0.2;
		static constexpr double addConnectionGeneProbability = 0.3;
		static constexpr double mutateConnectionGeneProbability = 0.35;
		static constexpr double toggleConnectionGeneProbability = 0.05;
	};

	struct CompatibilityCoefficients
	{
		static constexpr double excessCoefficient = 1.0;
		static constexpr double disjointCoefficient = 1.0;
		static constexpr double amplitudeDifferenceCoefficient = 0.8;
		static constexpr double widthDifferenceCoefficient = 0.2;
	};


	struct SimulationConstants
	{
		inline static std::string name = "simulation";
		static constexpr double deltaT = 25.0;

	};

	struct SolutionConstants
	{
		static constexpr uint8_t minInitialInputGenes = 1;
		static constexpr uint8_t minInitialOutputGenes = 1;
		static constexpr double initialConnectionProbability = 0.0;
	};

	struct SpeciesConstants
	{
		static constexpr double compatibilityThreshold = 3.0;
		static constexpr double excessGenesCompatibilityWeight = 0.5;
		static constexpr double disjointGenesCompatibilityWeight = 0.5;
		static constexpr double averageConnectionDifferenceCompatibilityWeight = 0.5;
	};

	struct PopulationConstants
	{
		static constexpr double killRatio = 0.5;
	};

}