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
		static constexpr std::string namePrefix = "nf ";
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
		static constexpr std::string namePrefix = "gk ";
		static constexpr std::string namePrefixConnectionGene = "gk cg ";
		static constexpr double sigma = 2;
		static constexpr double amplitude = 1;
		static constexpr double initialSigmaMin = 0.0;
		static constexpr double initialSigmaMax = 10.0;
		static constexpr double initialAmplitudeMin = 0.0;
		static constexpr double initialAmplitudeMax = 10.0;
	};

	struct LateralInteractionsConstants
	{
		static constexpr std::string namePrefix = "lik ";
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



}