#pragma once

namespace neat_dnfs
{
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
		static constexpr double sigma = 2;
		static constexpr double amplitude = 1;
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
	};


}