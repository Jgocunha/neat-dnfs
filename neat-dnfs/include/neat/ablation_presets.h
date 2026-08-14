#pragma once

#include <cstddef>

namespace neat_dnfs
{
	// One place per ablation condition: sets AblationConstants (including the
	// data-folder label) so the executables and the tests exercise the same config.
	struct AblationPresets
	{
		static void noGrowthIOOnly();
		static void noGrowthOneHidden();
		static void noSpeciation();
		static void noCrossover();
		static void randomInitialTopology();
	};

	// Shared run protocol for the baseline and every ablation condition.
	struct AblationProtocol
	{
		static constexpr size_t populationSize    = 500;
		static constexpr size_t numberGenerations = 200;
		static constexpr size_t numberRuns        = 30;
		static constexpr double targetFitness     = 0.9;
	};
}
