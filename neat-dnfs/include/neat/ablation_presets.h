#pragma once

#include <cstddef>
#include <string_view>
#include <vector>

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

		/// @brief Applies the preset named by @p name (kebab-case, e.g. "no-crossover").
		/// @return false if @p name is not a recognized ablation; AblationConstants is
		/// left untouched in that case.
		static bool applyByName(std::string_view name);

		/// @return The kebab-case names accepted by applyByName(), in a stable order.
		static std::vector<std::string_view> names();
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
