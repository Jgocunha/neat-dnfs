#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace neat_dnfs
{
	// One place per ablation condition: sets AblationConstants (including the
	// data-folder label) so the executables and the tests exercise the same config.
	// The values themselves live in config/ablations/<name>.json, one flat,
	// self-contained file per condition.
	struct AblationPresets
	{
		static void noGrowthIOOnly();
		static void noGrowthReferenceHiddenFieldCount();
		static void noSpeciation();
		static void noCrossover();
		static void randomInitialTopology();

		/// @brief Applies the preset named by @p name (e.g. "no-crossover"),
		/// i.e. loads config/ablations/<name>.json.
		/// @return false if no such preset file exists; AblationConstants is
		/// left untouched in that case.
		/// @throws std::runtime_error if the file exists but is malformed or
		/// omits a key.
		static bool applyByName(std::string_view name);

		/// @return The names accepted by applyByName() -- every
		/// config/ablations/*.json -- sorted, so the order is stable.
		static std::vector<std::string> names();
	};
}
