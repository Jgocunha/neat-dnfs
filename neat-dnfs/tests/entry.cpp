// main() is provided by Catch2::Catch2WithMain (see CMakeLists.txt); this
// translation unit only contributes the global test-environment setup below.
#include <catch2/catch_test_macros.hpp>
#include <dnf_composer/tools/logger.h>

#include "neat_tools/config_loader.h"
#include "neat/population.h"

// dnf_composer's logger writes to shared global output without synchronisation.
// Parallel solution evaluation (PopulationParameters::parallelEvolution) emits many
// INFO logs concurrently from std::async threads, which races and crashes. Production
// examples already raise the level to ERROR; tests do the same so they can exercise the
// parallel path safely. Remove once dnf_composer's logger is made thread-safe.
//
// Population's internal invariant checks (validateUniqueSolutions, validateElitism,
// etc.) only log by default in production. Flipping the default policy to Throw here
// makes every Population constructed inside this test binary raise ValidationError on
// a violation instead of silently logging it, without changing production behaviour.
namespace
{
	struct DnfComposerLogLevelInitializer
	{
		DnfComposerLogLevelInitializer()
		{
			dnf_composer::tools::logger::Logger::setMinLogLevel(
				dnf_composer::tools::logger::LogLevel::ERROR);
			neat_dnfs::Population::setDefaultValidationPolicy(neat_dnfs::ValidationPolicy::Throw);

			// The values in constants.h are loaded from config/neat_dnfs.json at
			// startup and have no compiled-in defaults. Tests never go through
			// apps/'s main(), so without this every test would silently run with
			// xSize == 0 and every probability at 0.0 rather than failing loudly.
			neat_dnfs::ConfigLoader::loadGlobalConfig(
				neat_dnfs::ConfigLoader::defaultGlobalConfigPath());
		}
	};
	const DnfComposerLogLevelInitializer dnfComposerLogLevelInitializer;
}
