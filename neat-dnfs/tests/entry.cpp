#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <dnf_composer/tools/logger.h>

// dnf_composer's logger writes to shared global output without synchronisation.
// Parallel solution evaluation (PopulationParameters::parallelEvolution) emits many
// INFO logs concurrently from std::async threads, which races and crashes. Production
// examples already raise the level to ERROR; tests do the same so they can exercise the
// parallel path safely. Remove once dnf_composer's logger is made thread-safe.
namespace
{
	struct DnfComposerLogLevelInitializer
	{
		DnfComposerLogLevelInitializer()
		{
			dnf_composer::tools::logger::Logger::setMinLogLevel(
				dnf_composer::tools::logger::LogLevel::ERROR);
		}
	};
	const DnfComposerLogLevelInitializer dnfComposerLogLevelInitializer;
}
