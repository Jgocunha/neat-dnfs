#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <thread>

#include "neat_tools/profiler.h"

using namespace neat_dnfs::tools;

TEST_CASE("ScopedTimer accumulates elapsed time per phase when profiling is enabled", "[Profiler]")
{
    profiler::resetGeneration();

    {
        profiler::ScopedTimer timer("test-phase");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

#ifdef NEAT_DNFS_PROFILE
    REQUIRE(profiler::elapsedSeconds("test-phase") > 0.0);

    const auto snapshot = profiler::snapshot();
    const bool hasTestPhase = std::any_of(snapshot.begin(), snapshot.end(),
        [](const auto& entry) { return entry.first == "test-phase"; });
    REQUIRE(hasTestPhase);
#else
    REQUIRE(profiler::elapsedSeconds("test-phase") == 0.0);
    REQUIRE(profiler::snapshot().empty());
#endif
}

TEST_CASE("profiler::resetGeneration clears accumulated buckets", "[Profiler]")
{
    {
        profiler::ScopedTimer timer("phase-to-clear");
    }

    profiler::resetGeneration();

    REQUIRE(profiler::elapsedSeconds("phase-to-clear") == 0.0);
    REQUIRE(profiler::snapshot().empty());
}

TEST_CASE("profiler::elapsedSeconds returns zero for an unknown phase and never throws", "[Profiler]")
{
    profiler::resetGeneration();

    REQUIRE_NOTHROW(profiler::elapsedSeconds("never-timed"));
    REQUIRE(profiler::elapsedSeconds("never-timed") == 0.0);
}
