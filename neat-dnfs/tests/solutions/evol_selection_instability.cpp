#include <catch2/catch_test_macros.hpp>

#include <cmath>

#include "solutions/selection_instability.h"
#include "test_evolution_helpers.h"

using namespace neat_dnfs;
using namespace neat_dnfs::test;

TEST_CASE("SelectionInstability evolves without crashing, without violating invariants, and reports consistent fitness statistics",
    "[Evolution][Solutions][SelectionInstability]")
{
    std::vector<EvolutionRunResult> results;
    REQUIRE_NOTHROW(results = runEvolution<SelectionInstability>(makeTopology(1, 1)));
    REQUIRE(results.size() == 5);

    for (const auto& r : results)
    {
        CHECK(r.generationsRun > 0);
        CHECK(r.finalPopulationSize == 50);
        CHECK(r.finalBestFitness >= 0.0);
        CHECK(r.finalBestFitness <= 1.0);
        INFO("validation:\n" << join(r.validationMessages, "\n"));
        CHECK(r.validationViolations == 0);

        // Deterministic consistency checks. Nothing here asserts that fitness
        // improved: that is a property of the search on a noisy landscape, not
        // of the code (see the note in test_evolution_helpers.h). These hold
        // for every realization and catch real defects -- generation
        // accounting, best-solution/history desync, NaN leaking into fitness,
        // and a broken early-stop condition.
        REQUIRE_FALSE(r.bestFitnessHistory.empty());
        CHECK(static_cast<int>(r.bestFitnessHistory.size()) == r.generationsRun);
        CHECK(r.finalBestFitness == r.bestFitnessHistory.back());
        for (const double fitness : r.bestFitnessHistory)
        {
            CHECK(std::isfinite(fitness));
            CHECK(fitness >= 0.0);
            CHECK(fitness <= 1.0);
        }
        // evolve() either exhausts its generation budget or stops early
        // because the target was passed -- never anything else.
        CHECK((r.generationsRun == r.numGenerations || r.finalBestFitness > r.targetFitness));
    }
}
