#include <catch2/catch_test_macros.hpp>

#include "solutions/selection_instability.h"
#include "test_evolution_helpers.h"

using namespace neat_dnfs;
using namespace neat_dnfs::test;

TEST_CASE("SelectionInstability evolves without crashing, without violating invariants, and improves fitness",
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
    }

    int improved = 0;
    for (const auto& r : results)
    {
        if (r.finalBestFitness > r.initialBestFitness + 1e-9)
        {
            ++improved;
        }
    }
    REQUIRE(improved >= 3);
}
