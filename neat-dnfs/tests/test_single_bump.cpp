#include <catch2/catch_test_macros.hpp>

#include "neat/population.h"
#include "solutions/single_bump.h"

TEST_CASE("Single Bump Evolutionary Run Elitism", "[SingleBump]")
{
    neat_dnfs::SolutionTopology topology(1, 1);
    const neat_dnfs::PopulationParameters params(100, 1000, 0.95);
    const auto initialSolution = std::make_shared<neat_dnfs::SingleBumpSolution>(topology);

    neat_dnfs::Population population(params, initialSolution);
    population.initialize();

    constexpr static uint16_t numAttempts = 100;

    double prevBestFitness = 0.0;

    population.evolve();

    /*for (uint16_t i = 0; i < numAttempts; ++i)
    {
        population.evaluate();
        population.speciate();
        population.reproduceAndSelect();
        population.upkeep();

        const auto bestSolution = population.getBestSolution();
        REQUIRE(bestSolution != nullptr);
        REQUIRE(bestSolution->getFitness() >= prevBestFitness);

        for (const auto& solution : population.getSolutions())
            REQUIRE(solution->getFitness() <= bestSolution->getFitness());
        prevBestFitness = bestSolution->getFitness();
    }*/
}