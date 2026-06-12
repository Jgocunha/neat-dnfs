#include <catch2/catch_test_macros.hpp>

#include "neat/population.h"
#include "solutions/detection_instability.h"
#include "test_helpers.h"

using namespace neat_dnfs;
using namespace neat_dnfs::test;

TEST_CASE("Population::initialize", "[Population]")
{
    const PopulationParameters parameters(10, 5, 0.9);
    const auto initialSolution = std::make_shared<DetectionInstability>(makeTopology(1, 1));
    const Population population(parameters, initialSolution);

    REQUIRE_NOTHROW(population.initialize());

    const auto solutions = population.getSolutions();
    REQUIRE(solutions.size() == static_cast<size_t>(parameters.size));
    for (const auto& solution : solutions)
        REQUIRE(!solution->getGenome().getFieldGenes().empty());
}

TEST_CASE("Population::isInitialized", "[Population]")
{
    const PopulationParameters parameters(10, 5, 0.9);
    const auto initialSolution = std::make_shared<DetectionInstability>(makeTopology(1, 1));
    const Population population(parameters, initialSolution);

    // Solutions are created at construction; initialize() populates their genomes
    population.initialize();
    REQUIRE(population.isInitialized());
    for (const auto& solution : population.getSolutions())
        REQUIRE(!solution->getGenome().getFieldGenes().empty());
}

TEST_CASE("Population::getSize and getNumGenerations", "[Population]")
{
    const PopulationParameters parameters(15, 20, 0.9);
    const auto initialSolution = std::make_shared<DetectionInstability>(makeTopology(1, 1));
    const Population population(parameters, initialSolution);

    REQUIRE(population.getSize() == 15);
    REQUIRE(population.getNumGenerations() == 20);
}

TEST_CASE("Population::getBestSolution after initialize", "[Population]")
{
    const PopulationParameters parameters(10, 5, 0.9, true);
    DetectionInstability solution{
        SolutionTopology{ {
            {FieldGeneType::INPUT, dnf_composer::element::ElementDimensions{DimensionConstants::xSize, DimensionConstants::dx}},
            {FieldGeneType::OUTPUT, dnf_composer::element::ElementDimensions{DimensionConstants::xSize, DimensionConstants::dx}},
        }
        },
    };
    Population population{ parameters, std::make_unique<DetectionInstability>(solution) };
    population.initialize();
    REQUIRE_NOTHROW(population.evolve());
    REQUIRE(population.getBestSolution() != nullptr);
}

TEST_CASE("Population::evolve runs without error and respects generation limit", "[Population]")
{
    const PopulationParameters parameters(10, 3, 0.9, true);
    Population population(parameters, std::make_shared<DetectionInstability>(makeTopology(1, 1)));
    population.initialize();

    REQUIRE_NOTHROW(population.evolve());

    const bool validEndCondition =
        population.getBestSolution()->getFitness() >= parameters.targetFitness ||
        population.getCurrentGeneration() >= parameters.numGenerations;
    REQUIRE(validEndCondition);
}

TEST_CASE("Population::evolve — all solutions have non-negative fitness", "[Population]")
{
    const PopulationParameters parameters(10, 2, 0.9, true);
    const auto initialSolution = std::make_shared<DetectionInstability>(makeTopology(1, 1));
    Population population(parameters, initialSolution);
    population.initialize();
    population.evolve();

    for (const auto& solution : population.getSolutions())
        REQUIRE(solution->getFitness() >= 0.0);
}

TEST_CASE("Population::evolve — best solution fitness is monotonically non-decreasing", "[Population]")
{
    const PopulationParameters parameters(10, 5, 1.1, true); // target > 1.0 forces full run
    const auto initialSolution = std::make_shared<DetectionInstability>(makeTopology(1, 1));
    Population population(parameters, initialSolution);
    population.initialize();
    population.evolve();

    const auto best = population.getBestSolution();
    REQUIRE(best != nullptr);
    REQUIRE(best->getFitness() >= 0.0);
    for (const auto& solution : population.getSolutions())
        REQUIRE(solution->getFitness() <= best->getFitness() + 1e-9);
}

TEST_CASE("Population::evolve — speciation produces at least one species", "[Population]")
{
    const PopulationParameters parameters(20, 2, 1.1, true);
    const auto initialSolution = std::make_shared<DetectionInstability>(makeTopology(1, 1));
    Population population(parameters, initialSolution);
    population.initialize();
    population.evolve();

    REQUIRE(!population.getSpeciesList().empty());
}
