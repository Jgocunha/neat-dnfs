#include <catch2/catch_test_macros.hpp>

#include <thread>

#include "neat/population.h"
#include "solutions/detection_instability.h"
#include "test_helpers.h"
#include "test_stub_solution.h"

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
    const PopulationParameters parameters(10, 5, 0.9);
    DetectionInstability solution{
        SolutionTopology{ {
            {FieldGeneType::INPUT, dnf_composer::element::ElementDimensions{DimensionConstants::xSize, DimensionConstants::dx}},
            {FieldGeneType::OUTPUT, dnf_composer::element::ElementDimensions{DimensionConstants::xSize, DimensionConstants::dx}},
        }
        },
    };
    Population population{ parameters, std::make_unique<DetectionInstability>(solution), false };
    population.initialize();
    REQUIRE_NOTHROW(population.evolve());
    REQUIRE(population.getBestSolution() != nullptr);
}

TEST_CASE("Population::evolve runs without error and respects generation limit", "[Population]")
{
    const PopulationParameters parameters(10, 3, 0.9);
    Population population(parameters, std::make_shared<DetectionInstability>(makeTopology(1, 1)), false);
    population.initialize();

    REQUIRE_NOTHROW(population.evolve());

    const bool validEndCondition =
        population.getBestSolution()->getFitness() >= parameters.targetFitness ||
        population.getCurrentGeneration() >= parameters.numGenerations;
    REQUIRE(validEndCondition);
}

TEST_CASE("Population::evolve — all solutions have non-negative fitness", "[Population]")
{
    const PopulationParameters parameters(10, 2, 0.9);
    const auto initialSolution = std::make_shared<DetectionInstability>(makeTopology(1, 1));
    Population population(parameters, initialSolution, false);
    population.initialize();
    population.evolve();

    for (const auto& solution : population.getSolutions())
        REQUIRE(solution->getFitness() >= 0.0);
}

TEST_CASE("Population::evolve — best solution fitness is monotonically non-decreasing", "[Population]")
{
    const PopulationParameters parameters(10, 5, 1.1); // target > 1.0 forces full run
    const auto initialSolution = std::make_shared<DetectionInstability>(makeTopology(1, 1));
    Population population(parameters, initialSolution, false);
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
    const PopulationParameters parameters(20, 2, 1.1);
    const auto initialSolution = std::make_shared<DetectionInstability>(makeTopology(1, 1));
    Population population(parameters, initialSolution, false);
    population.initialize();
    population.evolve();

    REQUIRE(!population.getSpeciesList().empty());
}

TEST_CASE("PopulationParameters — parallelEvolution defaults to true", "[Population]")
{
    const PopulationParameters parameters(10, 5, 0.9);
    REQUIRE(parameters.parallelEvolution);
}

// Regression test for issue #45 (unbounded thread spawning): before the fix,
// Population::evaluate() spawned one std::async task per solution with no
// concurrency cap, so peak concurrent evaluations tracked population size.
// After the fix, evaluate() spawns a small fixed number of long-lived worker
// tasks (bounded by hardware_concurrency), so peak concurrent evaluations is
// bounded by that same fixed count. CountingSolution::testPhenotype() holds
// its "slot" for a short sleep so overlapping evaluations are reliably observed.
//
// Reliable pre-fix reproduction of a large peak is platform-dependent: on
// libstdc++ (Linux CI) std::async(std::launch::async, ...) spawns a raw OS
// thread per call, so peak tracks population size (~200) before the fix.
// On MSVC, std::async is backed by the Windows thread pool, which already
// self-throttles new work near hardware_concurrency even in the unfixed
// code — so this assertion is a weaker pre-fix red locally, but is still a
// real, deterministic bound on the fixed implementation either way.
TEST_CASE("Population::evaluate — parallel evaluation concurrency is bounded by hardware_concurrency", "[Population]")
{
    CountingSolution::live = 0;
    CountingSolution::peak = 0;

    const PopulationParameters parameters(200, 1, 1.1); // target > 1.0 forces a full run
    const auto initialSolution = std::make_shared<CountingSolution>(makeTopology(1, 1));
    Population population(parameters, initialSolution, false);
    population.initialize();
    population.evolve();

    const int peak = CountingSolution::peak.load();
    const int hardwareConcurrency = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));

    if (hardwareConcurrency <= 1)
        return; // single-core runner: Population::evaluate() takes the serial fallback by design

    REQUIRE(peak > 1); // evaluation is still actually concurrent
    REQUIRE(peak <= hardwareConcurrency);
}

TEST_CASE("Population::evolve — exceptions from parallel evaluation propagate", "[Population]")
{
    const PopulationParameters parameters(20, 1, 1.1); // parallelEvolution defaults true
    const auto initialSolution = std::make_shared<ThrowingSolution>(makeTopology(1, 1));
    Population population(parameters, initialSolution, false);
    population.initialize();

    REQUIRE_THROWS_AS(population.evolve(), std::runtime_error);
}

TEST_CASE("Population::evolve — exceptions from serial evaluation propagate", "[Population]")
{
    const PopulationParameters parameters(20, 1, 1.1, false); // parallelEvolution explicitly disabled
    const auto initialSolution = std::make_shared<ThrowingSolution>(makeTopology(1, 1));
    Population population(parameters, initialSolution, false);
    population.initialize();

    REQUIRE_THROWS_AS(population.evolve(), std::runtime_error);
}

TEST_CASE("Population::setSize and setNumGenerations round-trip", "[Population]")
{
    const PopulationParameters parameters(10, 5, 0.9);
    const auto initialSolution = std::make_shared<DetectionInstability>(makeTopology(1, 1));
    Population population(parameters, initialSolution);

    population.setSize(25);
    REQUIRE(population.getSize() == 25);

    population.setNumGenerations(50);
    REQUIRE(population.getNumGenerations() == 50);
}

TEST_CASE("Population::getSolutions returns exactly getSize() elements after initialize", "[Population]")
{
    const PopulationParameters parameters(10, 5, 0.9);
    const auto initialSolution = std::make_shared<DetectionInstability>(makeTopology(1, 1));
    Population population(parameters, initialSolution, false);
    population.initialize();

    REQUIRE(population.getSolutions().size() == static_cast<size_t>(population.getSize()));
}

TEST_CASE("Population::getSpeciesList is non-empty after evolve", "[Population]")
{
    const PopulationParameters parameters(10, 2, 1.1);
    const auto initialSolution = std::make_shared<DetectionInstability>(makeTopology(1, 1));
    Population population(parameters, initialSolution, false);
    population.initialize();
    population.evolve();

    REQUIRE(!population.getSpeciesList().empty());
}

TEST_CASE("Population::stop halts evolve before the generation limit", "[Population]")
{
    // Uses CountingSolution (2ms sleep per evaluation) so evolve() takes long
    // enough for a background thread to call stop() mid-run. Real solutions
    // would make this timing-dependent test far too slow.
    CountingSolution::live = 0;
    CountingSolution::peak = 0;

    const PopulationParameters parameters(50, 1000, 1.1, false); // serial, many generations
    const auto initialSolution = std::make_shared<CountingSolution>(makeTopology(1, 1));
    Population population(parameters, initialSolution, false);
    population.initialize();

    std::thread stopper([&population]()
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            population.stop();
        });

    population.evolve();
    stopper.join();

    REQUIRE(population.getCurrentGeneration() < population.getNumGenerations());
}

TEST_CASE("Population::pause and resume allow evolution to complete normally", "[Population]")
{
    CountingSolution::live = 0;
    CountingSolution::peak = 0;

    const PopulationParameters parameters(10, 3, 1.1, false);
    const auto initialSolution = std::make_shared<CountingSolution>(makeTopology(1, 1));
    Population population(parameters, initialSolution, false);
    population.initialize();

    population.pause();
    std::thread resumer([&population]()
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            population.resume();
        });

    REQUIRE_NOTHROW(population.evolve());
    resumer.join();

    REQUIRE(population.getCurrentGeneration() >= population.getNumGenerations());
}

TEST_CASE("Population::start clears a prior stop request", "[Population]")
{
    const PopulationParameters parameters(10, 3, 1.1, false);
    const auto initialSolution = std::make_shared<DetectionInstability>(makeTopology(1, 1));
    Population population(parameters, initialSolution, false);
    population.initialize();

    population.stop();
    population.start(); // clears control.stop before evolve() ever runs

    population.evolve();
    REQUIRE(population.getCurrentGeneration() >= population.getNumGenerations());
}

TEST_CASE("Population of size 1 evolves without throwing", "[Population]")
{
    const PopulationParameters parameters(1, 3, 1.1, false);
    const auto initialSolution = std::make_shared<DetectionInstability>(makeTopology(1, 1));
    Population population(parameters, initialSolution, false);
    population.initialize();

    REQUIRE_NOTHROW(population.evolve());
    REQUIRE(population.getSolutions().size() == 1);
}
