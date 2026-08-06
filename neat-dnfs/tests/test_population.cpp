#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <thread>
#include <set>

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

TEST_CASE("Population::evolve - all solutions have non-negative fitness", "[Population]")
{
    const PopulationParameters parameters(10, 2, 0.9);
    const auto initialSolution = std::make_shared<DetectionInstability>(makeTopology(1, 1));
    Population population(parameters, initialSolution, false);
    population.initialize();
    population.evolve();

    for (const auto& solution : population.getSolutions())
        REQUIRE(solution->getFitness() >= 0.0);
}

TEST_CASE("Population::evolve - best solution fitness is monotonically non-decreasing", "[Population]")
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

// Regression test for missing global elitism: bestSolution was recomputed
// from scratch each generation with nothing guaranteeing the previous best
// genome survived reproduction, so the recorded best fitness could and did
// decrease across generations (observed drops up to ~13% relative). A larger
// population with more generations gives speciation more room to fragment,
// which is when the bug was empirically observed to manifest.
TEST_CASE("Population::evolve - best fitness history never decreases across generations", "[Population]")
{
    const PopulationParameters parameters(50, 15, 1.1); // target > 1.0 forces full run
    const auto initialSolution = std::make_shared<DetectionInstability>(makeTopology(1, 1));
    Population population(parameters, initialSolution, false);
    population.initialize();
    population.evolve();

    const auto& history = population.getBestFitnessHistory();
    REQUIRE(history.size() > 1);
    for (size_t i = 1; i < history.size(); ++i)
    {
        INFO("generation " << i << ": " << history[i - 1] << " -> " << history[i]);
        CHECK(history[i] >= history[i - 1] - PopulationConstants::elitismFitnessEpsilon);
    }
}

// Regression test for missing global elitism (see the fitness-history test
// above): once a genome sets the fitness high-water mark, elitism must keep
// it available, so the recorded best never collapses back to a genuinely
// worse lineage.
//
// This deliberately does NOT assert that the genome recorded each generation
// was itself previously recorded as best. The DNF simulation is stochastic
// (NoiseConstants::amplitude), so every solution -- the preserved elite
// included -- re-evaluates to a slightly different fitness each generation
// (measured spread ~3.5e-4 on an unchanged genome). When the elite drifts
// down by a fraction of that noise, another solution can legitimately hold
// the top *measured* spot with a genome never recorded before, which made
// the previous form of this test fail intermittently on nothing but noise.
//
// Tolerating that noise the same way validateElitism() does, via
// elitismFitnessEpsilon, keeps the real regression covered: the bug this
// guards against dropped the best genome outright and cost ~13% fitness,
// orders of magnitude beyond the epsilon.
TEST_CASE("Population::evolve - the recorded best never falls below the established high-water mark", "[Population]")
{
    const PopulationParameters parameters(50, 15, 1.1); // target > 1.0 forces full run
    const auto initialSolution = std::make_shared<DetectionInstability>(makeTopology(1, 1));
    Population population(parameters, initialSolution, false);
    population.initialize();
    population.evolve();

    const auto& fitnessHistory = population.getBestFitnessHistory();
    const auto& idHistory = population.getBestSolutionIdHistory();
    const auto& genomeHistory = population.getBestSolutionGenomeHistory();
    REQUIRE(fitnessHistory.size() > 1);
    REQUIRE(idHistory.size() == fitnessHistory.size());
    REQUIRE(genomeHistory.size() == fitnessHistory.size());

    double highWaterMark = fitnessHistory[0];
    for (size_t i = 1; i < fitnessHistory.size(); ++i)
    {
        INFO("generation " << i << ": high-water " << highWaterMark
            << " -> recorded " << fitnessHistory[i]);
        CHECK(fitnessHistory[i] >= highWaterMark - PopulationConstants::elitismFitnessEpsilon);
        highWaterMark = std::max(highWaterMark, fitnessHistory[i]);
    }
}

TEST_CASE("Population::evolve - speciation produces at least one species", "[Population]")
{
    const PopulationParameters parameters(20, 2, 1.1);
    const auto initialSolution = std::make_shared<DetectionInstability>(makeTopology(1, 1));
    Population population(parameters, initialSolution, false);
    population.initialize();
    population.evolve();

    REQUIRE(!population.getSpeciesList().empty());
}

// Regression test for issue #58 (null-pointer and empty-container hazards in
// Population): calculateAdjustedFitness(), resetGenerationalInnovations(),
// and upkeepPerGenerationStatistics() all run every generation of evolve().
// This pins down that their corrected forms -- an asserted (not dereferenced
// without checking) species lookup, a static Genome::clearGenerationalInnovations()
// call instead of one routed through the possibly-null bestSolution instance
// pointer, and an asserted per-generation statistics pass -- keep producing
// sane, finite statistics with a history entry per generation actually run.
TEST_CASE("Population::evolve - per-generation statistics stay finite and history matches generations run", "[Population]")
{
    const PopulationParameters parameters(30, 10, 1.1); // target > 1.0 forces a full run
    const auto initialSolution = std::make_shared<DetectionInstability>(makeTopology(1, 1));
    Population population(parameters, initialSolution, false);
    population.initialize();

    REQUIRE_NOTHROW(population.evolve());

    const auto& fitnessHistory = population.getBestFitnessHistory();
    REQUIRE(fitnessHistory.size() == static_cast<size_t>(population.getCurrentGeneration()));
    for (const double fitness : fitnessHistory)
        REQUIRE(std::isfinite(fitness));

    REQUIRE(!population.getSpeciesList().empty());
}

TEST_CASE("PopulationParameters - parallelEvolution defaults to true", "[Population]")
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
TEST_CASE("Population::evaluate - parallel evaluation concurrency is bounded by hardware_concurrency", "[Population]")
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

TEST_CASE("Population::evolve - exceptions from parallel evaluation propagate", "[Population]")
{
    const PopulationParameters parameters(20, 1, 1.1); // parallelEvolution defaults true
    const auto initialSolution = std::make_shared<ThrowingSolution>(makeTopology(1, 1));
    Population population(parameters, initialSolution, false);
    population.initialize();

    REQUIRE_THROWS_AS(population.evolve(), std::runtime_error);
}

TEST_CASE("Population::evolve - exceptions from serial evaluation propagate", "[Population]")
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

// Regression test for the null-bestSolution crash reachable through the public
// evolve() path: a Population built with size <= 0 created no solutions, so
// upkeepBestSolution() left bestSolution null and endConditionMet() dereferenced
// it. A population with no individuals is not a degenerate run but a category
// error, so it is rejected at the boundary rather than tolerated at each use site.
TEST_CASE("PopulationParameters rejects non-positive sizes", "[Population]")
{
    REQUIRE_THROWS_AS(PopulationParameters(0, 3, 1.1, false), std::invalid_argument);
    REQUIRE_THROWS_AS(PopulationParameters(-1, 3, 1.1, false), std::invalid_argument);
    REQUIRE_NOTHROW(PopulationParameters(1, 3, 1.1, false));
}

// Regression test for the Release-mode SIGSEGV in parallel Population::evaluate()
// (see .claude/issues/release-parallel-eval-sigsegv.md). Root cause:
// dnf_composer::element::ElementIdentifiers::uniqueIdentifierCounter is a plain
// `static inline int`, incremented unguarded by every element constructor.
// Population::evaluate() constructs elements (NeuralField, kernels, noise) from
// hardware_concurrency() worker threads concurrently, racing on that non-atomic
// increment. Two threads reading the same counter value produce two elements
// with the same uniqueName; Simulation::addElement() silently drops the second
// one, and getElement() later returns nullptr for it, which is dereferenced
// without a null check on the fitness-evaluation path -> SIGSEGV in Release.
// This test constructs ElementIdentifiers concurrently from many threads and
// asserts every id and every generated name is unique.
TEST_CASE("ElementIdentifiers::uniqueIdentifierCounter is unique under concurrent construction", "[Population][thread-safety]")
{
    using dnf_composer::element::ElementIdentifiers;
    using dnf_composer::element::ElementLabel;

    const unsigned hardwareConcurrency = std::max(4U, std::thread::hardware_concurrency());
    constexpr int perThread = 500;

    std::vector<std::vector<int>> idsByThread(hardwareConcurrency);
    std::vector<std::vector<std::string>> namesByThread(hardwareConcurrency);
    std::vector<std::thread> threads;
    threads.reserve(hardwareConcurrency);

    for (unsigned t = 0; t < hardwareConcurrency; ++t)
    {
        idsByThread[t].reserve(perThread);
        namesByThread[t].reserve(perThread);
        threads.emplace_back([t, &idsByThread, &namesByThread]()
            {
                for (int i = 0; i < perThread; ++i)
                {
                    const ElementIdentifiers identifiers(ElementLabel::NEURAL_FIELD);
                    idsByThread[t].push_back(identifiers.uniqueIdentifier);
                    namesByThread[t].push_back(identifiers.uniqueName);
                }
            });
    }
    for (auto& thread : threads)
        thread.join();

    std::set<int> allIds;
    std::set<std::string> allNames;
    for (unsigned t = 0; t < hardwareConcurrency; ++t)
    {
        allIds.insert(idsByThread[t].begin(), idsByThread[t].end());
        allNames.insert(namesByThread[t].begin(), namesByThread[t].end());
    }

    const size_t expectedCount = static_cast<size_t>(hardwareConcurrency) * perThread;
    REQUIRE(allIds.size() == expectedCount);
    REQUIRE(allNames.size() == expectedCount);
}

// End-to-end regression test: runs a real Population::evolve() with parallel
// evaluation (the default) using a real DNF-simulating solution. Runs several
// times because the underlying races this guards against are non-deterministic.
TEST_CASE("Population::evolve with parallel evaluation and a real solution does not crash", "[Population][thread-safety]")
{
    for (int attempt = 0; attempt < 5; ++attempt)
    {
        const PopulationParameters parameters(10, 3, 0.9); // parallelEvolution defaults to true
        const auto initialSolution = std::make_shared<DetectionInstability>(makeTopology(1, 1));
        Population population(parameters, initialSolution, false);
        population.initialize();

        REQUIRE_NOTHROW(population.evolve());
        REQUIRE(population.getBestSolution() != nullptr);
    }
}
