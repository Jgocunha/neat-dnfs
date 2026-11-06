#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <ranges>
#include <set>
#include <sstream>
#include <utility>

#include <dnf_composer/exceptions/exception.h>
#include <dnf_composer/simulation/simulation_file_manager.h>

#include "solutions/memory_instability.h"
#include "solutions/selection_instability.h"
#include "solutions/memory_trace.h"
#include "solutions/delayed_match_to_sample.h"
#include "solutions/inhibition_of_return.h"
#include "solutions/and.h"
#include "solutions/xor.h"
#include "solutions/hri_packaging_task.h"
#include "solutions/incremental_recall.h"
#include "solutions/sequence_and_timing_recall.h"
#include "solutions/sequence_timing_behaviour.h"
#include "solutions/serial_order_behaviour.h"
#include "solutions/serial_order_schedule.h"
#include "neat_tools/config_loader.h"
#include "neat_tools/resource_paths.h"
#include "neat_tools/solution_registry.h"
#include "test_helpers.h"

using namespace neat_dnfs;
using namespace neat_dnfs::test;

// Fast contract checks shared by every concrete task Solution: construction,
// initialize(), clone()/copy() independence, and mutate() robustness. These
// require no DNF simulation. Templated locally (test-only) to avoid repeating
// the same six assertions per class; not shared production code.
template <typename SolutionType>
static void checkSolutionContract(const SolutionTopology& topology)
{
    REQUIRE_NOTHROW(SolutionType(topology));

    SolutionType solution(topology);
    solution.initialize();
    REQUIRE(solution.getGenome().getFieldGenes().size() == topology.geneTopology.size());

    const auto cloned = solution.clone();
    REQUIRE(cloned->getId() != solution.getId());
    REQUIRE(cloned->hasTheSameTopology(std::make_shared<SolutionType>(solution)));

    // Independence: mutating the clone must not affect the original's genome,
    // i.e. clone() does not share mutable state with the source solution.
    const size_t originalSizeBeforeCloneMutation = solution.getNumConnectionGenes();
    for (int i = 0; i < 20; ++i)
        cloned->mutate();
    REQUIRE(solution.getNumConnectionGenes() == originalSizeBeforeCloneMutation);

    solution.buildPhenotype();
    const auto copied = solution.copy();
    REQUIRE(copied->getId() != solution.getId());
    // copy()'s (topology, phenotype) ctor rebuilds the genome via
    // translatePhenotypeToGenome() rather than copying it directly, so the
    // guarantee is topology + gene count, not byte-for-byte genome equality.
    REQUIRE(copied->hasTheSameTopology(std::make_shared<SolutionType>(solution)));
    REQUIRE(copied->getNumConnectionGenes() == solution.getNumConnectionGenes());

    // Independence: mutating the copy must not affect the original's genome.
    const size_t originalSizeBeforeCopyMutation = solution.getNumConnectionGenes();
    for (int i = 0; i < 20; ++i)
        copied->mutate();
    REQUIRE(solution.getNumConnectionGenes() == originalSizeBeforeCopyMutation);

    size_t previousNumConnectionGenes = solution.getNumConnectionGenes();
    for (int i = 0; i < 100; ++i)
    {
        solution.mutate();
        REQUIRE(solution.getNumConnectionGenes() >= previousNumConnectionGenes);
        previousNumConnectionGenes = solution.getNumConnectionGenes();
    }
}

TEST_CASE("MemoryInstability contract", "[Solutions][MemoryInstability]")
{
    resetGlobalState();
    checkSolutionContract<MemoryInstability>(makeTopology(1, 1));
}

TEST_CASE("MemoryInstability evaluate produces a bounded fitness", "[Solutions][MemoryInstability]")
{
    resetGlobalState();
    MemoryInstability solution(makeTopology(1, 1));
    solution.initialize();

    REQUIRE_NOTHROW(solution.evaluate());
    REQUIRE(solution.getFitness() >= 0.0);
    REQUIRE(solution.getFitness() <= 1.0);
    REQUIRE(solution.getParameters().partialFitness.size() == 4);
}

TEST_CASE("SelectionInstability contract", "[Solutions][SelectionInstability]")
{
    resetGlobalState();
    checkSolutionContract<SelectionInstability>(makeTopology(1, 1));
}

TEST_CASE("SelectionInstability evaluate produces a bounded fitness", "[Solutions][SelectionInstability]")
{
    resetGlobalState();
    SelectionInstability solution(makeTopology(1, 1));
    solution.initialize();

    REQUIRE_NOTHROW(solution.evaluate());
    REQUIRE(solution.getFitness() >= 0.0);
    REQUIRE(solution.getFitness() <= 1.0);
    REQUIRE(solution.getParameters().partialFitness.size() == 4);
}

TEST_CASE("MemoryTrace contract", "[Solutions][MemoryTrace]")
{
    resetGlobalState();
    checkSolutionContract<MemoryTrace>(makeTopology(2, 1));
}

TEST_CASE("MemoryTrace evaluate produces a bounded fitness", "[Solutions][MemoryTrace]")
{
    resetGlobalState();
    MemoryTrace solution(makeTopology(2, 1)); // nf1, nf2 inputs; nf3 output
    solution.initialize();

    REQUIRE_NOTHROW(solution.evaluate());
    REQUIRE(solution.getFitness() >= 0.0);
    REQUIRE(solution.getFitness() <= 1.0);
    REQUIRE(solution.getParameters().partialFitness.size() == 8);
}

TEST_CASE("DelayedMatchToSample contract", "[Solutions][DelayedMatchToSample]")
{
    resetGlobalState();
    checkSolutionContract<DelayedMatchToSample>(makeTopology(1, 1));
}

// DelayedMatchToSample is tuned for a 360-wide field (its stimuli represent
// hue): testPhenotype() places the second sample at position 100
// (src/solutions/delayed_match_to_sample.cpp), outside the [0, size) range of
// the global default xSize=100. config/solutions/dmts.json overrides xSize to
// 360 for this task; ScopedTaskConfig applies that override for the scope of
// this test the same way tests/entry.cpp's global config never does on its own.
//
// makeTopology() hardcodes dimensions at 100 (right for every other task in
// this file), so it cannot be used here -- the topology has to be built at
// whatever xSize the ScopedTaskConfig override just set, read live rather
// than baked in, or the field genes end up 100-wide while the stimuli below
// (which do read the live global) are built at 360 and never attach.
TEST_CASE("DelayedMatchToSample evaluate produces a bounded fitness", "[Solutions][DelayedMatchToSample]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "dmts" };

    using dnf_composer::element::ElementDimensions;
    const ElementDimensions dims{ DimensionConstants::xSize, DimensionConstants::dx };
    const SolutionTopology topology({
        { FieldGeneType::INPUT, dims },
        { FieldGeneType::OUTPUT, dims }
    });

    DelayedMatchToSample solution(topology);
    solution.initialize();

    REQUIRE_NOTHROW(solution.evaluate());
    REQUIRE(solution.getFitness() >= 0.0);
    REQUIRE(solution.getFitness() <= 1.0);
    REQUIRE(solution.getParameters().partialFitness.size() == 6);
}

TEST_CASE("InhibitionOfReturn contract", "[Solutions][InhibitionOfReturn]")
{
    resetGlobalState();
    checkSolutionContract<InhibitionOfReturn>(makeTopology(1, 1));
}

TEST_CASE("InhibitionOfReturn evaluate produces a bounded fitness", "[Solutions][InhibitionOfReturn]")
{
    resetGlobalState();
    InhibitionOfReturn solution(makeTopology(1, 1));
    solution.initialize();

    REQUIRE_NOTHROW(solution.evaluate());
    REQUIRE(solution.getFitness() >= 0.0);
    REQUIRE(solution.getFitness() <= 1.0);
    REQUIRE(solution.getParameters().partialFitness.size() == 5);
}

TEST_CASE("AND contract", "[Solutions][AND]")
{
    resetGlobalState();
    checkSolutionContract<AND>(makeTopology(2, 1)); // nf1, nf2 inputs; nf3 output
}

TEST_CASE("AND evaluate produces a bounded fitness", "[Solutions][AND]")
{
    resetGlobalState();
    AND solution(makeTopology(2, 1));
    solution.initialize();

    REQUIRE_NOTHROW(solution.evaluate());
    REQUIRE(solution.getFitness() >= 0.0);
    REQUIRE(solution.getFitness() <= 1.0);
    REQUIRE(solution.getParameters().partialFitness.size() == 8);
}

TEST_CASE("XOR contract", "[Solutions][XOR]")
{
    resetGlobalState();
    checkSolutionContract<XOR>(makeTopology(2, 1)); // nf1, nf2 inputs; nf3 output
}

TEST_CASE("XOR evaluate produces a bounded fitness", "[Solutions][XOR]")
{
    resetGlobalState();
    XOR solution(makeTopology(2, 1));
    solution.initialize();

    REQUIRE_NOTHROW(solution.evaluate());
    REQUIRE(solution.getFitness() >= 0.0);
    REQUIRE(solution.getFitness() <= 1.0);
    REQUIRE(solution.getParameters().partialFitness.size() == 4);
}

TEST_CASE("HRIPackagingTask contract", "[Solutions][HRIPackagingTask]")
{
    resetGlobalState();
    checkSolutionContract<HRIPackagingTask>(makeTopology(3, 1)); // nf1, nf2, nf3 inputs; nf4 output
}

TEST_CASE("HRIPackagingTask evaluate produces a bounded fitness", "[Solutions][HRIPackagingTask]")
{
    resetGlobalState();
    HRIPackagingTask solution(makeTopology(3, 1));
    solution.initialize();

    REQUIRE_NOTHROW(solution.evaluate());
    REQUIRE(solution.getFitness() >= 0.0);
    REQUIRE(solution.getFitness() <= 1.0);
    REQUIRE(solution.getParameters().partialFitness.size() == 7);
}

TEST_CASE("HRIPackagingTask is registered with three inputs and one output", "[SolutionRegistry]")
{
    const auto* task = findTask("hri-packaging");
    REQUIRE(task != nullptr);
    REQUIRE(task->templateFile == "hri-packaging-task.dnf");
    REQUIRE(task->inputs == 3);
    REQUIRE(task->outputs == 1);
}

// Like dmts, the serial-order and sequence-timing tasks place stimuli beyond the global default xSize=100
// (events at 120 and 180), so the topology is built at the live xSize that
// their configs set rather than via makeTopology().
static SolutionTopology sequenceTimingTopology()
{
    using dnf_composer::element::ElementDimensions;
    const ElementDimensions dims{ DimensionConstants::xSize, DimensionConstants::dx };
    return SolutionTopology({
        { FieldGeneType::INPUT, dims },
        { FieldGeneType::OUTPUT, dims }
    });
}

// Loads a saved phenotype from tests/fixtures. Keep it alive while a Solution decodes a copy of it: a copied
// Simulation's elements still point at the original's, which its destructor disconnects.
static std::shared_ptr<dnf_composer::Simulation> loadFixtureSimulation(const std::string& fileName)
{
    const auto simulation = std::make_shared<dnf_composer::Simulation>();
    const dnf_composer::SimulationFileManager sfm(simulation,
        (std::filesystem::path(PROJECT_DIR) / "tests" / "fixtures" / fileName).generic_string());
    sfm.loadElementsFromJson();
    return simulation;
}

// Loads a saved solution for a serial-order or sequence-timing task from tests/fixtures.
template <typename Task>
static Task loadSequenceTimingFixture(const std::string& fileName)
{
    const auto simulation = loadFixtureSimulation(fileName);
    return Task(sequenceTimingTopology(), *simulation);
}

// The tasks of the serial-order experiment, intended behaviour alone and with guides.
static const std::array<std::string, 2> serialOrderTasks{ "serial-order-behaviour", "serial-order-behaviour-guided" };

TEST_CASE("SerialOrderBehaviour contract", "[Solutions][SerialOrderBehaviour]")
{
    resetGlobalState();
    checkSolutionContract<SerialOrderBehaviour>(makeTopology(1, 1));
}

TEST_CASE("SerialOrderBehaviourGuided contract", "[Solutions][SerialOrderBehaviour]")
{
    resetGlobalState();
    checkSolutionContract<SerialOrderBehaviourGuided>(makeTopology(1, 1));
}

// Runs are filed under the solution's name, and a population is built from clone() and copy(), so a copy that kept the
// other variant's name or type would evolve and save as the other task.
TEST_CASE("SerialOrderBehaviour and its guided variant name their runs apart and clone and copy as themselves", "[Solutions][SerialOrderBehaviour]")
{
    resetGlobalState();

    SerialOrderBehaviour behaviour(sequenceTimingTopology());
    SerialOrderBehaviourGuided guided(sequenceTimingTopology());
    for (Solution* solution : std::initializer_list<Solution*>{ &behaviour, &guided })
    {
        solution->initialize();
        solution->buildPhenotype();
    }

    CHECK(behaviour.getName() == "Serial Order Behaviour");
    CHECK(guided.getName() == "Serial Order Behaviour Guided");
    CHECK(behaviour.clone()->getName() == behaviour.getName());
    CHECK(behaviour.copy()->getName() == behaviour.getName());
    CHECK(std::dynamic_pointer_cast<SerialOrderBehaviourGuided>(guided.clone()) != nullptr);
    CHECK(std::dynamic_pointer_cast<SerialOrderBehaviourGuided>(guided.copy()) != nullptr);
}

// The reference is the design whose responses the fitness was written from: the hand-built serial-order design of
// 18 October 2026, with the competition in the output.
TEST_CASE("SerialOrderBehaviourGuided scores the reference design above 0.9 on every term", "[Solutions][SerialOrderBehaviour]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order-behaviour-guided" };
    const auto simulation = loadFixtureSimulation("serial-order-fitness-test-reference.dnf");
    SerialOrderBehaviourGuided solution(sequenceTimingTopology(), *simulation);

    for (int evaluation = 0; evaluation < 3; ++evaluation)
    {
        solution.evaluate();
        const auto& partials = solution.getParameters().partialFitness;
        REQUIRE(partials.size() == 7);
        for (size_t term = 0; term < partials.size(); ++term)
        {
            INFO("evaluation " << evaluation << ", f" << term + 1);
            CHECK(partials[term] > 0.9);
        }
    }
}

// The reference with Gauss self-kernels (amplitude 10, width 1.25, no global inhibition) in place of the Mexican hats of
// nf 3 and the done field nf 7: the same design within a Gauss-only search space. Gauss amplitude 9-13, width 1-1.75
// and global 0 to -0.05 all score 0.95-0.96; a Gauss kernel shaped like the Mexican hat (amplitude 20, width 2.5) does
// not, 0.30-0.38.
TEST_CASE("SerialOrderBehaviourGuided scores the Gauss-only reference design above 0.9 on every term", "[Solutions][SerialOrderBehaviour]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order-behaviour-guided" };
    auto solution = loadSequenceTimingFixture<SerialOrderBehaviourGuided>("serial-order-fitness-test-reference-gauss.dnf");

    for (int evaluation = 0; evaluation < 3; ++evaluation)
    {
        solution.evaluate();
        const auto& partials = solution.getParameters().partialFitness;
        REQUIRE(partials.size() == 7);
        for (size_t term = 0; term < partials.size(); ++term)
        {
            INFO("evaluation " << evaluation << ", f" << term + 1);
            CHECK(partials[term] > 0.9);
        }
    }
}

// Recall, release and stop are scored as in the guided task; only what lies below threshold no longer counts.
TEST_CASE("SerialOrderBehaviour scores the Gauss-only reference design above 0.9 on each of its five terms", "[Solutions][SerialOrderBehaviour]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order-behaviour" };
    auto solution = loadSequenceTimingFixture<SerialOrderBehaviour>("serial-order-fitness-test-reference-gauss.dnf");

    solution.evaluate();

    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 5);
    for (size_t term = 0; term < partials.size(); ++term)
    {
        INFO("f" << term + 1);
        CHECK(partials[term] > 0.9);
    }
}

// From-scratch run h3 of 20 October 2026 stores D, C and A in order as pre-shapes, then from the first X sinks D and C
// into deep troughs without either ever peaking. The guided task pays the pre-shapes; this one pays only what the output
// does above threshold, which is nothing, so it scores no more than an output that never responds.
TEST_CASE("SerialOrderBehaviour gives pre-shapes and troughs without recall no more than an output that never responds", "[Solutions][SerialOrderBehaviour]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order-behaviour" };
    auto troughs = loadSequenceTimingFixture<SerialOrderBehaviour>("serial-order-fitness-test-troughs-without-recall.dnf");
    auto silent = loadSequenceTimingFixture<SerialOrderBehaviour>("serial-order-profile-silent.dnf");

    troughs.evaluate();
    silent.evaluate();

    const auto& partials = troughs.getParameters().partialFitness;
    REQUIRE(partials.size() == 5);
    CHECK(partials[2] == 0.0); // nothing was recalled
    CHECK(partials[3] < 0.05);
    CHECK(partials[4] < 0.05);
    CHECK(troughs.getFitness() <= silent.getFitness() + 0.005);
}

// The reference design with its done field's inhibition of the output weakened from -30 to -12: each recalled event
// returns to about the rest of the field rather than below it. The guided task pays half of a release for lying below
// the field; this one asks only that the event lie below threshold, so the same release earns more here.
TEST_CASE("SerialOrderBehaviour asks a released event to lie below threshold, not below the field", "[Solutions][SerialOrderBehaviour]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order-behaviour" };
    auto weakRelease = loadSequenceTimingFixture<SerialOrderBehaviour>("serial-order-fitness-test-weak-release.dnf");
    auto weakReleaseGuided = loadSequenceTimingFixture<SerialOrderBehaviourGuided>("serial-order-fitness-test-weak-release.dnf");

    weakRelease.evaluate();
    weakReleaseGuided.evaluate();

    REQUIRE(weakRelease.getParameters().partialFitness.size() == 5);
    CHECK(weakRelease.getParameters().partialFitness[3] > weakReleaseGuided.getParameters().partialFitness[5] + 0.1);
}

// From-scratch run B01 of 23 October 2026 has no memory. On each cue its whole output flashes above threshold except
// around the event shown last, B. dnf-composer joins a stretch above threshold at both ends of the field into one bump
// centred opposite the gap, so the flash reads as a bump about 185 wide centred on D, and earned D's recall. A recalled
// peak is now scored by its width too, so the flash earns nothing, and the design falls below one that honestly recalls
// D alone (L2).
TEST_CASE("Both serial-order behaviour tasks give a field-wide flash no recall credit", "[Solutions][SerialOrderBehaviour]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order-behaviour" };
    auto flash = loadSequenceTimingFixture<SerialOrderBehaviour>("serial-order-behaviour-wrap-around-flash.dnf");
    auto flashGuided = loadSequenceTimingFixture<SerialOrderBehaviourGuided>("serial-order-behaviour-wrap-around-flash.dnf");
    auto dAlone = loadSequenceTimingFixture<SerialOrderBehaviour>("serial-order-profile-d-only.dnf");

    flash.evaluate();
    flashGuided.evaluate();
    dAlone.evaluate();

    REQUIRE(flash.getParameters().partialFitness.size() == 5);
    REQUIRE(flashGuided.getParameters().partialFitness.size() == 7);
    CHECK(flash.getParameters().partialFitness[2] < 0.05);
    CHECK(flashGuided.getParameters().partialFitness[3] < 0.05);
    CHECK(flash.getFitness() < dAlone.getFitness());
}

// From-scratch run B07 of 24 October 2026 remembers only D. On each cue its output forms one peak at D that then travels
// around the field, about a third of a position per unit of time, and passes the other events in turn. Each passage
// counted as that event recalled, which paid D then C. A recalled peak is now scored by how far it stays in place, so
// the travelling bump earns no more than D's share of recall.
TEST_CASE("Both serial-order behaviour tasks give a travelling bump recall credit for no more than D", "[Solutions][SerialOrderBehaviour]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order-behaviour" };
    auto travelling = loadSequenceTimingFixture<SerialOrderBehaviour>("serial-order-behaviour-travelling-bump.dnf");
    auto travellingGuided = loadSequenceTimingFixture<SerialOrderBehaviourGuided>("serial-order-behaviour-travelling-bump.dnf");

    travelling.evaluate();
    travellingGuided.evaluate();

    REQUIRE(travelling.getParameters().partialFitness.size() == 5);
    REQUIRE(travellingGuided.getParameters().partialFitness.size() == 7);
    CHECK(travelling.getParameters().partialFitness[2] < 0.3);
    CHECK(travellingGuided.getParameters().partialFitness[3] < 0.3);
}

// From-scratch run B06 of 23 October 2026 recalls D, C, A and B in order on every cue, then peaks at D again before the
// next cue, at times that vary: up to 500 after X inside the recall window, and once about 720 after the third X. The
// output after recall was read only at 600 after X and just before the next X, so those peaks cost nothing: that term
// equalled its recall, 0.95. It is now read at every sample from the moment every event has been released until the
// next X, and B06 has a peak at about a tenth of them.
TEST_CASE("SerialOrderBehaviour charges a peak at any moment between the end of the sequence and the next cue", "[Solutions][SerialOrderBehaviour]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order-behaviour" };
    auto peaksAgain = loadSequenceTimingFixture<SerialOrderBehaviour>("serial-order-behaviour-peaks-again-after-recall.dnf");

    peaksAgain.evaluate();

    const auto& partials = peaksAgain.getParameters().partialFitness;
    REQUIRE(partials.size() == 5);
    CHECK(partials[2] > 0.9);                // the sequence itself is recalled in order
    CHECK(partials[4] < partials[2] - 0.05); // and the output peaks again after it
}

// From-scratch run BG05 of 23 October 2026 recalls D, C, A and B in order on every cue, one stationary peak 4-9 wide at
// a time, and releases each through a field the cue resets. Scoring a recalled peak's width and how far it stays in
// place leaves its recall credit as it was, 0.92-0.93.
TEST_CASE("Both serial-order behaviour tasks leave an evolved recall in order its recall credit", "[Solutions][SerialOrderBehaviour]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order-behaviour" };
    auto inOrder = loadSequenceTimingFixture<SerialOrderBehaviour>("serial-order-behaviour-recalls-with-done-field.dnf");
    auto inOrderGuided = loadSequenceTimingFixture<SerialOrderBehaviourGuided>("serial-order-behaviour-recalls-with-done-field.dnf");

    inOrder.evaluate();
    inOrderGuided.evaluate();

    REQUIRE(inOrder.getParameters().partialFitness.size() == 5);
    REQUIRE(inOrderGuided.getParameters().partialFitness.size() == 7);
    CHECK(inOrder.getParameters().partialFitness[2] > 0.88);
    CHECK(inOrderGuided.getParameters().partialFitness[3] > 0.88);
}

// A saved solution carries what createPhenotypeEnvironment() adds, and the evaluation runs on that same schedule, so a
// saved design opened in dnf-composer replays what was scored.
TEST_CASE("SerialOrderBehaviourGuided saves its schedule with each solution, so a saved design replays its evaluation", "[Solutions][SerialOrderBehaviour]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order-behaviour-guided" };
    const auto fixture = loadFixtureSimulation("serial-order-fitness-test-reference.dnf");
    SerialOrderBehaviourGuided solution(sequenceTimingTopology(), *fixture);
    const auto saveDirectory = std::filesystem::temp_directory_path() / "neat-dnfs-serial-order-behaviour-replay";

    // As PopulationFileManager saves a solution: into <directory>/<identifier>/<identifier>.dnf.
    Solution& saved = solution;
    saved.buildPhenotype();
    saved.createPhenotypeEnvironment();
    std::filesystem::path savedFile;
    {
        const auto simulation = std::make_shared<dnf_composer::Simulation>(saved.getPhenotype());
        dnf_composer::SimulationFileManager(simulation, saveDirectory.generic_string()).saveElementsToJson();
        const std::string identifier = simulation->getUniqueIdentifier();
        savedFile = saveDirectory / identifier / (identifier + ".dnf");
    }
    saved.clearPhenotype();
    REQUIRE(std::filesystem::exists(savedFile));

    const auto replay = std::make_shared<dnf_composer::Simulation>();
    dnf_composer::SimulationFileManager(replay, savedFile.generic_string()).loadElementsFromJson();
    replay->init();
    const auto output = std::dynamic_pointer_cast<dnf_composer::element::NeuralField>(replay->getElement("nf 2"));
    REQUIRE(output != nullptr);
    // When the evaluation scores each recalled peak of the first cycle.
    for (const auto& [time, position] : std::vector<std::pair<double, double>>{ { 2700.0, 180.0 }, { 2840.0, 120.0 },
        { 2950.0, 20.0 }, { 3050.0, 80.0 } })
    {
        while (replay->getT() < time - 1e-9)
        {
            replay->step();
        }
        const auto bumps = output->getBumps();
        INFO("t " << time);
        REQUIRE(bumps.size() == 1);
        CHECK(std::abs(bumps.front().centroid - position) <= 1.0);
    }
    std::filesystem::remove_all(saveDirectory);
}

// Each evaluation records what the input and output looked like at every point it was scored, so a run's best
// solution can be read from its overview without re-simulating it.
TEST_CASE("SerialOrderBehaviourGuided traces the input and output at every point it scores them", "[Solutions][SerialOrderBehaviour]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order-behaviour-guided" };
    auto solution = loadSequenceTimingFixture<SerialOrderBehaviourGuided>("serial-order-fitness-test-reference.dnf");

    solution.evaluate();

    const auto trace = nlohmann::json::parse(solution.getParameters().evaluationTrace);
    const auto& snapshots = trace.at("snapshots");
    REQUIRE(snapshots.size() == 27); // 9 in the demonstration, 6 in each of three recall cycles
    const auto snapshotAt = [&snapshots](const std::string& stage) {
        const auto found = std::ranges::find_if(snapshots, [&stage](const auto& s) { return s.at("stage") == stage; });
        REQUIRE(found != snapshots.end());
        return *found;
    };

    const auto shown = snapshotAt("C shown");
    CHECK(shown.at("t").get<double>() == Catch::Approx(650.0));
    const auto& inputBumps = shown.at("input").at("bumps");
    REQUIRE(inputBumps.size() == 1);
    CHECK(std::abs(inputBumps[0][0].get<double>() - 120.0) <= 1.0);

    const auto stored = snapshotAt("after B").at("output");
    CHECK(stored.at("bumps").empty());
    const auto& atEvents = stored.at("atEvents");
    CHECK(atEvents.at("D").get<double>() > atEvents.at("C").get<double>());
    CHECK(atEvents.at("C").get<double>() > atEvents.at("A").get<double>());
    CHECK(atEvents.at("A").get<double>() > atEvents.at("B").get<double>());
    CHECK(atEvents.at("B").get<double>() > stored.at("median").get<double>());

    for (const auto& [stage, position] : std::vector<std::pair<std::string, double>>{ { "recall 1 D", 180.0 },
        { "recall 1 C", 120.0 }, { "recall 1 A", 20.0 }, { "recall 1 B", 80.0 } })
    {
        INFO(stage);
        const auto recalled = snapshotAt(stage).at("output").at("bumps");
        REQUIRE(recalled.size() == 1);
        CHECK(std::abs(recalled[0][0].get<double>() - position) <= 1.0);
    }

    // Each cycle's recall summary: every event peaked, first D, then C, A and B, each in its place, and each was released.
    const auto& cycles = trace.at("recall");
    REQUIRE(cycles.size() == 3);
    for (const auto& cycle : cycles)
    {
        INFO("cycle " << cycle.at("cycle"));
        const auto& events = cycle.at("events");
        double previousPeak = 0.0;
        for (const std::string label : { "D", "C", "A", "B" })
        {
            INFO(label);
            REQUIRE(!events.at(label).at("firstPeak").is_null());
            CHECK(events.at(label).at("firstPeak").get<double>() > previousPeak);
            previousPeak = events.at(label).at("firstPeak").get<double>();
            CHECK(events.at(label).at("inPlace").get<double>() == 1.0);
            CHECK(!events.at(label).at("release").is_null());
        }
    }
}

// Two weighted from-scratch runs of 20 October 2026 converged on one output peak that rises before any X and is never
// let go: w1's at 138, near no event, and w2's at C. Under the old rule, one bump of the right size earned more than
// half of a recall point wherever it was: f4 0.53 and 0.63.
// - w1's earns nothing: only a peak near an event counts.
// - w2's earns nothing: recall is scored by order, and C counts only after D.
TEST_CASE("SerialOrderBehaviourGuided gives a peak held through recall no credit off an event or out of order", "[Solutions][SerialOrderBehaviour]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order-behaviour-guided" };

    auto offEvent = loadSequenceTimingFixture<SerialOrderBehaviourGuided>("serial-order-fitness-test-held-peak-off-event.dnf");
    offEvent.evaluate();
    CHECK(offEvent.getParameters().partialFitness[3] < 0.05);

    auto atC = loadSequenceTimingFixture<SerialOrderBehaviourGuided>("serial-order-fitness-test-held-peak-before-cue.dnf");
    atC.evaluate();
    CHECK(atC.getParameters().partialFitness[3] < 0.05);
}

// The reference design with its done field's inhibition of the output (gk cg 7-2) weakened from -30 to -12. It recalls
// D, C, A and B in order every cycle, each alone, but each peak comes about 100 later, and a recalled event returns to
// about the rest of the field rather than below it. Scored at the reference's own recall times, it earned f4 0.40-0.47
// and f6 0.04. Scored by order, it earns most of the recall credit and part of the release.
TEST_CASE("SerialOrderBehaviourGuided pays recall in order whatever its pace, and a shallow release in part", "[Solutions][SerialOrderBehaviour]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order-behaviour-guided" };

    auto weakRelease = loadSequenceTimingFixture<SerialOrderBehaviourGuided>("serial-order-fitness-test-weak-release.dnf");
    weakRelease.evaluate();

    const auto& partials = weakRelease.getParameters().partialFitness;
    CHECK(partials[3] > 0.8);
    CHECK(partials[5] > 0.4);
    CHECK(partials[5] < 0.9); // the reference's release, below the field, still earns more
}

// From-scratch run t1 of 20 October 2026: after X, D and C rise together and are held. Rising together is not
// recalling in order, so only D, the first, earns recall credit.
TEST_CASE("SerialOrderBehaviourGuided gives two events that peak together order credit for the first only", "[Solutions][SerialOrderBehaviour]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order-behaviour-guided" };

    auto together = loadSequenceTimingFixture<SerialOrderBehaviourGuided>("serial-order-fitness-test-held-double-peak.dnf");
    together.evaluate();

    CHECK(together.getParameters().partialFitness[3] < 0.25);
}

// From-scratch run t6 of 20 October 2026 steps D, C, A in turn, a slot later than the reference. Each released event goes
// back to being pre-shaped rather than below the field. That is the start of a release, so it earns some f6, and
// stepping beats t1's held peaks.
TEST_CASE("SerialOrderBehaviourGuided pays stepping from one event to the next over holding them", "[Solutions][SerialOrderBehaviour]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order-behaviour-guided" };

    auto stepping = loadSequenceTimingFixture<SerialOrderBehaviourGuided>("serial-order-fitness-test-late-stepping.dnf");
    stepping.evaluate();
    auto held = loadSequenceTimingFixture<SerialOrderBehaviourGuided>("serial-order-fitness-test-held-double-peak.dnf");
    held.evaluate();

    CHECK(stepping.getParameters().partialFitness[5] > 0.1);
    CHECK(stepping.getFitness() > held.getFitness() + 0.05);
}

// From-scratch run o1 of 21 October 2026 holds D, C, A and B as four self-sustained peaks in the input from their pulses
// on, reads them out as graded pre-shapes in the output, and steps through them one peak at a time. The input is a
// detection field: it peaks while an event is shown and lets go after. Pre-shapes now count as memory only as far as the
// input is at rest, so this keeps its recall credit but earns little for its memory, which earned f3 0.60 and f5 0.93.
TEST_CASE("SerialOrderBehaviourGuided gives little memory credit while the input holds the events", "[Solutions][SerialOrderBehaviour]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order-behaviour-guided" };

    auto inputHolds = loadSequenceTimingFixture<SerialOrderBehaviourGuided>("serial-order-fitness-test-input-holds-items.dnf");
    inputHolds.evaluate();

    const auto& partials = inputHolds.getParameters().partialFitness;
    CHECK(partials[2] < 0.3);
    CHECK(partials[4] < 0.3);
    CHECK(partials[3] > 0.7);
}

// From-scratch run o2 of 21 October 2026: X pushes all four events over threshold within 20, highest pre-shape first,
// and all four fade together. Sampled every 10, they peak in order, so each earned most of a recall point: f4 0.54.
// Recall is one peak at a time, so each sample's recall credit is now shared among the peaks up together. The burst
// keeps a little, a start to build on, but far less than o1's recall one peak at a time. Its common fade earned f6 0.47;
// a release now counts only as far as the event was recalled in its place.
TEST_CASE("SerialOrderBehaviourGuided shares recall credit among the peaks up together", "[Solutions][SerialOrderBehaviour]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order-behaviour-guided" };

    auto burst = loadSequenceTimingFixture<SerialOrderBehaviourGuided>("serial-order-fitness-test-burst.dnf");
    burst.evaluate();
    auto oneAtATime = loadSequenceTimingFixture<SerialOrderBehaviourGuided>("serial-order-fitness-test-input-holds-items.dnf");
    oneAtATime.evaluate();

    const double burstRecall = burst.getParameters().partialFitness[3];
    CHECK(burstRecall < 0.4);
    CHECK(burstRecall > 0.05);
    CHECK(burstRecall < 0.5 * oneAtATime.getParameters().partialFitness[3]);
    CHECK(burst.getParameters().partialFitness[5] < 0.3);
}

// From-scratch run t6 of 20 October 2026 forms a peak at D during the demonstration, before any X. Memory is pre-shapes
// below threshold, so a demonstration sample's memory now counts only as far as the output has no peak; it earned f3
// 0.82.
TEST_CASE("SerialOrderBehaviourGuided gives less memory credit while the output peaks during the demonstration", "[Solutions][SerialOrderBehaviour]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order-behaviour-guided" };

    auto peaksEarly = loadSequenceTimingFixture<SerialOrderBehaviourGuided>("serial-order-fitness-test-late-stepping.dnf");
    peaksEarly.evaluate();

    CHECK(peaksEarly.getParameters().partialFitness[2] < 0.7);
}

// From-scratch run g2 of 20 October 2026: the output sits at threshold and crosses it by 0.0-0.1 at A and B exactly when
// they are due. Height was a twentieth of a recall point, so those crossings earned f4 0.30-0.38. A recalled peak now
// earns its credit in proportion to its height, in full from 2, and they earn almost nothing.
TEST_CASE("SerialOrderBehaviourGuided gives an output flickering at threshold almost no recall credit", "[Solutions][SerialOrderBehaviour]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order-behaviour-guided" };

    auto flicker = loadSequenceTimingFixture<SerialOrderBehaviourGuided>("serial-order-fitness-test-threshold-flicker.dnf");
    flicker.evaluate();

    CHECK(flicker.getParameters().partialFitness[3] < 0.05);
}

// From-scratch run h3 of 20 October 2026 stores D, C and A in order, then from the first X sinks D and C into deep
// troughs without either ever peaking. Troughs at every earlier event were paid whether or not that event had been
// recalled, so this earned f6 0.89 and f7 0.75. A trough now earns credit only for an event that peaked when due in that
// cycle, and an output with no peak only as far as the cycle recalled, so these earn nothing: f6 and f7 are 0.
TEST_CASE("SerialOrderBehaviourGuided pays a trough only at an event recalled first", "[Solutions][SerialOrderBehaviour]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order-behaviour-guided" };

    auto troughs = loadSequenceTimingFixture<SerialOrderBehaviourGuided>("serial-order-fitness-test-troughs-without-recall.dnf");
    troughs.evaluate();

    const auto& partials = troughs.getParameters().partialFitness;
    REQUIRE(partials[3] == 0.0); // nothing was recalled
    CHECK(partials[5] < 0.05);
    CHECK(partials[6] < 0.05);
}

// With no connection into the output, it stays at rest: quiet through the demonstration, but with nothing stored,
// recalled or suppressed. Noise moves single samples by hundredths, so those terms come out near zero, not exactly.
// An output with no peak after recall used to earn half of f7 even when nothing was recalled; silence now counts only
// as far as the cycle recalled in order, so a silent output earns no f7.
TEST_CASE("SerialOrderBehaviourGuided gives no memory or recall credit to an output that never responds", "[Solutions][SerialOrderBehaviour]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order-behaviour-guided" };
    SerialOrderBehaviourGuided solution(sequenceTimingTopology());
    solution.addFieldGene(makeFixedFieldGene(FieldGeneType::INPUT, 1));
    solution.addFieldGene(makeFixedFieldGene(FieldGeneType::OUTPUT, 2));

    solution.evaluate();

    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 7);
    CHECK(partials[1] == Catch::Approx(1.0));             // no peak while the sequence is demonstrated
    CHECK(partials[2] < 0.05);                            // no pre-shapes of the stored events
    CHECK(partials[3] == 0.0);                            // no recall peaks
    CHECK(partials[4] < 0.05);                            // no pre-shapes of the events still to recall
    CHECK(partials[5] < 0.05);                            // no troughs at the recalled events
    CHECK(partials[6] < 0.05);                            // after recall: quiet, but nothing was recalled to stop
}

static bool withinRange(const double value, const double min, const double max)
{
    return value >= min && value <= max;
}

// A new connection starts with no global amplitude, and mutation moves it in fixed steps, so only multiples of the
// step are reachable.
static bool onConnectionGlobalStep(const double amplitudeGlobal)
{
    const double step = ConnectionGeneConstants::ampGlobalStep.value_or(GaussKernelConstants::ampGlobalStep);
    return std::abs(std::remainder(amplitudeGlobal, step)) < 1e-9;
}

// Mutation moves a value by whole steps, so stepping lands on a setting only if it sits on a multiple of the step.
static bool onStep(const double value, const double step)
{
    return std::abs(std::remainder(value, step)) < 1e-9;
}

// A parameter the task fixes for the input or output is only reachable at exactly that value.
static void checkFixedParametersMatch(const FieldGene& gene)
{
    using namespace dnf_composer::element;
    const auto type = gene.getParameters().type;
    if (type == FieldGeneType::HIDDEN)
    {
        return;
    }
    const auto& fixed = type == FieldGeneType::INPUT ? FieldGeneConstants::fixedInputField : FieldGeneConstants::fixedOutputField;
    const auto field = gene.getNeuralField()->getParameters();
    CHECK(field.tau == fixed.tau.value_or(field.tau));
    CHECK(field.startingRestingLevel == fixed.restingLevel.value_or(field.startingRestingLevel));
    if (fixed.gaussKernel)
    {
        const auto gauss = std::dynamic_pointer_cast<GaussKernel>(gene.getKernel());
        REQUIRE(gauss != nullptr);
        CHECK(gauss->getParameters().width == fixed.gaussKernel->width);
        CHECK(gauss->getParameters().amplitude == fixed.gaussKernel->amplitude);
        CHECK(gauss->getParameters().amplitudeGlobal == fixed.gaussKernel->amplitudeGlobal);
    }
}

static void checkFieldGeneEvolvable(const FieldGene& gene)
{
    using namespace dnf_composer::element;
    const auto field = gene.getNeuralField()->getParameters();
    INFO("field " << gene.getParameters().id);
    CHECK(withinRange(field.tau, NeuralFieldConstants::tauMinVal, NeuralFieldConstants::tauMaxVal));
    CHECK(withinRange(field.startingRestingLevel, NeuralFieldConstants::restingLevelMinVal, NeuralFieldConstants::restingLevelMaxVal));
    checkFixedParametersMatch(gene);

    if (const auto gauss = std::dynamic_pointer_cast<GaussKernel>(gene.getKernel()))
    {
        CHECK(FieldGeneConstants::gaussKernelProbability > 0.0);
        const auto kernel = gauss->getParameters();
        CHECK(withinRange(kernel.width, GaussKernelConstants::widthMinVal, GaussKernelConstants::widthMaxVal));
        CHECK(withinRange(kernel.amplitude, GaussKernelConstants::ampMinVal, GaussKernelConstants::ampMaxVal));
        CHECK(withinRange(kernel.amplitudeGlobal, GaussKernelConstants::ampGlobalMinVal, GaussKernelConstants::ampGlobalMaxVal));
        return;
    }
    const auto mexicanHat = std::dynamic_pointer_cast<MexicanHatKernel>(gene.getKernel());
    REQUIRE(mexicanHat != nullptr);
    CHECK(FieldGeneConstants::mexicanHatKernelProbability > 0.0);
    const auto kernel = mexicanHat->getParameters();
    CHECK(withinRange(kernel.widthExc, MexicanHatKernelConstants::widthExcMinVal, MexicanHatKernelConstants::widthExcMaxVal));
    CHECK(withinRange(kernel.amplitudeExc, MexicanHatKernelConstants::ampExcMinVal, MexicanHatKernelConstants::ampExcMaxVal));
    CHECK(withinRange(kernel.widthInh, MexicanHatKernelConstants::widthInhMinVal, MexicanHatKernelConstants::widthInhMaxVal));
    CHECK(withinRange(kernel.amplitudeInh, MexicanHatKernelConstants::ampInhMinVal, MexicanHatKernelConstants::ampInhMaxVal));
    CHECK(withinRange(kernel.amplitudeGlobal, MexicanHatKernelConstants::ampGlobMin, MexicanHatKernelConstants::ampGlobMax));
}

static void checkConnectionGeneEvolvable(const ConnectionGene& gene)
{
    const auto gauss = std::dynamic_pointer_cast<dnf_composer::element::GaussKernel>(gene.getKernel());
    INFO("connection " << gene.getInFieldGeneId() << "-" << gene.getOutFieldGeneId());
    REQUIRE(gauss != nullptr);
    CHECK(ConnectionGeneConstants::gaussKernelProbability > 0.0);
    const auto kernel = gauss->getParameters();
    CHECK(withinRange(kernel.width, GaussKernelConstants::widthMinVal, GaussKernelConstants::widthMaxVal));
    CHECK(withinRange(std::abs(kernel.amplitude),
        ConnectionGeneConstants::gaussAmpMinVal.value_or(GaussKernelConstants::ampMinVal),
        ConnectionGeneConstants::gaussAmpMaxVal.value_or(GaussKernelConstants::ampMaxVal)));
    CHECK(withinRange(kernel.amplitudeGlobal,
        ConnectionGeneConstants::ampGlobalMinVal.value_or(GaussKernelConstants::ampGlobalMinVal),
        ConnectionGeneConstants::ampGlobalMaxVal.value_or(GaussKernelConstants::ampGlobalMaxVal)));
    CHECK(onConnectionGlobalStep(kernel.amplitudeGlobal));
}

// The fitness is written from the reference's responses, so evolving it from scratch is only a fair test if each task's
// ranges reach every setting of that design. Evolution here makes Gauss kernels only, so the design it can reach is the
// reference with Gauss self-kernels in place of the Mexican hats of nf 3 and the done field nf 7.
TEST_CASE("Every serial-order task's evolvable ranges reach every setting of the Gauss-only reference design", "[Solutions][SerialOrderBehaviour]")
{
    for (const std::string slug : serialOrderTasks)
    {
        DYNAMIC_SECTION("task " << slug)
        {
            resetGlobalState();
            const ScopedTaskConfig taskConfig{ slug };
            const auto* task = findTask(slug);
            REQUIRE(task != nullptr);

            const auto solution = task->makeFromTemplate(defaultTopologyFor(*task),
                *loadFixtureSimulation("serial-order-fitness-test-reference-gauss.dnf"));

            for (const auto& gene : solution->getGenome().getFieldGenes())
            {
                checkFieldGeneEvolvable(gene);
            }
            for (const auto& gene : solution->getGenome().getConnectionGenes())
            {
                checkConnectionGeneEvolvable(gene);
            }
        }
    }
}

// Evolution never wires a connection out of the output field, but a template is loaded without that check.
static void checkOnlyConnectionsEvolutionCanMake(const Solution& solution)
{
    const auto legalTuples = solution.getGenome().legalConnectionTuples();
    for (const auto& gene : solution.getGenome().getConnectionGenes())
    {
        INFO("connection " << gene.getInFieldGeneId() << "-" << gene.getOutFieldGeneId());
        CHECK(std::ranges::find(legalTuples, gene.getParameters().connectionTuple) != legalTuples.end());
    }
}

// The tasks of the sequence-timing experiment, intended behaviour alone and with guides.
static const std::array<std::string, 2> sequenceTimingTasks{ "sequence-timing-behaviour", "sequence-timing-behaviour-guided" };

// The reference design the fitness was written from (the hand-built boost design with its order and timing gradient
// read out into the output, 25 October 2026), and the same design with Gauss self-kernels in place of its Mexican hats,
// as evolution here makes them.
static const std::array<std::string, 2> sequenceTimingReferences{ "sequence-timing-behaviour-reference.dnf",
    "sequence-timing-behaviour-reference-gauss.dnf" };

// Every trial of an evaluation, read from its trace.
static nlohmann::json timingTrials(const Solution& solution)
{
    return nlohmann::json::parse(solution.getParameters().evaluationTrace).at("trials");
}

TEST_CASE("SequenceTimingBehaviour contract", "[Solutions][SequenceTimingBehaviour]")
{
    resetGlobalState();
    checkSolutionContract<SequenceTimingBehaviour>(makeTopology(1, 1));
}

TEST_CASE("SequenceTimingBehaviourGuided contract", "[Solutions][SequenceTimingBehaviour]")
{
    resetGlobalState();
    checkSolutionContract<SequenceTimingBehaviourGuided>(makeTopology(1, 1));
}

TEST_CASE("SequenceTimingBehaviour and its guided variant name their runs apart and clone and copy as themselves", "[Solutions][SequenceTimingBehaviour]")
{
    resetGlobalState();

    SequenceTimingBehaviour behaviour(sequenceTimingTopology());
    SequenceTimingBehaviourGuided guided(sequenceTimingTopology());
    for (Solution* solution : std::initializer_list<Solution*>{ &behaviour, &guided })
    {
        solution->initialize();
        solution->buildPhenotype();
    }

    CHECK(behaviour.getName() == "Sequence Timing Behaviour");
    CHECK(guided.getName() == "Sequence Timing Behaviour Guided");
    CHECK(behaviour.clone()->getName() == behaviour.getName());
    CHECK(behaviour.copy()->getName() == behaviour.getName());
    CHECK(std::dynamic_pointer_cast<SequenceTimingBehaviourGuided>(guided.clone()) != nullptr);
    CHECK(std::dynamic_pointer_cast<SequenceTimingBehaviourGuided>(guided.copy()) != nullptr);
}

TEST_CASE("SequenceTimingBehaviourGuided scores both reference designs above 0.9 on every term", "[Solutions][SequenceTimingBehaviour]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "sequence-timing-behaviour-guided" };

    for (const auto& fixture : sequenceTimingReferences)
    {
        auto solution = loadSequenceTimingFixture<SequenceTimingBehaviourGuided>(fixture);
        for (int evaluation = 0; evaluation < 2; ++evaluation)
        {
            solution.evaluate();
            const auto& partials = solution.getParameters().partialFitness;
            REQUIRE(partials.size() == 8);
            for (size_t term = 0; term < partials.size(); ++term)
            {
                INFO(fixture << ", evaluation " << evaluation << ", f" << term + 1);
                CHECK(partials[term] > 0.9);
            }
            CHECK(solution.getFitness() > 0.95);
        }
    }
}

// Recall, timing, release and stop are scored as in the guided task; only what lies below threshold no longer counts.
TEST_CASE("SequenceTimingBehaviour scores both reference designs above 0.9 on each of its six terms", "[Solutions][SequenceTimingBehaviour]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "sequence-timing-behaviour" };

    for (const auto& fixture : sequenceTimingReferences)
    {
        auto solution = loadSequenceTimingFixture<SequenceTimingBehaviour>(fixture);
        solution.evaluate();
        const auto& partials = solution.getParameters().partialFitness;
        REQUIRE(partials.size() == 6);
        for (size_t term = 0; term < partials.size(); ++term)
        {
            INFO(fixture << ", f" << term + 1);
            CHECK(partials[term] > 0.9);
        }
    }
}

// The hand-built design the reference was made from recalls the same way, but holds the order and timing only in hidden
// fields: its output shows no pre-shape before X, and each waiting event's only shortly before its peak. The guides do
// not pay it; the guided fitness is the larger of the guided score and the intended behaviour's, so it keeps the
// latter.
TEST_CASE("SequenceTimingBehaviourGuided pays the reference's pre-shapes, which the hand-built design it extends lacks", "[Solutions][SequenceTimingBehaviour]")
{
    resetGlobalState();
    double intendedFitness = 0.0;
    {
        const ScopedTaskConfig taskConfig{ "sequence-timing-behaviour" };
        auto handBuilt = loadSequenceTimingFixture<SequenceTimingBehaviour>("sequence-timing-boost-handbuilt.dnf");
        handBuilt.evaluate();
        intendedFitness = handBuilt.getFitness();
        CHECK(intendedFitness > 0.9);
    }
    const ScopedTaskConfig taskConfig{ "sequence-timing-behaviour-guided" };
    auto handBuilt = loadSequenceTimingFixture<SequenceTimingBehaviourGuided>("sequence-timing-boost-handbuilt.dnf");
    auto reference = loadSequenceTimingFixture<SequenceTimingBehaviourGuided>("sequence-timing-behaviour-reference.dnf");
    handBuilt.evaluate();
    reference.evaluate();

    const auto& partials = handBuilt.getParameters().partialFitness;
    REQUIRE(partials.size() == 8);
    CHECK(partials[2] < 0.1); // no memory in the output before X
    CHECK(partials[5] < 0.2); // the waiting events barely pre-shaped
    CHECK(partials[3] > 0.9); // but recalled in order
    CHECK(partials[4] > 0.9); // and on time
    const auto scores = nlohmann::json::parse(handBuilt.getParameters().evaluationTrace).at("fitness");
    CHECK(scores.at("guided").get<double>() < reference.getFitness() - 0.1);
    CHECK(handBuilt.getFitness() == Catch::Approx(intendedFitness).margin(0.02));
}

// Saved from timing-batch-1 (run TG01, 25 October 2026): D, C, A and B in order, at a steady pace of about 480 each that
// ignores the demonstrated intervals, with every guide in place. A design that recalls on time without the guides ranks
// above it.
TEST_CASE("SequenceTimingBehaviourGuided ranks recall on time without the guides above recall off time with them", "[Solutions][SequenceTimingBehaviour]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "sequence-timing-behaviour-guided" };

    auto onTime = loadSequenceTimingFixture<SequenceTimingBehaviourGuided>("sequence-timing-boost-handbuilt.dnf");
    auto steadyPace = loadSequenceTimingFixture<SequenceTimingBehaviourGuided>("sequence-timing-behaviour-steady-pace.dnf");
    onTime.evaluate();
    steadyPace.evaluate();

    CHECK(onTime.getFitness() > 0.9);
    CHECK(onTime.getFitness() > steadyPace.getFitness());
}

// With no connection into the output field, nothing can ever drive a peak there: the output earns full credit for
// staying quiet before X and none for recall.
TEST_CASE("SequenceTimingBehaviour gives an output that never peaks no recall, timing, release or stop credit", "[Solutions][SequenceTimingBehaviour]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "sequence-timing-behaviour" };

    SequenceTimingBehaviour solution(sequenceTimingTopology());
    solution.addFieldGene(makeFixedFieldGene(FieldGeneType::INPUT, 1));
    solution.addFieldGene(makeFixedFieldGene(FieldGeneType::OUTPUT, 2));
    solution.evaluate();

    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 6);
    CHECK(partials[1] == Catch::Approx(1.0));
    CHECK(partials[2] == 0.0);
    CHECK(partials[3] == 0.0);
    CHECK(partials[4] == 0.0);
    CHECK(partials[5] == 0.0);
}

// The hand-built serial-order design recalls all four, one at a time, at its own pace (+50, +300, +430, +560) rather
// than at the demonstrated times. Under this demonstration it brings B before A, so its order credit is not full either.
TEST_CASE("SequenceTimingBehaviour pays recall in order at its own pace for the order, not the timing", "[Solutions][SequenceTimingBehaviour]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "sequence-timing-behaviour" };

    auto solution = loadSequenceTimingFixture<SequenceTimingBehaviour>("serial-order-handbuilt.dnf");
    solution.evaluate();

    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 6);
    CHECK(partials[2] > 0.8);
    CHECK(partials[3] < 0.3);
}

// Saved from a from-scratch boost run: a few hundred time units after X the output ignites at every event at once.
TEST_CASE("SequenceTimingBehaviour gives a burst of simultaneous peaks little recall credit", "[Solutions][SequenceTimingBehaviour]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "sequence-timing-behaviour" };

    auto solution = loadSequenceTimingFixture<SequenceTimingBehaviour>("sequence-timing-boost-burst.dnf");
    solution.evaluate();

    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 6);
    CHECK(partials[2] < 0.3);
    CHECK(partials[3] < 0.3);
}

// Saved from a from-scratch run (run 23): it replays D, C, A, B in order at a fixed lag after the demonstration, and so
// also fires when X never comes.
TEST_CASE("SequenceTimingBehaviour scales recall credit by how quiet the output stays in the probe trial without X", "[Solutions][SequenceTimingBehaviour]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "sequence-timing-behaviour" };

    auto solution = loadSequenceTimingFixture<SequenceTimingBehaviour>("sequence-timing-boost-fixed-lag.dnf");
    solution.evaluate();

    const auto trials = timingTrials(solution);
    REQUIRE(trials.front().at("kind") == "probe");
    const double probeQuiet = trials.front().at("quietWithoutCue").get<double>();
    REQUIRE(probeQuiet < 0.9);
    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 6);
    for (size_t term = 2; term < partials.size(); ++term)
    {
        INFO("f" << term + 1);
        CHECK(partials[term] <= probeQuiet + 1e-9);
    }
}

// Saved from timing-batch-1 (run T01, 25 October 2026): about 2550 after the demonstration, whatever the gap, it fires D,
// C, A and B one after another within 80, and lets go. So brief a burst leaves the probe trial quiet at almost every
// sample, but every event peaks there without X.
TEST_CASE("SequenceTimingBehaviour gives almost no credit to a brief burst at a fixed lag after the demonstration", "[Solutions][SequenceTimingBehaviour]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "sequence-timing-behaviour" };

    auto solution = loadSequenceTimingFixture<SequenceTimingBehaviour>("sequence-timing-behaviour-fixed-lag-burst.dnf");
    solution.evaluate();

    CHECK(solution.getFitness() < 0.2);
}

// Saved from timing-batch-1 (run T02, 25 October 2026): after X it recalls D, C, A and B in order, but 40, 80, 100 and
// 110 after it, far earlier than their times. Far from its time the Gaussian is flat, so the timing credit keeps a share
// that grows with the delay up to the due time.
TEST_CASE("SequenceTimingBehaviour gives recall far too early a little timing credit", "[Solutions][SequenceTimingBehaviour]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "sequence-timing-behaviour" };

    auto solution = loadSequenceTimingFixture<SequenceTimingBehaviour>("sequence-timing-behaviour-fast-cascade.dnf");
    solution.evaluate();

    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 6);
    REQUIRE(partials[2] > 0.5);
    CHECK(partials[3] > 0.02);
    CHECK(partials[3] < 0.2);
}

// Saved from a from-scratch run: the output is active almost everywhere, almost all the time (quiet at 1-33% of the
// samples, varying between evaluations), and after X travelling waves sweep the field, so each event sees a peak pass.
// Each recall's credit is scaled by how quiet the output was before X.
TEST_CASE("SequenceTimingBehaviour gives no recall credit to an output that is never quiet before X", "[Solutions][SequenceTimingBehaviour]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "sequence-timing-behaviour" };

    auto solution = loadSequenceTimingFixture<SequenceTimingBehaviour>("sequence-timing-wave-exploit.dnf");
    solution.evaluate();

    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 6);
    REQUIRE(partials[1] < 0.5);
    CHECK(partials[2] < 0.05);
    CHECK(partials[3] < 0.05);
}

// Fresh gaps on every evaluation stop a design from tuning its timing to a fixed set of gaps; one from each third of
// 300-1500 spans short and long gaps in every evaluation. Rounded to the sampling interval, X falls on a sample.
TEST_CASE("SequenceTimingBehaviour runs the probe, then draws a fresh gap from each third of 300-1500 for each recall", "[Solutions][SequenceTimingBehaviour]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "sequence-timing-behaviour" };

    SequenceTimingBehaviour solution(sequenceTimingTopology());
    solution.addFieldGene(makeFixedFieldGene(FieldGeneType::INPUT, 1));
    solution.addFieldGene(makeFixedFieldGene(FieldGeneType::OUTPUT, 2));
    const auto recallGaps = [&solution]() {
        solution.evaluate();
        const auto trials = timingTrials(solution);
        REQUIRE(trials.size() == 4);
        CHECK(trials.front().at("kind") == "probe");
        std::vector<double> gaps;
        for (size_t trial = 1; trial < trials.size(); ++trial)
        {
            CHECK(trials[trial].at("kind") == "recall");
            gaps.push_back(trials[trial].at("gap").get<double>());
        }
        return gaps;
    };

    const auto first = recallGaps();
    const auto second = recallGaps();
    for (size_t i = 0; i < first.size(); ++i)
    {
        CHECK(first[i] >= 300.0 + 400.0 * static_cast<double>(i));
        CHECK(first[i] <= 700.0 + 400.0 * static_cast<double>(i));
        CHECK(std::fmod(first[i], 10.0) == Catch::Approx(0.0).margin(1e-9));
    }
    CHECK(first != second);
}

// Each recall trial's trace says when each event first peaked after X and how close that came to its time.
TEST_CASE("SequenceTimingBehaviourGuided traces when each event first peaked after X, and its timing credit", "[Solutions][SequenceTimingBehaviour]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "sequence-timing-behaviour-guided" };

    auto solution = loadSequenceTimingFixture<SequenceTimingBehaviourGuided>("sequence-timing-behaviour-reference.dnf");
    solution.evaluate();

    const std::map<std::string, double> demonstrated{ { "D", 100.0 }, { "C", 600.0 }, { "A", 800.0 }, { "B", 1800.0 } };
    const auto trials = timingTrials(solution);
    REQUIRE(trials.size() == 4);
    for (size_t trial = 1; trial < trials.size(); ++trial)
    {
        const auto& events = trials[trial].at("recall").at("events");
        for (const auto& [label, onset] : demonstrated)
        {
            INFO("trial " << trial << ", " << label);
            const double afterCue = events.at(label).at("afterCue").get<double>();
            CHECK(std::abs(afterCue - onset) <= 0.2 * onset);
            CHECK(events.at(label).at("onTime").get<double>() > 0.5);
        }
        CHECK(trials[trial].at("quietBeforeCue").get<double>() == Catch::Approx(1.0));
        CHECK(trials[trial].at("snapshots").size() > 8);
    }
}

// A saved solution carries what createPhenotypeEnvironment() adds: a recall trial with a gap of 700, the trial of the
// reference's behaviour table, so a saved design opened in dnf-composer replays it.
TEST_CASE("SequenceTimingBehaviourGuided saves a recall trial with a gap of 700, so a saved design replays it", "[Solutions][SequenceTimingBehaviour]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "sequence-timing-behaviour-guided" };
    const auto fixture = loadFixtureSimulation("sequence-timing-behaviour-reference.dnf");
    SequenceTimingBehaviourGuided solution(sequenceTimingTopology(), *fixture);
    const auto saveDirectory = std::filesystem::temp_directory_path() / "neat-dnfs-sequence-timing-behaviour-replay";

    // As PopulationFileManager saves a solution: into <directory>/<identifier>/<identifier>.dnf.
    Solution& saved = solution;
    saved.buildPhenotype();
    saved.createPhenotypeEnvironment();
    std::filesystem::path savedFile;
    {
        const auto simulation = std::make_shared<dnf_composer::Simulation>(saved.getPhenotype());
        dnf_composer::SimulationFileManager(simulation, saveDirectory.generic_string()).saveElementsToJson();
        const std::string identifier = simulation->getUniqueIdentifier();
        savedFile = saveDirectory / identifier / (identifier + ".dnf");
    }
    saved.clearPhenotype();
    REQUIRE(std::filesystem::exists(savedFile));

    const auto replay = std::make_shared<dnf_composer::Simulation>();
    dnf_composer::SimulationFileManager(replay, savedFile.generic_string()).loadElementsFromJson();
    replay->init();
    const auto output = std::dynamic_pointer_cast<dnf_composer::element::NeuralField>(replay->getElement("nf 2"));
    REQUIRE(output != nullptr);
    // Inside the reference's peaks: X comes again at 2550.
    for (const auto& [time, position] : std::vector<std::pair<double, double>>{ { 2690.0, 180.0 }, { 3170.0, 120.0 },
        { 3370.0, 20.0 }, { 4340.0, 80.0 } })
    {
        while (replay->getT() < time - 1e-9)
        {
            replay->step();
        }
        const auto bumps = output->getBumps();
        INFO("t " << time);
        REQUIRE(bumps.size() == 1);
        CHECK(std::abs(bumps.front().centroid - position) <= 1.0);
    }
    std::filesystem::remove_all(saveDirectory);
}

// The fitness is written from the reference's responses, so evolving it from scratch is only a fair test if each task's
// ranges reach every setting of that design. Evolution here makes Gauss kernels only, so the design it can reach is the
// reference with Gauss self-kernels in place of the Mexican hats of the memory nf 3 and the past-events field nf 6.
TEST_CASE("Every sequence-timing task's evolvable ranges reach every setting of the Gauss-only reference design", "[Solutions][SequenceTimingBehaviour]")
{
    for (const std::string slug : sequenceTimingTasks)
    {
        DYNAMIC_SECTION("task " << slug)
        {
            OutputSourceRuleGuard guard;
            resetGlobalState();
            const ScopedTaskConfig taskConfig{ slug };
            const auto* task = findTask(slug);
            REQUIRE(task != nullptr);

            const auto solution = task->makeFromTemplate(defaultTopologyFor(*task),
                *loadFixtureSimulation("sequence-timing-behaviour-reference-gauss.dnf"));
            REQUIRE(solution->getGenome().getFieldGenes().size() == 8);
            for (const auto& gene : solution->getGenome().getFieldGenes())
            {
                checkFieldGeneEvolvable(gene);
            }
            for (const auto& gene : solution->getGenome().getConnectionGenes())
            {
                checkConnectionGeneEvolvable(gene);
            }
            // Reachable without output feedback, which these configs turn on.
            GenomeMutationConstants::allowOutputAsConnectionSource = false;
            checkOnlyConnectionsEvolutionCanMake(*solution);
        }
    }
}

TEST_CASE("Every sequence-timing task is registered with one input and one output, starting from the sequence-timing template", "[SolutionRegistry]")
{
    for (const std::string slug : sequenceTimingTasks)
    {
        INFO("task " << slug);
        const auto* task = findTask(slug);
        REQUIRE(task != nullptr);
        CHECK(task->templateFile == "sequence-timing.dnf");
        CHECK(task->inputs == 1);
        CHECK(task->outputs == 1);
    }
}

// The final task of the serial-order and timing work (from 29 October 2026): ten events 20 apart on fields of 240.
static const std::string sequenceAndTimingRecall{ "sequence-and-timing-recall" };

// The ten events in their order, each with when it is due after the memory recall trigger.
static const std::array<std::pair<std::string, double>, 10> recallEvents{ { { "LA2B", 50.0 }, { "RA3A", 150.0 },
    { "LA1A", 350.0 }, { "RA1B", 450.0 }, { "RA2B", 650.0 }, { "LA3A", 750.0 }, { "RA1A", 950.0 }, { "LA1B", 1050.0 },
    { "HOL", 1900.0 }, { "HOS", 2800.0 } } };

// Where each event lies on the input and output fields, in the same order.
static const std::array<double, 10> recallEventPositions{ 40.0, 160.0, 60.0, 100.0, 120.0, 80.0, 140.0, 20.0, 200.0,
    220.0 };

static SequenceAndTimingRecall loadRecallFixture(const std::string& name)
{
    return loadSequenceTimingFixture<SequenceAndTimingRecall>("sequence-and-timing-recall-" + name + ".dnf");
}

// How many triggers come in turn after the one demonstration of an evaluation.
static constexpr size_t recallTriggers = 5;

// Every recall of an evaluation, in turn: the five of its one demonstration.
static std::vector<nlohmann::json> recallsOf(const nlohmann::json& trials)
{
    std::vector<nlohmann::json> recalls;
    for (const auto& trial : trials)
    {
        if (trial.at("kind") == "consecutive")
        {
            for (const auto& recall : trial.at("recalls"))
            {
                recalls.push_back(recall);
            }
        }
    }
    return recalls;
}

// One value of every recall of the last evaluation, in turn.
static std::vector<double> perRecallOf(const Solution& solution, const std::string& key)
{
    std::vector<double> values;
    for (const auto& recall : recallsOf(timingTrials(solution)))
    {
        values.push_back(recall.at(key).get<double>());
    }
    return values;
}

// The share of samples with no output bump in the demonstration and first wait, the probe and the tail: every recall
// term is multiplied by it.
static double recallGateOf(const Solution& solution)
{
    return nlohmann::json::parse(solution.getParameters().evaluationTrace).at("gate").at("gate").get<double>();
}

TEST_CASE("SequenceAndTimingRecall contract", "[Solutions][SequenceAndTimingRecall]")
{
    resetGlobalState();
    checkSolutionContract<SequenceAndTimingRecall>(makeTopology(1, 1));
}

TEST_CASE("SequenceAndTimingRecall names its runs, and clones and copies as itself", "[Solutions][SequenceAndTimingRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ sequenceAndTimingRecall };
    SequenceAndTimingRecall solution(sequenceTimingTopology());
    solution.initialize();
    solution.buildPhenotype();

    CHECK(solution.getName() == "Sequence And Timing Recall");
    CHECK(std::dynamic_pointer_cast<SequenceAndTimingRecall>(solution.clone()) != nullptr);
    CHECK(std::dynamic_pointer_cast<SequenceAndTimingRecall>(solution.copy()) != nullptr);
}

// The hand-built reference recalls all ten in order and on time after every trigger, lets each go into a trough, and
// holds a primacy gradient between recalls. Its HOL and HOS sit level in that gradient, so the guides that read the
// gradient (f2, f6 and the guided half of f8) stay below 1.
TEST_CASE("SequenceAndTimingRecall scores the reference design at least 0.9", "[Solutions][SequenceAndTimingRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ sequenceAndTimingRecall };
    auto solution = loadRecallFixture("reference");

    for (int evaluation = 0; evaluation < 2; ++evaluation)
    {
        solution.evaluate();
        const auto& partials = solution.getParameters().partialFitness;
        REQUIRE(partials.size() == 8);
        for (const size_t term : std::initializer_list<size_t>{ 0, 2, 3, 4, 6 })
        {
            INFO("evaluation " << evaluation << ", f" << term + 1);
            CHECK(partials[term] > 0.9);
        }
        for (const size_t term : std::initializer_list<size_t>{ 1, 5, 7 })
        {
            INFO("evaluation " << evaluation << ", guide f" << term + 1);
            CHECK(partials[term] > 0.7);
        }
        CHECK(recallGateOf(solution) == Catch::Approx(1.0));
        const auto scores = nlohmann::json::parse(solution.getParameters().evaluationTrace).at("fitness");
        CHECK(scores.at("intended").get<double>() >= 0.9);
        CHECK(solution.getFitness() >= 0.9);
    }
}

// With no connection into the output, nothing drives a peak or a pre-shape there: the design earns the input terms
// alone, a twentieth of either weighted sum.
TEST_CASE("SequenceAndTimingRecall gives an output that never responds the input terms alone", "[Solutions][SequenceAndTimingRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ sequenceAndTimingRecall };
    SequenceAndTimingRecall solution(sequenceTimingTopology());
    solution.addFieldGene(makeFixedFieldGene(FieldGeneType::INPUT, 1));
    solution.addFieldGene(makeFixedFieldGene(FieldGeneType::OUTPUT, 2));
    solution.evaluate();

    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 8);
    CHECK(partials[0] > 0.9);
    CHECK(partials[1] == 0.0);
    CHECK(partials[2] > 0.9);
    for (size_t term = 3; term < partials.size(); ++term)
    {
        INFO("f" << term + 1);
        CHECK(partials[term] == 0.0);
    }
    CHECK(solution.getFitness() == Catch::Approx(0.05).margin(0.005));
}

// The reference with 4-5 at 0 records no stamps, so every event fires at once after each trigger. Three or more events
// first peaking at one sample are not in order (the user's tie rule of 5 November 2026), and none is on time.
TEST_CASE("SequenceAndTimingRecall gives a burst almost no order and no timing credit", "[Solutions][SequenceAndTimingRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ sequenceAndTimingRecall };
    auto solution = loadRecallFixture("burst");
    solution.evaluate();

    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 8);
    CHECK(partials[3] < 0.2);
    CHECK(partials[4] < 0.1);
    CHECK(solution.getFitness() < 0.2);
}

// The reference with 8-5 and 8-10 at 800: it recalls the ten in order at about a quarter of their times, after every
// trigger.
TEST_CASE("SequenceAndTimingRecall pays a fast cascade in order for the order, not the timing", "[Solutions][SequenceAndTimingRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ sequenceAndTimingRecall };
    auto solution = loadRecallFixture("fast-cascade");
    solution.evaluate();

    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 8);
    CHECK(partials[3] > 0.8);
    CHECK(partials[4] < 0.2);
}

// The reference with nf 6 at tau 120: each event is released later, so two or three peaks are up together, each where
// and when it is due. Two up together are clean; three are not, but only for a few samples.
TEST_CASE("SequenceAndTimingRecall gives ordered peaks that overlap in time full order and timing credit", "[Solutions][SequenceAndTimingRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ sequenceAndTimingRecall };
    auto solution = loadRecallFixture("overlapping-peaks");
    solution.evaluate();

    const auto recalls = recallsOf(timingTrials(solution));
    REQUIRE(recalls.size() == recallTriggers);
    for (size_t recall = 0; recall < recalls.size(); ++recall)
    {
        const auto& events = recalls[recall].at("recall").at("events");
        size_t mostUpTogether = 0;
        for (const auto& [label, due] : recallEvents)
        {
            const double firstPeak = events.at(label).at("firstPeak").get<double>();
            // Up at that moment: peaked already, and not yet released.
            const auto upTogether = std::ranges::count_if(recallEvents, [&](const auto& other) {
                const auto& otherEvent = events.at(other.first);
                const auto& releasedAt = otherEvent.at("releasedAt");
                return !otherEvent.at("firstPeak").is_null() && otherEvent.at("firstPeak").get<double>() <= firstPeak
                    && (releasedAt.is_null() || releasedAt.get<double>() > firstPeak);
            });
            mostUpTogether = std::max(mostUpTogether, static_cast<size_t>(upTogether));
        }
        INFO("recall " << recall);
        CHECK(mostUpTogether >= 2U);
        CHECK(recalls[recall].at("clean").get<double>() > 0.9);
    }
    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 8);
    CHECK(partials[3] > 0.9);
    CHECK(partials[4] > 0.9);
}

// The reference with 7-8's global part at 0.2: ten pending events open the window by themselves, so it starts recalling
// without any trigger and keeps firing. The probe and the first wait catch it, and the gate takes most of every recall
// term.
TEST_CASE("SequenceAndTimingRecall gives an output that fires without the trigger almost no recall credit", "[Solutions][SequenceAndTimingRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ sequenceAndTimingRecall };
    auto solution = loadRecallFixture("fires-without-trigger");
    solution.evaluate();

    const auto trials = timingTrials(solution);
    REQUIRE(trials.front().at("kind") == "probe");
    CHECK(trials.front().at("quietWithoutCue").get<double>() < 0.5);
    CHECK(recallGateOf(solution) < 0.8);
    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 8);
    for (size_t term = 3; term < partials.size(); ++term)
    {
        INFO("f" << term + 1);
        CHECK(partials[term] < 0.2);
    }
    CHECK(solution.getFitness() < 0.2);
}

// From 6 November 2026 (the user): the probe watches 1500 after the demonstration without a trigger; then one
// demonstration and five triggers in turn, each wait drawn afresh from 500-1500 (from the end of the demonstration, then
// from the end of the recall window before it); then a tail of 1500 without a trigger.
TEST_CASE("SequenceAndTimingRecall runs a probe of 1500, then one demonstration, five triggers after waits of 500-1500 and a tail of 1500", "[Solutions][SequenceAndTimingRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ sequenceAndTimingRecall };
    SequenceAndTimingRecall solution(sequenceTimingTopology());
    solution.addFieldGene(makeFixedFieldGene(FieldGeneType::INPUT, 1));
    solution.addFieldGene(makeFixedFieldGene(FieldGeneType::OUTPUT, 2));

    constexpr double demonstrationEnd = 2850.0;
    constexpr double recallWindow = 3300.0;
    std::set<double> waitsSeen;
    for (int evaluation = 0; evaluation < 8; ++evaluation)
    {
        solution.evaluate();
        const auto trials = timingTrials(solution);
        REQUIRE(trials.size() == 2);
        CHECK(trials.front().at("kind") == "probe");
        CHECK(trials.front().at("watched").get<double>() == 1500.0);
        REQUIRE(trials.back().at("kind") == "consecutive");
        CHECK(trials.back().at("tail").at("watched").get<double>() == 1500.0);
        const auto recalls = recallsOf(trials);
        REQUIRE(recalls.size() == recallTriggers);

        double windowEnd = demonstrationEnd;
        for (size_t trigger = 0; trigger < recalls.size(); ++trigger)
        {
            INFO("evaluation " << evaluation << ", trigger " << trigger);
            const double wait = recalls[trigger].at("wait").get<double>();
            CHECK(wait >= 500.0);
            CHECK(wait <= 1500.0);
            CHECK(std::abs(std::fmod(wait, 10.0)) < 1e-9);
            CHECK(recalls[trigger].at("cue").get<double>() == Catch::Approx(windowEnd + wait));
            CHECK(recalls[trigger].at("recall").at("events").size() == 10);
            windowEnd = recalls[trigger].at("cue").get<double>() + recallWindow;
            waitsSeen.insert(wait);
        }
    }
    CHECK(waitsSeen.size() > 20);
}

// The reference with 9-5 at 0: no wind-back, so after its first recall every stamp stays above threshold, and each later
// trigger fires the events nearly together. Every recall counts the same, so recalling once earns about a fifth.
TEST_CASE("SequenceAndTimingRecall pays a design that recalls only after the first trigger about a fifth of the recall credit", "[Solutions][SequenceAndTimingRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ sequenceAndTimingRecall };
    auto solution = loadRecallFixture("recalls-once");
    solution.evaluate();

    const auto recalls = recallsOf(timingTrials(solution));
    REQUIRE(recalls.size() == recallTriggers);
    const auto eventsOnTime = [](const nlohmann::json& recall) {
        const auto& events = recall.at("recall").at("events");
        return std::ranges::count_if(recallEvents,
            [&events](const auto& event) { return events.at(event.first).at("onTime").get<double>() > 0.5; });
    };
    CHECK(eventsOnTime(recalls[0]) == 10);
    for (size_t recall = 1; recall < recalls.size(); ++recall)
    {
        INFO("recall " << recall);
        CHECK(eventsOnTime(recalls[recall]) <= 2);
    }
    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 8);
    CHECK(partials[3] < 0.35);
    CHECK(partials[4] < 0.35);
    CHECK(solution.getFitness() < 0.4);
    CHECK_FALSE(solution.reachedTarget(0.9));
}

// The memory a recall starts from is read in the last 100 before its trigger: the ten events' output levels, each below
// the one due before it. The reference's wind-back restores its gradient after each recall; recalls-once has no
// wind-back, so after its first recall the gradient is gone.
TEST_CASE("SequenceAndTimingRecall reads the memory before every trigger, not only after the demonstration", "[Solutions][SequenceAndTimingRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ sequenceAndTimingRecall };
    const auto meanBeforeLaterTriggers = [](const Solution& solution) {
        const auto memory = perRecallOf(solution, "memoryBeforeCue");
        REQUIRE(memory.size() == recallTriggers);
        return std::accumulate(memory.begin() + 1, memory.end(), 0.0) / static_cast<double>(memory.size() - 1);
    };
    auto reference = loadRecallFixture("reference");
    auto recallsOnce = loadRecallFixture("recalls-once");
    for (int evaluation = 0; evaluation < 2; ++evaluation)
    {
        reference.evaluate();
        recallsOnce.evaluate();
        INFO("evaluation " << evaluation);
        CHECK(perRecallOf(reference, "memoryBeforeCue").front() > 0.7);
        CHECK(meanBeforeLaterTriggers(reference) > 0.7);
        CHECK(meanBeforeLaterTriggers(recallsOnce) < 0.3);
    }
}

// The best of batch-2's R08 (31 October 2026): after each trigger the output flickers, a sample at a time, at places all
// over the field, in no order. A flash earns a third of a peak, and its place in the order is chance.
TEST_CASE("SequenceAndTimingRecall pays a one-sample flicker almost nothing", "[Solutions][SequenceAndTimingRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ sequenceAndTimingRecall };
    auto solution = loadRecallFixture("flickers");
    for (int evaluation = 0; evaluation < 2; ++evaluation)
    {
        solution.evaluate();
        const auto& partials = solution.getParameters().partialFitness;
        REQUIRE(partials.size() == 8);
        INFO("evaluation " << evaluation);
        CHECK(partials[3] < 0.15);
        CHECK(solution.getFitness() < 0.2);
    }
}

// The best of batch-3's R02 (5 November 2026): at every trigger a wave spreads out from LA2B's place, so each event peaks
// once, cleanly, in order of its distance from LA2B, not in the demonstrated order. The old order rule paid it about two
// thirds of its credit, which ranked it above the only run that recalled in order; asked of each event whether it came
// at its position, almost none does.
TEST_CASE("SequenceAndTimingRecall gives a wave across the field, in order of place, almost no order credit", "[Solutions][SequenceAndTimingRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ sequenceAndTimingRecall };
    auto solution = loadRecallFixture("wave");
    for (int evaluation = 0; evaluation < 2; ++evaluation)
    {
        solution.evaluate();
        const auto& partials = solution.getParameters().partialFitness;
        REQUIRE(partials.size() == 8);
        INFO("evaluation " << evaluation);
        CHECK(partials[3] < 0.25);
        CHECK(partials[4] < 0.15);
        CHECK(solution.getFitness() < 0.2);
    }
}

// The best of batch-2's R04 (31 October 2026) lets events go and then peaks there again. A sample with a released event
// peaking again, before the sequence is over, is not clean; after it, it is not quiet.
TEST_CASE("SequenceAndTimingRecall counts an event that peaks again after its release against the recall", "[Solutions][SequenceAndTimingRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ sequenceAndTimingRecall };

    SECTION("the reference's events never peak again, and its recalls are clean")
    {
        auto solution = loadRecallFixture("reference");
        solution.evaluate();
        const auto refired = perRecallOf(solution, "refiredSamples");
        REQUIRE(refired.size() == recallTriggers);
        CHECK(std::ranges::all_of(refired, [](const double samples) { return samples == 0.0; }));
        const auto clean = perRecallOf(solution, "clean");
        CHECK(std::ranges::all_of(clean, [](const double share) { return share > 0.95; }));
    }

    SECTION("a design whose events peak again loses order and after-recall credit")
    {
        auto solution = loadRecallFixture("refires");
        double mostRefired = 0.0;
        for (int evaluation = 0; evaluation < 2; ++evaluation)
        {
            solution.evaluate();
            const auto refired = perRecallOf(solution, "refiredSamples");
            mostRefired = std::max(mostRefired, std::ranges::max(refired));
            const auto& partials = solution.getParameters().partialFitness;
            REQUIRE(partials.size() == 8);
            INFO("evaluation " << evaluation);
            CHECK(partials[3] < 0.5);
            CHECK(partials[7] < 0.5);
        }
        CHECK(mostRefired >= 3.0);
    }
}

// The run stops on target only when the fitness and every term of the larger weighted sum, each recall term at its worst
// recall, exceed it (the user, 5 November 2026), so one recall that fails keeps a design short of it.
TEST_CASE("SequenceAndTimingRecall reaches its target only with a fitness above it and every recall's terms above it", "[Solutions][SequenceAndTimingRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ sequenceAndTimingRecall };
    auto solution = loadRecallFixture("reference");
    solution.evaluate();
    REQUIRE(solution.reachedTarget(0.9));
    CHECK(solution.getFitness() > 0.9);
    CHECK_FALSE(solution.reachedTarget(solution.getFitness()));
}

// Each recall's trace says when each event first peaked after the trigger, whether it came at its position, and how
// close it came to its time.
TEST_CASE("SequenceAndTimingRecall traces when each event first peaked after the trigger, its position and its timing credit", "[Solutions][SequenceAndTimingRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ sequenceAndTimingRecall };
    auto solution = loadRecallFixture("reference");
    solution.evaluate();

    const auto trials = timingTrials(solution);
    REQUIRE(trials.size() == 2);
    CHECK(trials.back().at("snapshots").size() > 50);
    const auto recalls = recallsOf(trials);
    REQUIRE(recalls.size() == recallTriggers);
    for (size_t recall = 0; recall < recalls.size(); ++recall)
    {
        const auto& events = recalls[recall].at("recall").at("events");
        for (const auto& [label, due] : recallEvents)
        {
            INFO("recall " << recall << ", " << label);
            const double afterCue = events.at(label).at("afterCue").get<double>();
            CHECK(std::abs(afterCue - due) <= 0.2 * due);
            CHECK(events.at(label).at("inPosition").get<bool>());
            CHECK(events.at(label).at("onTime").get<double>() > 0.5);
        }
        CHECK(recalls[recall].at("terms").contains("order"));
    }
}

// A saved solution carries what createPhenotypeEnvironment() adds: the judge's first schedule, triggers after waits of
// 500, 1500, 1000, 500 and 1500, so a saved design opened in dnf-composer replays one demonstration and five recalls,
// after short and long waits.
TEST_CASE("SequenceAndTimingRecall saves five triggers in turn, after 500, 1500, 1000, 500 and 1500, so a saved design replays them", "[Solutions][SequenceAndTimingRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ sequenceAndTimingRecall };
    const auto fixture = loadFixtureSimulation("sequence-and-timing-recall-reference.dnf");
    SequenceAndTimingRecall solution(sequenceTimingTopology(), *fixture);
    const auto saveDirectory = std::filesystem::temp_directory_path() / "neat-dnfs-sequence-and-timing-recall-replay";

    // As PopulationFileManager saves a solution: into <directory>/<identifier>/<identifier>.dnf.
    Solution& saved = solution;
    saved.buildPhenotype();
    saved.createPhenotypeEnvironment();
    std::filesystem::path savedFile;
    {
        const auto simulation = std::make_shared<dnf_composer::Simulation>(saved.getPhenotype());
        dnf_composer::SimulationFileManager(simulation, saveDirectory.generic_string()).saveElementsToJson();
        const std::string identifier = simulation->getUniqueIdentifier();
        savedFile = saveDirectory / identifier / (identifier + ".dnf");
    }
    saved.clearPhenotype();
    REQUIRE(std::filesystem::exists(savedFile));

    const auto replay = std::make_shared<dnf_composer::Simulation>();
    dnf_composer::SimulationFileManager(replay, savedFile.generic_string()).loadElementsFromJson();
    replay->init();
    const auto output = std::dynamic_pointer_cast<dnf_composer::element::NeuralField>(replay->getElement("nf 2"));
    REQUIRE(output != nullptr);
    // Each trigger comes its wait after the recall window before it ends (the demonstration's end, for the first).
    constexpr std::array<double, recallTriggers> cues{ 3350.0, 8150.0, 12450.0, 16250.0, 21050.0 };
    for (size_t recall = 0; recall < cues.size(); ++recall)
    {
        // When the output first peaks at each event after this trigger: the ten in their order.
        std::array<std::optional<double>, 10> firstPeaks{};
        while (replay->getT() < cues[recall] + 3300.0 - 1e-9)
        {
            replay->step();
            const double time = replay->getT();
            if (time < cues[recall])
            {
                continue;
            }
            for (const auto& bump : output->getBumps())
            {
                for (size_t event = 0; event < recallEventPositions.size(); ++event)
                {
                    if (!firstPeaks[event] && std::abs(bump.centroid - recallEventPositions[event]) <= 1.0)
                    {
                        firstPeaks[event] = time;
                    }
                }
            }
        }
        INFO("recall " << recall);
        REQUIRE(std::ranges::all_of(firstPeaks, [](const auto& time) { return time.has_value(); }));
        for (size_t event = 1; event < firstPeaks.size(); ++event)
        {
            INFO("recall " << recall << ", " << recallEvents[event].first);
            CHECK(*firstPeaks[event] > *firstPeaks[event - 1]);
        }
    }
    std::filesystem::remove_all(saveDirectory);
}

// The fitness is written from the reference's responses, so evolving it from scratch is only a fair test if the task's
// ranges reach every setting of that design.
TEST_CASE("SequenceAndTimingRecall's evolvable ranges reach every setting of the reference design", "[Solutions][SequenceAndTimingRecall]")
{
    OutputSourceRuleGuard guard;
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ sequenceAndTimingRecall };
    const auto* task = findTask(sequenceAndTimingRecall);
    REQUIRE(task != nullptr);

    const auto solution = task->makeFromTemplate(defaultTopologyFor(*task),
        *loadFixtureSimulation("sequence-and-timing-recall-reference.dnf"));
    REQUIRE(solution->getGenome().getFieldGenes().size() == 10);
    for (const auto& gene : solution->getGenome().getFieldGenes())
    {
        checkFieldGeneEvolvable(gene);
        INFO("field " << gene.getParameters().id);
        CHECK(onStep(gene.getNeuralField()->getParameters().startingRestingLevel, NeuralFieldConstants::restingLevelStep));
        const auto gauss = std::dynamic_pointer_cast<dnf_composer::element::GaussKernel>(gene.getKernel());
        REQUIRE(gauss != nullptr);
        CHECK(onStep(gauss->getParameters().width, GaussKernelConstants::widthStep));
    }
    for (const auto& gene : solution->getGenome().getConnectionGenes())
    {
        checkConnectionGeneEvolvable(gene);
    }
    // Reachable without output feedback, which the config turns on.
    GenomeMutationConstants::allowOutputAsConnectionSource = false;
    checkOnlyConnectionsEvolutionCanMake(*solution);
}

TEST_CASE("SequenceAndTimingRecall leaves the input and output fields free to evolve", "[Solutions][SequenceAndTimingRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ sequenceAndTimingRecall };
    for (const auto& fixed : { FieldGeneConstants::fixedInputField, FieldGeneConstants::fixedOutputField })
    {
        CHECK_FALSE(fixed.tau.has_value());
        CHECK_FALSE(fixed.restingLevel.has_value());
        CHECK_FALSE(fixed.gaussKernel.has_value());
    }
}

TEST_CASE("SequenceAndTimingRecall is registered with one input and one output, starting from its own template", "[SolutionRegistry]")
{
    const auto* task = findTask(sequenceAndTimingRecall);
    REQUIRE(task != nullptr);
    CHECK(task->templateFile == "sequence-and-timing-recall.dnf");
    CHECK(task->inputs == 1);
    CHECK(task->outputs == 1);
}

// The incremental-evolution test (from 2 November 2026): four events E1-E4 on fields of 100, one task per stage of
// behaviour, each stage's best seeding the next. Stage 1 recalls the order once.
using SingleOrderRecall = IncrementalRecall<RecallStage::singleOrder>;
static const std::string singleOrderRecall{ "incremental-recall-1-single-order" };

// The four events in their order, each with where it lies and when it is due after the memory recall trigger.
struct IncrementalRecallEvent
{
    std::string label;
    double position;
    double due;
};
static const std::array<IncrementalRecallEvent, 4> incrementalRecallEvents{ { { "E1", 40.0, 100.0 },
    { "E2", 90.0, 250.0 }, { "E3", 15.0, 600.0 }, { "E4", 65.0, 1200.0 } } };

static SingleOrderRecall loadSingleOrderFixture(const std::string& name)
{
    return loadSequenceTimingFixture<SingleOrderRecall>("incremental-recall-single-order-" + name + ".dnf");
}

static nlohmann::json trialOfKind(const Solution& solution, const std::string& kind)
{
    for (const auto& trial : timingTrials(solution))
    {
        if (trial.at("kind") == kind)
        {
            return trial;
        }
    }
    FAIL("no " << kind << " trial in the trace");
    return {};
}

TEST_CASE("SingleOrderRecall contract", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    checkSolutionContract<SingleOrderRecall>(makeTopology(1, 1));
}

TEST_CASE("SingleOrderRecall names its runs, and clones and copies as itself", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ singleOrderRecall };
    SingleOrderRecall solution(sequenceTimingTopology());
    solution.initialize();
    solution.buildPhenotype();

    CHECK(solution.getName() == "Incremental Recall 1 - Single Order");
    CHECK(std::dynamic_pointer_cast<SingleOrderRecall>(solution.clone()) != nullptr);
    CHECK(std::dynamic_pointer_cast<SingleOrderRecall>(solution.copy()) != nullptr);
}

// The stage-1 reference recalls the four in order, one at a time, a third faster than they were shown, and lets each go,
// leaving a trough. Below threshold its output holds the stored events, the earliest highest, so it earns the guides in
// part; it reaches 0.9 through the intended behaviour, whose terms then decide whether a run stops.
TEST_CASE("SingleOrderRecall scores the stage-1 reference above 0.9 on every behaviour term", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ singleOrderRecall };
    auto solution = loadSingleOrderFixture("reference");

    for (int evaluation = 0; evaluation < 2; ++evaluation)
    {
        solution.evaluate();
        const auto& partials = solution.getParameters().partialFitness;
        REQUIRE(partials.size() == 8);
        for (const size_t term : std::initializer_list<size_t>{ 0, 1, 3, 4, 6, 7 })
        {
            INFO("evaluation " << evaluation << ", f" << term + 1);
            CHECK(partials[term] > 0.9);
        }
        CHECK(partials[2] > 0.2); // memory: the stored events, the earliest highest
        CHECK(partials[2] < 0.9);
        CHECK(partials[5] > 0.2); // the events to come
        const auto scores = nlohmann::json::parse(solution.getParameters().evaluationTrace).at("fitness");
        CHECK(scores.at("intended").get<double>() > scores.at("guided").get<double>());
        CHECK(scores.at("intended").get<double>() > 0.9);
        CHECK(solution.getFitness() > 0.9);
        CHECK(solution.reachedTarget(0.9));
    }
}

// With no connection into the output, nothing drives a peak there, nor holds anything below threshold: the output earns
// the input and quiet terms alone.
TEST_CASE("SingleOrderRecall gives an output that never responds the input and quiet terms alone", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ singleOrderRecall };
    SingleOrderRecall solution(sequenceTimingTopology());
    solution.addFieldGene(makeFixedFieldGene(FieldGeneType::INPUT, 1));
    solution.addFieldGene(makeFixedFieldGene(FieldGeneType::OUTPUT, 2));
    solution.evaluate();

    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 8);
    CHECK(partials[1] == Catch::Approx(1.0));
    CHECK(partials[2] < 0.05);
    CHECK(partials[5] < 0.05);
    for (const size_t term : std::initializer_list<size_t>{ 3, 4, 6, 7 })
    {
        INFO("f" << term + 1);
        CHECK(partials[term] == 0.0);
    }
    CHECK(solution.getFitness() == Catch::Approx(0.15).margin(0.02));
    CHECK_FALSE(solution.reachedTarget(0.9));
}

// The best of stage-1-batch-1's R01: after the trigger each event flashes for a single sample, which the best single
// sample once paid as a full peak. A peak now earns its best three samples running, so a flash earns a third.
TEST_CASE("SingleOrderRecall pays a one-sample flash a third of a peak", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ singleOrderRecall };
    auto solution = loadSingleOrderFixture("flashes");

    for (int evaluation = 0; evaluation < 3; ++evaluation)
    {
        solution.evaluate();
        const auto& partials = solution.getParameters().partialFitness;
        REQUIRE(partials.size() == 8);
        INFO("evaluation " << evaluation);
        CHECK(partials[3] < 0.5);
        CHECK(partials[4] < 0.5);
        CHECK(solution.getFitness() < 0.6);
    }
}

// The reference with nf 6 at tau 80: each event is released later, so E1's peak is still up when E2's comes. The order
// is kept, but the peaks do not come one at a time.
TEST_CASE("SingleOrderRecall pays ordered peaks that overlap for the order, not for coming one at a time", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ singleOrderRecall };
    auto solution = loadSingleOrderFixture("overlapping-peaks");
    solution.evaluate();

    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 8);
    CHECK(partials[3] > 0.8);
    CHECK(partials[4] < 0.85);
}

// The reference with 8-5 at 2000: the stamps climb so fast that E1 and E2 peak together just after the trigger, with E3
// up by the time they go, and E4 after. E2 loses its place (a tie is no order), and the peaks are not one at a time.
TEST_CASE("SingleOrderRecall gives a burst part of the order credit and little for one at a time", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ singleOrderRecall };
    auto solution = loadSingleOrderFixture("burst");
    solution.evaluate();

    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 8);
    CHECK(partials[3] < 0.8);
    CHECK(partials[4] < 0.6);
    CHECK(solution.getFitness() < 0.7);
}

// The reference with 7-8's global part at 0.55: the pending events open the recall window by themselves as the
// demonstration ends, and it replays the sequence with no trigger. The probe, which never gives the trigger, sees it.
TEST_CASE("SingleOrderRecall gives an output that recalls without the trigger almost no recall credit", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ singleOrderRecall };
    auto solution = loadSingleOrderFixture("fires-without-boost");
    solution.evaluate();

    const auto probe = trialOfKind(solution, "probe");
    CHECK(probe.at("quietWithoutCue").get<double>() * probe.at("eventsQuietWithoutCue").get<double>() < 0.5);
    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 8);
    for (size_t term = 3; term < partials.size(); ++term)
    {
        INFO("f" << term + 1);
        CHECK(partials[term] < 0.3);
    }
    CHECK(solution.getFitness() < 0.35);
}

// The best of stage-1-batch-2's R03: after the trigger it recalls the four in order, but without it, it recalls them
// anyway from 2400 after the demonstration, past what a wait and a recall window reach. The probe runs 3200 to see it.
TEST_CASE("SingleOrderRecall gives an output that recalls without the trigger late almost no recall credit", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ singleOrderRecall };
    auto solution = loadSingleOrderFixture("recalls-late-without-trigger");
    solution.evaluate();

    const auto probe = trialOfKind(solution, "probe");
    CHECK(probe.at("eventsQuietWithoutCue").get<double>() < 0.5);
    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 8);
    CHECK(partials[3] < 0.5);
    CHECK(solution.getFitness() < 0.6);
}

// One demonstration, then three recalls, each after a wait drawn from its own third of 300-500, all three branching off
// the same run, which goes on without the trigger as the probe for 3200 after the demonstration. Every term is the mean
// over the three recalls.
TEST_CASE("SingleOrderRecall recalls after a wait from each third of 300-500 and probes the same demonstration without the trigger", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ singleOrderRecall };
    SingleOrderRecall solution(sequenceTimingTopology());
    solution.addFieldGene(makeFixedFieldGene(FieldGeneType::INPUT, 1));
    solution.addFieldGene(makeFixedFieldGene(FieldGeneType::OUTPUT, 2));

    std::vector<double> firstWaits;
    for (int evaluation = 0; evaluation < 4; ++evaluation)
    {
        solution.evaluate();
        const auto trials = timingTrials(solution);
        REQUIRE(trials.size() == 4);
        std::vector<double> quietBeforeCue;
        for (size_t trial = 0; trial < 3; ++trial)
        {
            INFO("evaluation " << evaluation << ", recall " << trial);
            REQUIRE(trials[trial].at("kind") == "recall");
            const double wait = trials[trial].at("wait").get<double>();
            const double third = 200.0 / 3.0;
            CHECK(wait >= 300.0 + third * static_cast<double>(trial) - 5.0);
            CHECK(wait <= 300.0 + third * static_cast<double>(trial + 1) + 5.0);
            CHECK(std::fmod(wait, 10.0) == Catch::Approx(0.0).margin(1e-9));
            CHECK(trials[trial].at("cue").get<double>() == Catch::Approx(1250.0 + wait));
            quietBeforeCue.push_back(trials[trial].at("quietBeforeCue").get<double>());
        }
        const auto& probe = trials[3];
        REQUIRE(probe.at("kind") == "probe");
        CHECK(probe.at("watchedFrom").get<double>() == Catch::Approx(1250.0));
        CHECK(probe.at("length").get<double>() == Catch::Approx(3200.0));
        CHECK(solution.getParameters().partialFitness[1] == Catch::Approx(
            std::accumulate(quietBeforeCue.begin(), quietBeforeCue.end(), 0.0) / 3.0));
        firstWaits.push_back(trials[0].at("wait").get<double>());
    }
    CHECK(std::ranges::adjacent_find(firstWaits, std::not_equal_to{}) != firstWaits.end());
}

// Each recall's trace says when each event first peaked after the trigger, its place in the order, and how far it peaked
// alone.
TEST_CASE("SingleOrderRecall traces each event's first peak after the trigger, its place and how far it was alone", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ singleOrderRecall };
    auto solution = loadSingleOrderFixture("reference");
    solution.evaluate();

    const auto recall = trialOfKind(solution, "recall");
    const auto& events = recall.at("recall").at("events");
    double previous = 0.0;
    for (const auto& event : incrementalRecallEvents)
    {
        INFO(event.label);
        const double afterCue = events.at(event.label).at("afterCue").get<double>();
        CHECK(afterCue > previous);
        CHECK(events.at(event.label).at("inPlace").get<double>() == Catch::Approx(1.0));
        CHECK(events.at(event.label).at("alone").get<double>() == Catch::Approx(1.0));
        CHECK_FALSE(events.at(event.label).at("releasedAt").is_null());
        previous = afterCue;
    }
}

// A saved solution carries what createPhenotypeEnvironment() adds: the demonstration and the trigger after a wait of 300,
// where stage 1 recalls, so a saved design opened in dnf-composer replays it.
TEST_CASE("SingleOrderRecall saves a recall after a wait of 300, so a saved design replays it", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ singleOrderRecall };
    const auto fixture = loadFixtureSimulation("incremental-recall-single-order-reference.dnf");
    SingleOrderRecall solution(sequenceTimingTopology(), *fixture);
    const auto saveDirectory = std::filesystem::temp_directory_path() / "neat-dnfs-incremental-recall-replay";

    Solution& saved = solution;
    saved.buildPhenotype();
    saved.createPhenotypeEnvironment();
    std::filesystem::path savedFile;
    {
        const auto simulation = std::make_shared<dnf_composer::Simulation>(saved.getPhenotype());
        dnf_composer::SimulationFileManager(simulation, saveDirectory.generic_string()).saveElementsToJson();
        const std::string identifier = simulation->getUniqueIdentifier();
        savedFile = saveDirectory / identifier / (identifier + ".dnf");
    }
    saved.clearPhenotype();
    REQUIRE(std::filesystem::exists(savedFile));

    const auto replay = std::make_shared<dnf_composer::Simulation>();
    dnf_composer::SimulationFileManager(replay, savedFile.generic_string()).loadElementsFromJson();
    replay->init();
    const auto output = std::dynamic_pointer_cast<dnf_composer::element::NeuralField>(replay->getElement("nf 2"));
    REQUIRE(output != nullptr);
    // The trigger comes again at 1550; the reference's peaks are up 100, 190, 440 and 870 after it.
    const std::array<double, 4> upAfterCue{ 100.0, 190.0, 440.0, 870.0 };
    for (size_t event = 0; event < incrementalRecallEvents.size(); ++event)
    {
        const double time = 1550.0 + upAfterCue[event];
        while (replay->getT() < time - 1e-9)
        {
            replay->step();
        }
        const auto bumps = output->getBumps();
        INFO("t " << time << ", " << incrementalRecallEvents[event].label);
        REQUIRE(bumps.size() == 1);
        CHECK(std::abs(bumps.front().centroid - incrementalRecallEvents[event].position) <= 1.0);
    }
    std::filesystem::remove_all(saveDirectory);
}

// The fitness is checked against the reference's behaviour, so evolving it is only a fair test if the task's ranges
// reach every setting of that design.
TEST_CASE("SingleOrderRecall's evolvable ranges reach every setting of the stage-1 reference", "[Solutions][IncrementalRecall]")
{
    OutputSourceRuleGuard guard;
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ singleOrderRecall };
    const auto* task = findTask(singleOrderRecall);
    REQUIRE(task != nullptr);

    const auto solution = task->makeFromTemplate(defaultTopologyFor(*task),
        *loadFixtureSimulation("incremental-recall-single-order-reference.dnf"));
    REQUIRE(solution->getGenome().getFieldGenes().size() == 9);
    for (const auto& gene : solution->getGenome().getFieldGenes())
    {
        checkFieldGeneEvolvable(gene);
    }
    for (const auto& gene : solution->getGenome().getConnectionGenes())
    {
        checkConnectionGeneEvolvable(gene);
    }
    GenomeMutationConstants::allowOutputAsConnectionSource = false;
    checkOnlyConnectionsEvolutionCanMake(*solution);
}

TEST_CASE("SingleOrderRecall is registered with one input and one output, starting from the incremental-recall template", "[SolutionRegistry]")
{
    const auto* task = findTask(singleOrderRecall);
    REQUIRE(task != nullptr);
    CHECK(task->templateFile == "incremental-recall.dnf");
    CHECK(task->inputs == 1);
    CHECK(task->outputs == 1);
}

// Stage 2 recalls the order once, each event at its demonstrated time after the trigger: stage 1's terms and a timing
// term, f6.
using SingleOrderAndTimingRecall = IncrementalRecall<RecallStage::singleOrderAndTiming>;
static const std::string singleOrderAndTimingRecall{ "incremental-recall-2-single-order-and-timing" };

TEST_CASE("SingleOrderAndTimingRecall contract", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    checkSolutionContract<SingleOrderAndTimingRecall>(makeTopology(1, 1));
}

TEST_CASE("SingleOrderAndTimingRecall names its runs, and clones and copies as itself", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ singleOrderAndTimingRecall };
    SingleOrderAndTimingRecall solution(sequenceTimingTopology());
    solution.initialize();
    solution.buildPhenotype();

    CHECK(solution.getName() == "Incremental Recall 2 - Single Order And Timing");
    CHECK(std::dynamic_pointer_cast<SingleOrderAndTimingRecall>(solution.clone()) != nullptr);
    CHECK(std::dynamic_pointer_cast<SingleOrderAndTimingRecall>(solution.copy()) != nullptr);
}

// The stage-2 reference recalls the four in order, one at a time, each at its demonstrated time after the trigger.
TEST_CASE("SingleOrderAndTimingRecall scores the stage-2 reference above 0.9 on every behaviour term", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ singleOrderAndTimingRecall };
    auto solution = loadSequenceTimingFixture<SingleOrderAndTimingRecall>(
        "incremental-recall-single-order-and-timing-reference.dnf");

    for (int evaluation = 0; evaluation < 2; ++evaluation)
    {
        solution.evaluate();
        const auto& partials = solution.getParameters().partialFitness;
        REQUIRE(partials.size() == 9);
        for (const size_t term : std::initializer_list<size_t>{ 0, 1, 3, 4, 5, 7, 8 })
        {
            INFO("evaluation " << evaluation << ", f" << term + 1);
            CHECK(partials[term] > 0.9);
        }
        const auto scores = nlohmann::json::parse(solution.getParameters().evaluationTrace).at("fitness");
        CHECK(scores.at("intended").get<double>() > 0.9);
        CHECK(solution.getFitness() > 0.9);
        CHECK(solution.reachedTarget(0.9));
    }
}

// The stage-1 reference recalls the order a third faster than it was shown: in order, but each event a quarter early.
TEST_CASE("SingleOrderAndTimingRecall pays the stage-1 reference's order in full and its timing in part", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ singleOrderAndTimingRecall };
    auto solution = loadSequenceTimingFixture<SingleOrderAndTimingRecall>("incremental-recall-single-order-reference.dnf");
    solution.evaluate();

    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 9);
    CHECK(partials[3] > 0.9);
    CHECK(partials[5] > 0.2);
    CHECK(partials[5] < 0.7);
    CHECK(solution.getFitness() < 0.9);
    CHECK_FALSE(solution.reachedTarget(0.9));
}

// The best of stage-2-batch-1's R04: it recalls the four in order and on time, but about 1900 after the trigger
// activity comes back at both ends of the field and travels along it. Silence after a recall is watched as long as
// the probe watches without the trigger, so it is seen.
TEST_CASE("SingleOrderAndTimingRecall charges activity that comes back long after the recall to the stop", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ singleOrderAndTimingRecall };
    auto solution = loadSequenceTimingFixture<SingleOrderAndTimingRecall>(
        "incremental-recall-single-order-and-timing-restarts-after-recall.dnf");
    solution.evaluate();

    for (const auto& trial : timingTrials(solution))
    {
        if (trial.at("kind") == "recall")
        {
            INFO("wait " << trial.at("wait"));
            CHECK(trial.at("recall").at("afterSequence").at("quiet").get<double>() < 0.8);
        }
    }
    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 9);
    CHECK(partials[5] > 0.85);
    CHECK(partials[8] < 0.9);
    CHECK_FALSE(solution.reachedTarget(0.9));
}

TEST_CASE("SingleOrderAndTimingRecall gives an output that never responds the input and quiet terms alone", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ singleOrderAndTimingRecall };
    SingleOrderAndTimingRecall solution(sequenceTimingTopology());
    solution.addFieldGene(makeFixedFieldGene(FieldGeneType::INPUT, 1));
    solution.addFieldGene(makeFixedFieldGene(FieldGeneType::OUTPUT, 2));
    solution.evaluate();

    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 9);
    for (const size_t term : std::initializer_list<size_t>{ 3, 4, 5, 7, 8 })
    {
        INFO("f" << term + 1);
        CHECK(partials[term] == 0.0);
    }
    CHECK(solution.getFitness() == Catch::Approx(0.10).margin(0.02));
    CHECK_FALSE(solution.reachedTarget(0.9));
}

TEST_CASE("SingleOrderAndTimingRecall is registered with one input and one output, starting from the incremental-recall template", "[SolutionRegistry]")
{
    const auto* task = findTask(singleOrderAndTimingRecall);
    REQUIRE(task != nullptr);
    CHECK(task->templateFile == "incremental-recall.dnf");
    CHECK(task->inputs == 1);
    CHECK(task->outputs == 1);
}

using RepeatableOrderRecall = IncrementalRecall<RecallStage::repeatableOrder>;
static const std::string repeatableOrderRecall{ "incremental-recall-3-repeatable-order" };

TEST_CASE("RepeatableOrderRecall contract", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    checkSolutionContract<RepeatableOrderRecall>(makeTopology(1, 1));
}

TEST_CASE("RepeatableOrderRecall names its runs, and clones and copies as itself", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ repeatableOrderRecall };
    RepeatableOrderRecall solution(sequenceTimingTopology());
    solution.initialize();
    solution.buildPhenotype();

    CHECK(solution.getName() == "Incremental Recall 3 - Repeatable Order");
    CHECK(std::dynamic_pointer_cast<RepeatableOrderRecall>(solution.clone()) != nullptr);
    CHECK(std::dynamic_pointer_cast<RepeatableOrderRecall>(solution.copy()) != nullptr);
}

// The stage-3 reference holds its memories through the longest wait: it recalls the four in order after any of them.
TEST_CASE("RepeatableOrderRecall scores the stage-3 reference above 0.9 on every behaviour term", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ repeatableOrderRecall };
    auto solution = loadSequenceTimingFixture<RepeatableOrderRecall>("incremental-recall-repeatable-order-reference.dnf");

    for (int evaluation = 0; evaluation < 2; ++evaluation)
    {
        solution.evaluate();
        const auto& partials = solution.getParameters().partialFitness;
        REQUIRE(partials.size() == 8);
        for (const size_t term : std::initializer_list<size_t>{ 0, 1, 3, 4, 6, 7 })
        {
            INFO("evaluation " << evaluation << ", f" << term + 1);
            CHECK(partials[term] > 0.9);
        }
        CHECK(solution.getFitness() > 0.9);
        CHECK(solution.reachedTarget(0.9));
    }
}

// The stage-2 reference's memories fade about 1800 after their event: it recalls after a short wait, not a long one.
TEST_CASE("RepeatableOrderRecall pays the stage-2 reference only for its recall after the shortest waits", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ repeatableOrderRecall };
    auto solution = loadSequenceTimingFixture<RepeatableOrderRecall>(
        "incremental-recall-single-order-and-timing-reference.dnf");
    solution.evaluate();

    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 8);
    CHECK(partials[3] < 0.7);
    CHECK(solution.getFitness() < 0.8);
    CHECK_FALSE(solution.reachedTarget(0.9));
}

// The three waits now span 300-1500, from the shortest wait of stage 1 to the longest the probe was built to cover.
TEST_CASE("RepeatableOrderRecall recalls after a wait from each third of 300-1500", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ repeatableOrderRecall };
    RepeatableOrderRecall solution(sequenceTimingTopology());
    solution.addFieldGene(makeFixedFieldGene(FieldGeneType::INPUT, 1));
    solution.addFieldGene(makeFixedFieldGene(FieldGeneType::OUTPUT, 2));

    for (int evaluation = 0; evaluation < 4; ++evaluation)
    {
        solution.evaluate();
        const auto trials = timingTrials(solution);
        REQUIRE(trials.size() == 4);
        for (size_t trial = 0; trial < 3; ++trial)
        {
            INFO("evaluation " << evaluation << ", recall " << trial);
            REQUIRE(trials[trial].at("kind") == "recall");
            const double wait = trials[trial].at("wait").get<double>();
            CHECK(wait >= 300.0 + 400.0 * static_cast<double>(trial) - 5.0);
            CHECK(wait <= 300.0 + 400.0 * static_cast<double>(trial + 1) + 5.0);
        }
        CHECK(trials[3].at("length").get<double>() == Catch::Approx(3200.0));
    }
}

// Stage 3 is about memories that hold, so a saved design replays the trigger after the longest wait, 1500.
TEST_CASE("RepeatableOrderRecall saves a recall after a wait of 1500, so a saved design replays it", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ repeatableOrderRecall };
    const auto fixture = loadFixtureSimulation("incremental-recall-repeatable-order-reference.dnf");
    RepeatableOrderRecall solution(sequenceTimingTopology(), *fixture);
    const auto saveDirectory = std::filesystem::temp_directory_path() / "neat-dnfs-incremental-recall-3-replay";

    Solution& saved = solution;
    saved.buildPhenotype();
    saved.createPhenotypeEnvironment();
    std::filesystem::path savedFile;
    {
        const auto simulation = std::make_shared<dnf_composer::Simulation>(saved.getPhenotype());
        dnf_composer::SimulationFileManager(simulation, saveDirectory.generic_string()).saveElementsToJson();
        const std::string identifier = simulation->getUniqueIdentifier();
        savedFile = saveDirectory / identifier / (identifier + ".dnf");
    }
    saved.clearPhenotype();
    REQUIRE(std::filesystem::exists(savedFile));

    const auto replay = std::make_shared<dnf_composer::Simulation>();
    dnf_composer::SimulationFileManager(replay, savedFile.generic_string()).loadElementsFromJson();
    replay->init();
    const auto output = std::dynamic_pointer_cast<dnf_composer::element::NeuralField>(replay->getElement("nf 2"));
    REQUIRE(output != nullptr);
    // The trigger comes again at 2750; the reference's peaks are up 230, 380, 730 and 1320 after it.
    const std::array<double, 4> upAfterCue{ 230.0, 380.0, 730.0, 1320.0 };
    for (size_t event = 0; event < incrementalRecallEvents.size(); ++event)
    {
        const double time = 2750.0 + upAfterCue[event];
        while (replay->getT() < time - 1e-9)
        {
            replay->step();
        }
        const auto bumps = output->getBumps();
        INFO("t " << time << ", " << incrementalRecallEvents[event].label);
        REQUIRE(bumps.size() == 1);
        CHECK(std::abs(bumps.front().centroid - incrementalRecallEvents[event].position) <= 1.0);
    }
    std::filesystem::remove_all(saveDirectory);
}

TEST_CASE("RepeatableOrderRecall is registered with one input and one output, starting from the incremental-recall template", "[SolutionRegistry]")
{
    const auto* task = findTask(repeatableOrderRecall);
    REQUIRE(task != nullptr);
    CHECK(task->templateFile == "incremental-recall.dnf");
    CHECK(task->inputs == 1);
    CHECK(task->outputs == 1);
}

using RepeatableOrderAndTimingRecall = IncrementalRecall<RecallStage::repeatableOrderAndTiming>;
static const std::string repeatableOrderAndTimingRecall{ "incremental-recall-4-repeatable-order-and-timing" };

TEST_CASE("RepeatableOrderAndTimingRecall contract", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    checkSolutionContract<RepeatableOrderAndTimingRecall>(makeTopology(1, 1));
}

TEST_CASE("RepeatableOrderAndTimingRecall names its runs, and clones and copies as itself", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ repeatableOrderAndTimingRecall };
    RepeatableOrderAndTimingRecall solution(sequenceTimingTopology());
    solution.initialize();
    solution.buildPhenotype();

    CHECK(solution.getName() == "Incremental Recall 4 - Repeatable Order And Timing");
    CHECK(std::dynamic_pointer_cast<RepeatableOrderAndTimingRecall>(solution.clone()) != nullptr);
    CHECK(std::dynamic_pointer_cast<RepeatableOrderAndTimingRecall>(solution.copy()) != nullptr);
}

// The stage-4 reference's stamps hold through the longest wait: each event at its time after any of them.
TEST_CASE("RepeatableOrderAndTimingRecall scores the stage-4 reference above 0.9 on every behaviour term", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ repeatableOrderAndTimingRecall };
    auto solution = loadSequenceTimingFixture<RepeatableOrderAndTimingRecall>(
        "incremental-recall-repeatable-order-and-timing-reference.dnf");

    for (int evaluation = 0; evaluation < 2; ++evaluation)
    {
        solution.evaluate();
        const auto& partials = solution.getParameters().partialFitness;
        REQUIRE(partials.size() == 9);
        for (const size_t term : std::initializer_list<size_t>{ 0, 1, 3, 4, 5, 7, 8 })
        {
            INFO("evaluation " << evaluation << ", f" << term + 1);
            CHECK(partials[term] > 0.9);
        }
        CHECK(solution.getFitness() > 0.9);
        CHECK(solution.reachedTarget(0.9));
    }
}

// The stage-3 reference's stamps leak through the wait: on time after a short wait, later and later after longer ones.
TEST_CASE("RepeatableOrderAndTimingRecall pays the stage-3 reference's order in full and its timing in part", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ repeatableOrderAndTimingRecall };
    auto solution = loadSequenceTimingFixture<RepeatableOrderAndTimingRecall>(
        "incremental-recall-repeatable-order-reference.dnf");
    solution.evaluate();

    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 9);
    CHECK(partials[3] > 0.9);
    CHECK(partials[5] < 0.9);
    CHECK_FALSE(solution.reachedTarget(0.9));
}

TEST_CASE("RepeatableOrderAndTimingRecall recalls after a wait from each third of 300-1500", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ repeatableOrderAndTimingRecall };
    RepeatableOrderAndTimingRecall solution(sequenceTimingTopology());
    solution.addFieldGene(makeFixedFieldGene(FieldGeneType::INPUT, 1));
    solution.addFieldGene(makeFixedFieldGene(FieldGeneType::OUTPUT, 2));
    solution.evaluate();

    const auto trials = timingTrials(solution);
    REQUIRE(trials.size() == 4);
    for (size_t trial = 0; trial < 3; ++trial)
    {
        INFO("recall " << trial);
        const double wait = trials[trial].at("wait").get<double>();
        CHECK(wait >= 300.0 + 400.0 * static_cast<double>(trial) - 5.0);
        CHECK(wait <= 300.0 + 400.0 * static_cast<double>(trial + 1) + 5.0);
    }
    CHECK(solution.getParameters().partialFitness.size() == 9);
}

TEST_CASE("RepeatableOrderAndTimingRecall is registered with one input and one output, starting from the incremental-recall template", "[SolutionRegistry]")
{
    const auto* task = findTask(repeatableOrderAndTimingRecall);
    REQUIRE(task != nullptr);
    CHECK(task->templateFile == "incremental-recall.dnf");
    CHECK(task->inputs == 1);
    CHECK(task->outputs == 1);
}

using ConsecutiveOrderRecall = IncrementalRecall<RecallStage::consecutiveOrder>;
static const std::string consecutiveOrderRecall{ "incremental-recall-5-consecutive-order" };

TEST_CASE("ConsecutiveOrderRecall contract", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    checkSolutionContract<ConsecutiveOrderRecall>(makeTopology(1, 1));
}

TEST_CASE("ConsecutiveOrderRecall names its runs, and clones and copies as itself", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ consecutiveOrderRecall };
    ConsecutiveOrderRecall solution(sequenceTimingTopology());
    solution.initialize();
    solution.buildPhenotype();

    CHECK(solution.getName() == "Incremental Recall 5 - Consecutive Order");
    CHECK(std::dynamic_pointer_cast<ConsecutiveOrderRecall>(solution.clone()) != nullptr);
    CHECK(std::dynamic_pointer_cast<ConsecutiveOrderRecall>(solution.copy()) != nullptr);
}

// The stage-5 reference winds back after each recall, so the trigger recalls the order again, and again. Its wind-back
// leaves shallower troughs than the guides ask (as at every stage), so released and stopped fall a little short with
// them; the intended behaviour, without the guides, passes the target.
TEST_CASE("ConsecutiveOrderRecall scores the stage-5 reference's intended behaviour above 0.9", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ consecutiveOrderRecall };
    auto solution = loadSequenceTimingFixture<ConsecutiveOrderRecall>("incremental-recall-consecutive-order-reference.dnf");

    for (int evaluation = 0; evaluation < 2; ++evaluation)
    {
        solution.evaluate();
        const auto& partials = solution.getParameters().partialFitness;
        REQUIRE(partials.size() == 8);
        for (const size_t term : std::initializer_list<size_t>{ 0, 1, 3, 4 })
        {
            INFO("evaluation " << evaluation << ", f" << term + 1);
            CHECK(partials[term] > 0.9);
        }
        CHECK(partials[6] > 0.8);
        CHECK(partials[7] > 0.8);
        const auto scores = nlohmann::json::parse(solution.getParameters().evaluationTrace).at("fitness");
        CHECK(scores.at("intended").get<double>() > 0.9);
        CHECK(solution.getFitness() > 0.9);
        CHECK(solution.reachedTarget(0.9));
    }
}

// The stage-4 reference's stamps are used up by its first recall: a second trigger fires all four together.
TEST_CASE("ConsecutiveOrderRecall pays the stage-4 reference for its first recall only", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ consecutiveOrderRecall };
    auto solution = loadSequenceTimingFixture<ConsecutiveOrderRecall>(
        "incremental-recall-repeatable-order-and-timing-reference.dnf");
    solution.evaluate();

    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 8);
    CHECK(partials[3] < 0.7);
    CHECK(solution.getFitness() < 0.9);
    CHECK_FALSE(solution.reachedTarget(0.9));
}

// R02 of stage-5-batch-1 stopped on its target, yet after its first recall its output holds one weak bump that rolls
// slowly round the field, over E2, E3 and E4 in turn. It moves about one unit a sample, which a drift read from one
// sample to the next takes for still; over three samples it moves three or more, so it peaks at no event it passes.
TEST_CASE("ConsecutiveOrderRecall counts no peak at an event a bump rolls past", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ consecutiveOrderRecall };
    auto solution = loadSequenceTimingFixture<ConsecutiveOrderRecall>(
        "incremental-recall-consecutive-order-rolling-bump.dnf");

    for (int evaluation = 0; evaluation < 5; ++evaluation)
    {
        solution.evaluate();
        const auto& partials = solution.getParameters().partialFitness;
        REQUIRE(partials.size() == 8);
        INFO("evaluation " << evaluation);
        // E1 peaks in every recall, and the bump stops at E4 or E3 in some; the rest it rolls past.
        CHECK(partials[3] < 0.8);
        CHECK_FALSE(solution.reachedTarget(0.9));
    }
}

// Six triggers in turn in one run after the demonstration.
static constexpr size_t consecutiveRecalls = 6;

// One value of each recall trial's evaluation trace, in turn.
static std::vector<double> perRecall(const Solution& solution, const std::string& key)
{
    std::vector<double> values;
    const auto trace = nlohmann::json::parse(solution.getParameters().evaluationTrace);
    for (const auto& trial : trace.at("trials"))
    {
        if (trial.at("kind") == "recall")
        {
            values.push_back(trial.at(key).get<double>());
        }
    }
    return values;
}

// One term each recall trial earned, as its evaluation trace records it, in turn.
static std::vector<double> perRecallTerm(const Solution& solution, const std::string& term)
{
    std::vector<double> values;
    const auto trace = nlohmann::json::parse(solution.getParameters().evaluationTrace);
    for (const auto& trial : trace.at("trials"))
    {
        if (trial.at("kind") == "recall")
        {
            values.push_back(trial.at("terms").at(term).get<double>());
        }
    }
    return values;
}

// How far, at each recall's samples with a bump on the output until the sequence is released, every bump stood still at
// an event: a bump standing between the events, or rolling past them, costs one at a time.
static std::vector<double> stillAtEventsPerRecall(const Solution& solution)
{
    return perRecall(solution, "stillAtEvents");
}

TEST_CASE("ConsecutiveOrderRecall traces how far the output's bumps stood still at events in each recall", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ consecutiveOrderRecall };

    SECTION("the stage-5 reference's bumps stand still at events in every recall")
    {
        auto solution = loadSequenceTimingFixture<ConsecutiveOrderRecall>(
            "incremental-recall-consecutive-order-reference.dnf");
        solution.evaluate();
        const auto shares = stillAtEventsPerRecall(solution);
        REQUIRE(shares.size() == consecutiveRecalls);
        for (const double share : shares)
        {
            CHECK(share > 0.95);
        }
    }

    SECTION("the rolling bump's stand still at events in the first recall only")
    {
        auto solution = loadSequenceTimingFixture<ConsecutiveOrderRecall>(
            "incremental-recall-consecutive-order-rolling-bump.dnf");
        // Later its still samples are E1's peak and any stop the bump makes at an event, as each draw of waits allows.
        std::vector<double> later;
        for (int evaluation = 0; evaluation < 5; ++evaluation)
        {
            solution.evaluate();
            const auto shares = stillAtEventsPerRecall(solution);
            REQUIRE(shares.size() == consecutiveRecalls);
            for (size_t recall = 0; recall < shares.size(); ++recall)
            {
                INFO("evaluation " << evaluation << ", recall " << recall);
                if (recall == 0)
                {
                    CHECK(shares[recall] > 0.95);
                }
                else
                {
                    later.push_back(shares[recall]);
                }
            }
        }
        CHECK(std::accumulate(later.begin(), later.end(), 0.0) / static_cast<double>(later.size()) < 0.6);
    }
}

// R02 of stage-5-batch-2 stopped on its target, yet at its handovers the event before is still up as the next comes, for
// a sample or three: two peaks up together, which one peak at a time forbids. A peak's credit fell only by its share of
// samples not alone, so such a recall still earned 0.85-0.95. Each sample two peaks are up together now costs a third
// of the recall's one-at-a-time credit, so one such sample in any recall keeps the term below the target.
TEST_CASE("ConsecutiveOrderRecall's one at a time loses a third of a recall's credit for each sample two peaks are up together", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ consecutiveOrderRecall };

    SECTION("the stage-5 reference's peaks never share the output")
    {
        auto solution = loadSequenceTimingFixture<ConsecutiveOrderRecall>(
            "incremental-recall-consecutive-order-reference.dnf");
        for (int evaluation = 0; evaluation < 3; ++evaluation)
        {
            solution.evaluate();
            const auto shared = perRecall(solution, "sharedSamples");
            REQUIRE(shared.size() == consecutiveRecalls);
            INFO("evaluation " << evaluation);
            CHECK(std::ranges::all_of(shared, [](const double samples) { return samples == 0.0; }));
            CHECK(solution.reachedTarget(0.9));
        }
    }

    SECTION("overlapping handovers never reach the target")
    {
        auto solution = loadSequenceTimingFixture<ConsecutiveOrderRecall>(
            "incremental-recall-consecutive-order-overlapping-handovers.dnf");
        double mostShared = 0.0;
        for (int evaluation = 0; evaluation < 6; ++evaluation)
        {
            solution.evaluate();
            const auto shared = perRecall(solution, "sharedSamples");
            REQUIRE(shared.size() == consecutiveRecalls);
            mostShared = std::max(mostShared, std::ranges::max(shared));
            const auto oneAtATime = perRecallTerm(solution, "oneAtATime");
            REQUIRE(oneAtATime.size() == consecutiveRecalls);
            for (size_t recall = 0; recall < shared.size(); ++recall)
            {
                INFO("evaluation " << evaluation << ", recall " << recall << ", shared " << shared[recall]);
                CHECK(oneAtATime[recall] <= std::max(0.0, 1.0 - shared[recall] / 3.0) + 0.005);
            }
            INFO("evaluation " << evaluation);
            CHECK_FALSE(solution.reachedTarget(0.9));
        }
        CHECK(mostShared > 0.0);
    }
}

// The recall trials of one evaluation, in turn.
static std::vector<nlohmann::json> recallTrials(const Solution& solution)
{
    std::vector<nlohmann::json> recalls;
    for (const auto& trial : timingTrials(solution))
    {
        if (trial.at("kind") == "recall")
        {
            recalls.push_back(trial);
        }
    }
    return recalls;
}

// Each trigger's wait in turn.
static std::vector<double> waitsInTurn(const Solution& solution)
{
    std::vector<double> waits;
    for (const auto& recall : recallTrials(solution))
    {
        waits.push_back(recall.at("wait").get<double>());
    }
    return waits;
}

// Which of the two sequences of six triggers @p waits follow: 0 for 300, M1, 1500, 300, 1500, M2, and 1 for 1500, 300,
// M1, 1500, M2, 300, with M1 from 700-900 and M2 from 900-1100, each on the sampling grid; nothing if neither.
static std::optional<size_t> consecutiveSequenceOf(const std::vector<double>& waits)
{
    constexpr std::array<std::array<char, 6>, 2> sequences{ { { 'S', '1', 'L', 'S', 'L', '2' },
        { 'L', 'S', '1', 'L', '2', 'S' } } };
    const auto fits = [](const char wait, const double value) {
        const bool onGrid = std::abs(std::fmod(value, 10.0)) < 1e-9;
        switch (wait)
        {
        case 'S': return std::abs(value - 300.0) < 1e-9;
        case 'L': return std::abs(value - 1500.0) < 1e-9;
        case '1': return onGrid && value >= 700.0 && value <= 900.0;
        default: return onGrid && value >= 900.0 && value <= 1100.0;
        }
    };
    for (size_t sequence = 0; sequence < sequences.size(); ++sequence)
    {
        if (waits.size() == sequences[sequence].size()
            && std::ranges::equal(sequences[sequence], waits, fits))
        {
            return sequence;
        }
    }
    return std::nullopt;
}

// R01 of stage-5-batch-3 recalled in turn only when each wait was shorter than the one before: one evaluation scored
// one order of the waits, so it scored 0.999 on that order and 0.6-0.93 on the others, and evolution kept the lucky
// draws. R01 of stage-5-batch-4 then recalled out of order whenever the first trigger came 460 or less after the
// demonstration. Both were caught by three orders of three waits, each from one demonstration.
// From 5 November 2026 (the user: one demonstration, then recall as many times as asked) one run recalls after six
// triggers in turn: each of 300 and 1500 twice, and two middle waits, one from 700-900 and one from 900-1100, in one of
// two sequences. Each wait comes once among the first three triggers and once among the last three, and between the
// two sequences every wait follows every other.
TEST_CASE("ConsecutiveOrderRecall recalls after six triggers in turn in one run, each wait once early and once late", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ consecutiveOrderRecall };
    ConsecutiveOrderRecall solution(sequenceTimingTopology());
    solution.addFieldGene(makeFixedFieldGene(FieldGeneType::INPUT, 1));
    solution.addFieldGene(makeFixedFieldGene(FieldGeneType::OUTPUT, 2));

    std::set<size_t> sequencesSeen;
    std::set<double> middleWaits;
    for (int evaluation = 0; evaluation < 8; ++evaluation)
    {
        solution.evaluate();
        const auto trials = timingTrials(solution);
        REQUIRE(trials.size() == consecutiveRecalls + 1);
        CHECK(trials.back().at("kind") == "probe");
        const auto recalls = recallTrials(solution);
        REQUIRE(recalls.size() == consecutiveRecalls);

        const auto waits = waitsInTurn(solution);
        const auto sequence = consecutiveSequenceOf(waits);
        INFO("evaluation " << evaluation);
        REQUIRE(sequence.has_value());
        sequencesSeen.insert(*sequence);
        middleWaits.insert(waits.begin(), waits.end());

        // The run branches off the demonstration after its first wait, and each later trigger comes its wait after the
        // recall window before it.
        double windowEnd = 1250.0;
        for (size_t trigger = 0; trigger < recalls.size(); ++trigger)
        {
            INFO("evaluation " << evaluation << ", trigger " << trigger);
            CHECK(recalls[trigger].at("cue").get<double>() == Catch::Approx(windowEnd + waits[trigger]));
            windowEnd = recalls[trigger].at("cue").get<double>() + 1700.0;
        }
    }
    // Either sequence, drawn afresh each evaluation, and middle waits drawn afresh too.
    CHECK(sequencesSeen.size() == 2);
    CHECK(middleWaits.size() > 4);
}

// A design that recalls in turn after some sequences of the waits only fails the others. Confirmations take the two
// sequences in turn, so any two together replay both.
TEST_CASE("ConsecutiveOrderRecall's confirmations of a reached target take the two sequences of triggers in turn", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ consecutiveOrderRecall };
    CHECK(PopulationConstants::targetConfirmationEvaluations == 6);

    for (size_t confirmation = 0; confirmation < 4; ++confirmation)
    {
        ConsecutiveOrderRecall solution(sequenceTimingTopology());
        solution.addFieldGene(makeFixedFieldGene(FieldGeneType::INPUT, 1));
        solution.addFieldGene(makeFixedFieldGene(FieldGeneType::OUTPUT, 2));
        solution.prepareTargetConfirmation(confirmation);
        solution.evaluate();
        INFO("confirmation " << confirmation);
        const auto sequence = consecutiveSequenceOf(waitsInTurn(solution));
        REQUIRE(sequence.has_value());
        CHECK(*sequence == confirmation % 2);
    }
}

// Each event's place, from when every event first peaked: the share of the other three on the right side of it.
static double placeAmongAll(const std::array<std::optional<double>, 4>& firstPeaks, const size_t event)
{
    if (!firstPeaks[event])
    {
        return 0.0;
    }
    double right = 0.0;
    for (size_t other = 0; other < firstPeaks.size(); ++other)
    {
        if (other < event)
        {
            right += firstPeaks[other] && *firstPeaks[other] < *firstPeaks[event] ? 1.0 : 0.0;
        }
        else if (other > event)
        {
            right += !firstPeaks[other] || *firstPeaks[other] > *firstPeaks[event] ? 1.0 : 0.0;
        }
    }
    return right / 3.0;
}

// A recall that always ended E4, E3 kept E3 in full, as E1 and E2 came before it, and so kept 0.92 of its credit, past
// the target. From stage 5 each event's place is the share of the other three on the right side of it, so both events
// of a swapped pair lose a third. R03 of stage-5-batch-3 recalls events out of turn in most of its recalls.
TEST_CASE("ConsecutiveOrderRecall places each event among all the others, so both events of a swapped pair lose a third", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ consecutiveOrderRecall };
    auto solution = loadSequenceTimingFixture<ConsecutiveOrderRecall>("incremental-recall-consecutive-order-refires.dnf");

    size_t outOfPlace = 0;
    for (int evaluation = 0; evaluation < 2; ++evaluation)
    {
        solution.evaluate();
        const auto recalls = recallTrials(solution);
        REQUIRE(recalls.size() == consecutiveRecalls);
        for (size_t recall = 0; recall < recalls.size(); ++recall)
        {
            const auto& events = recalls[recall].at("recall").at("events");
            std::array<std::optional<double>, 4> firstPeaks;
            for (size_t event = 0; event < incrementalRecallEvents.size(); ++event)
            {
                const auto& firstPeak = events.at(incrementalRecallEvents[event].label).at("firstPeak");
                if (!firstPeak.is_null())
                {
                    firstPeaks[event] = firstPeak.get<double>();
                }
            }
            for (size_t event = 0; event < incrementalRecallEvents.size(); ++event)
            {
                INFO("evaluation " << evaluation << ", recall " << recall << ", " << incrementalRecallEvents[event].label);
                const double place = placeAmongAll(firstPeaks, event);
                CHECK(events.at(incrementalRecallEvents[event].label).at("inPlace").get<double>()
                    == Catch::Approx(place).margin(0.005));
                outOfPlace += place < 1.0 ? 1 : 0;
            }
        }
    }
    CHECK(outOfPlace > 0);
}

// The target asks each recall to pass, not the mean over them, so one recall out of order or overlapping keeps a design
// short of it however well the others go.
TEST_CASE("ConsecutiveOrderRecall reaches its target only if every recall passes it", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ consecutiveOrderRecall };

    SECTION("every recall of the stage-5 reference passes")
    {
        auto solution = loadSequenceTimingFixture<ConsecutiveOrderRecall>(
            "incremental-recall-consecutive-order-reference.dnf");
        solution.evaluate();
        for (const std::string term : { "recalled", "oneAtATime", "releasedWithoutGuide", "stoppedWithoutGuide" })
        {
            const auto values = perRecallTerm(solution, term);
            REQUIRE(values.size() == consecutiveRecalls);
            INFO(term);
            CHECK(std::ranges::min(values) > 0.9);
        }
        CHECK(solution.reachedTarget(0.9));
    }

    SECTION("a design that recalls in turn after one order of the waits never does")
    {
        auto solution = loadSequenceTimingFixture<ConsecutiveOrderRecall>(
            "incremental-recall-consecutive-order-one-wait-order.dnf");
        for (int evaluation = 0; evaluation < 3; ++evaluation)
        {
            solution.evaluate();
            const auto recalled = perRecallTerm(solution, "recalled");
            const auto oneAtATime = perRecallTerm(solution, "oneAtATime");
            REQUIRE(recalled.size() == consecutiveRecalls);
            INFO("evaluation " << evaluation);
            CHECK(std::min(std::ranges::min(recalled), std::ranges::min(oneAtATime)) <= 0.9);
            CHECK_FALSE(solution.reachedTarget(0.9));
        }
    }
}

// R01 of stage-5-batch-4 recalls out of order when the first trigger comes 460 or less after the demonstration, and
// scored 0.999 on the evaluations that drew a longer first wait. 300 comes twice in every evaluation, first in one of the
// two sequences, so it fails every time.
TEST_CASE("ConsecutiveOrderRecall fails a design that recalls out of order after the shortest wait on every evaluation", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ consecutiveOrderRecall };
    auto solution = loadSequenceTimingFixture<ConsecutiveOrderRecall>(
        "incremental-recall-consecutive-order-short-first-wait.dnf");
    for (size_t confirmation = 0; confirmation < 4; ++confirmation)
    {
        solution.prepareTargetConfirmation(confirmation);
        solution.evaluate();
        const auto recalls = recallTrials(solution);
        REQUIRE(recalls.size() == consecutiveRecalls);
        INFO("confirmation " << confirmation);
        if (confirmation % 2 == 0)
        {
            // The first sequence starts with the shortest wait, so its first recall comes 300 after the demonstration.
            CHECK(recalls.front().at("wait").get<double>() == Catch::Approx(300.0));
            CHECK(recalls.front().at("terms").at("recalled").get<double>() < 0.9);
        }
        CHECK(solution.getFitness() < 0.95);
        CHECK_FALSE(solution.reachedTarget(0.9));
    }
}

// R01 of stage-5-batch-5 logged 0.97, recalls in order twice, then out of order or with stray peaks at a third trigger
// (the user's replay, 5 November 2026: "at the third one the behaviour was overt").
TEST_CASE("ConsecutiveOrderRecall fails a design whose third recall breaks down", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ consecutiveOrderRecall };
    auto solution = loadSequenceTimingFixture<ConsecutiveOrderRecall>(
        "incremental-recall-consecutive-order-fails-third-recall.dnf");
    for (size_t confirmation = 0; confirmation < 2; ++confirmation)
    {
        solution.prepareTargetConfirmation(confirmation);
        solution.evaluate();
        INFO("confirmation " << confirmation);
        CHECK_FALSE(solution.reachedTarget(0.9));
    }
}

// To pass, the fitness itself must exceed the target as well as each term (the user, 5 November 2026). With weights
// that add up to 1 the terms already imply it; the rule holds whatever they add up to.
TEST_CASE("ConsecutiveOrderRecall reaches its target only with a fitness above it", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ consecutiveOrderRecall };
    auto solution = loadSequenceTimingFixture<ConsecutiveOrderRecall>(
        "incremental-recall-consecutive-order-reference.dnf");
    solution.evaluate();
    REQUIRE(solution.reachedTarget(0.9));
    CHECK(solution.getFitness() > 0.9);
    CHECK_FALSE(solution.reachedTarget(solution.getFitness()));
}

// S01 of stage-5-batch-3 lets E3 or E4 go, then peaks there again after the sequence, in most recalls. Below threshold
// for most of the samples after its release, such an event kept about 97% of its release credit. With the shortest
// and longest waits in every evaluation S01 no longer peaks again; R03 of the same batch does, in most recalls. From stage 5 each
// sample an event peaks again after its release, until the next trigger, costs a third of the recall's release credit.
TEST_CASE("ConsecutiveOrderRecall's release loses a third of a recall's credit for each sample an event peaks again", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ consecutiveOrderRecall };

    SECTION("the stage-5 reference's events never peak again")
    {
        auto solution = loadSequenceTimingFixture<ConsecutiveOrderRecall>(
            "incremental-recall-consecutive-order-reference.dnf");
        for (int evaluation = 0; evaluation < 2; ++evaluation)
        {
            solution.evaluate();
            const auto refired = perRecall(solution, "refiredSamples");
            REQUIRE(refired.size() == consecutiveRecalls);
            INFO("evaluation " << evaluation);
            CHECK(std::ranges::all_of(refired, [](const double samples) { return samples == 0.0; }));
            CHECK(solution.reachedTarget(0.9));
        }
    }

    SECTION("an event that peaks again costs its recall's release")
    {
        auto solution = loadSequenceTimingFixture<ConsecutiveOrderRecall>("incremental-recall-consecutive-order-refires.dnf");
        double mostRefired = 0.0;
        for (int evaluation = 0; evaluation < 2; ++evaluation)
        {
            solution.evaluate();
            const auto refired = perRecall(solution, "refiredSamples");
            REQUIRE(refired.size() == consecutiveRecalls);
            mostRefired = std::max(mostRefired, std::ranges::max(refired));
            for (const std::string term : { "released", "releasedWithoutGuide" })
            {
                const auto released = perRecallTerm(solution, term);
                REQUIRE(released.size() == consecutiveRecalls);
                for (size_t recall = 0; recall < refired.size(); ++recall)
                {
                    INFO("evaluation " << evaluation << ", " << term << ", recall " << recall << ", refired " << refired[recall]);
                    CHECK(released[recall] <= std::max(0.0, 1.0 - refired[recall] / 3.0) + 0.005);
                }
            }
            CHECK_FALSE(solution.reachedTarget(0.9));
        }
        CHECK(mostRefired >= 3.0);
    }
}

// A saved design replays the first sequence of six triggers, each middle wait at the middle of its range: waits of 300,
// 800, 1500, 300, 1500 and 1000, each from the end of the recall window before it.
TEST_CASE("ConsecutiveOrderRecall saves six triggers in turn, so a saved design replays consecutive recall", "[Solutions][IncrementalRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ consecutiveOrderRecall };
    const auto fixture = loadFixtureSimulation("incremental-recall-consecutive-order-reference.dnf");
    ConsecutiveOrderRecall solution(sequenceTimingTopology(), *fixture);
    const auto saveDirectory = std::filesystem::temp_directory_path() / "neat-dnfs-incremental-recall-5-replay";

    Solution& saved = solution;
    saved.buildPhenotype();
    saved.createPhenotypeEnvironment();
    std::filesystem::path savedFile;
    {
        const auto simulation = std::make_shared<dnf_composer::Simulation>(saved.getPhenotype());
        dnf_composer::SimulationFileManager(simulation, saveDirectory.generic_string()).saveElementsToJson();
        const std::string identifier = simulation->getUniqueIdentifier();
        savedFile = saveDirectory / identifier / (identifier + ".dnf");
    }
    saved.clearPhenotype();
    REQUIRE(std::filesystem::exists(savedFile));

    const auto replay = std::make_shared<dnf_composer::Simulation>();
    dnf_composer::SimulationFileManager(replay, savedFile.generic_string()).loadElementsFromJson();
    replay->init();
    const auto output = std::dynamic_pointer_cast<dnf_composer::element::NeuralField>(replay->getElement("nf 2"));
    REQUIRE(output != nullptr);
    // The triggers come at 1550, 4050, 7250, 9250, 12450 and 15150. The reference's first recall peaks 130, 270, 610 and
    // 1210 after its trigger; it winds back a little early, so the later ones peak 90, 180 and 520 after theirs, and E4
    // about 10 earlier with each recall, from 1100 to 1060 after its trigger.
    const std::array<double, 6> cues{ 1550.0, 4050.0, 7250.0, 9250.0, 12450.0, 15150.0 };
    const std::array<std::array<double, 4>, 6> upAfterCue{ { { 130.0, 270.0, 610.0, 1210.0 },
        { 90.0, 180.0, 520.0, 1100.0 }, { 90.0, 180.0, 520.0, 1090.0 }, { 90.0, 180.0, 520.0, 1080.0 },
        { 90.0, 180.0, 520.0, 1070.0 }, { 90.0, 180.0, 520.0, 1060.0 } } };
    for (size_t recall = 0; recall < cues.size(); ++recall)
    {
        for (size_t event = 0; event < incrementalRecallEvents.size(); ++event)
        {
            const double time = cues[recall] + upAfterCue[recall][event];
            while (replay->getT() < time - 1e-9)
            {
                replay->step();
            }
            const auto bumps = output->getBumps();
            INFO("recall " << recall << ", t " << time << ", " << incrementalRecallEvents[event].label);
            REQUIRE(bumps.size() == 1);
            CHECK(std::abs(bumps.front().centroid - incrementalRecallEvents[event].position) <= 1.0);
        }
    }
    std::filesystem::remove_all(saveDirectory);
}

TEST_CASE("ConsecutiveOrderRecall is registered with one input and one output, starting from the incremental-recall template", "[SolutionRegistry]")
{
    const auto* task = findTask(consecutiveOrderRecall);
    REQUIRE(task != nullptr);
    CHECK(task->templateFile == "incremental-recall.dnf");
    CHECK(task->inputs == 1);
    CHECK(task->outputs == 1);
}

TEST_CASE("Genome lists the connections evolution cannot make, honouring allowOutputAsConnectionSource", "[genome][legal_connections]")
{
    OutputSourceRuleGuard guard;
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order-behaviour" };
    GenomeMutationConstants::allowOutputAsConnectionSource = false;

    // Built from the earlier serial-order design, whose output fed its done field (gk cg 2-7).
    const auto outputFedDone = loadSequenceTimingFixture<SerialOrderBehaviour>("serial-order-overlapping-peaks.dnf");
    const auto handBuilt = loadSequenceTimingFixture<SerialOrderBehaviour>("serial-order-handbuilt.dnf");

    CHECK(outputFedDone.getGenome().connectionTuplesEvolutionCannotMake() == std::vector{ ConnectionTuple{ 2, 7 } });
    CHECK(handBuilt.getGenome().connectionTuplesEvolutionCannotMake().empty());

    GenomeMutationConstants::allowOutputAsConnectionSource = true;
    CHECK(outputFedDone.getGenome().connectionTuplesEvolutionCannotMake().empty());
}

// Everything written to std::cout while @p action runs, which is where the logger prints.
static std::string consoleOutputOf(const std::function<void()>& action)
{
    std::ostringstream captured;
    std::streambuf* const originalCoutBuffer = std::cout.rdbuf(captured.rdbuf());
    try
    {
        action();
    }
    catch (...)
    {
        std::cout.rdbuf(originalCoutBuffer);
        throw;
    }
    std::cout.rdbuf(originalCoutBuffer);
    return captured.str();
}

// A template is where a structure evolution could not have built enters a run: its offspring keep it.
TEST_CASE("Loading a template warns about each connection evolution cannot make", "[SolutionRegistry]")
{
    OutputSourceRuleGuard guard;
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order-behaviour" };
    GenomeMutationConstants::allowOutputAsConnectionSource = false;
    const auto* task = findTask("serial-order-behaviour");
    REQUIRE(task != nullptr);
    const auto loadTemplate = [task](const std::string& fixture) {
        const auto simulation = loadFixtureSimulation(fixture);
        return consoleOutputOf([task, &simulation] {
            const auto solution = task->makeFromTemplate(defaultTopologyFor(*task), *simulation);
        });
    };

    const auto outputFedDone = loadTemplate("serial-order-overlapping-peaks.dnf");
    CHECK(outputFedDone.find("connection 2-7") != std::string::npos);
    CHECK(outputFedDone.find("evolution cannot make") != std::string::npos);
    CHECK(loadTemplate("serial-order-handbuilt.dnf").find("evolution cannot make") == std::string::npos);

    GenomeMutationConstants::allowOutputAsConnectionSource = true;
    CHECK(loadTemplate("serial-order-overlapping-peaks.dnf").find("evolution cannot make") == std::string::npos);
}

TEST_CASE("Every serial-order task is registered with one input and one output, starting from serial order's template", "[SolutionRegistry]")
{
    for (const std::string slug : serialOrderTasks)
    {
        INFO("task " << slug);
        const auto* task = findTask(slug);
        REQUIRE(task != nullptr);
        REQUIRE(task->templateFile == "serial-order.dnf");
        REQUIRE(task->inputs == 1);
        REQUIRE(task->outputs == 1);
    }
}

// ConfigLoader::loadConfig() sets every process-wide constant a task's config
// names, not just the field size. Leaving any of them behind would run every
// later test in the same process under that task's settings.
TEST_CASE("ScopedTaskConfig restores every constant the task's config overrode", "[SolutionRegistry]")
{
    const double deltaTBefore = SimulationConstants::deltaT;
    const int xSizeBefore = DimensionConstants::xSize;
    {
        const ScopedTaskConfig taskConfig{ "hri-packaging" };
        REQUIRE(SimulationConstants::deltaT != deltaTBefore);
        REQUIRE(DimensionConstants::xSize != xSizeBefore);
    }
    REQUIRE(SimulationConstants::deltaT == deltaTBefore);
    REQUIRE(DimensionConstants::xSize == xSizeBefore);
}

// Population::endConditionMet() stops a run once every *recorded* partial
// fitness passes the target, while the fitness it reports is the weighted sum
// testPhenotype() computed. If a task records a different term than it weighs,
// a run can stop early on a solution whose fitness never reached the target.
// A task may instead pay the larger of that sum and a second score it records in
// its trace (sequence-timing-behaviour-guided: the intended behaviour's); its
// fitness is then never below the sum, so the stop rule is just as safe, also
// where reachedTarget() checks the terms of the larger score instead.
TEST_CASE("Every registered task's fitness is the weighted sum of the partial fitness it records", "[Solutions][SolutionRegistry]")
{
    for (const auto& task : taskEntries())
    {
        DYNAMIC_SECTION("task: " << task.slug)
        {
            resetGlobalState();
            const ScopedTaskConfig taskConfig{ std::string(task.slug) };

            const auto solution = task.makeFromTopology(defaultTopologyFor(task));
            solution->initialize();
            solution->evaluate();

            const auto& recorded = solution->getParameters().partialFitness;
            const auto weights = ConfigLoader::loadFitnessWeights(std::string(task.slug), recorded.size());
            const double weightedSum = std::inner_product(recorded.begin(), recorded.end(), weights.begin(), 0.0);

            const auto& traceText = solution->getParameters().evaluationTrace;
            const auto trace = traceText.empty() ? nlohmann::json::object() : nlohmann::json::parse(traceText);
            if (trace.contains("fitness"))
            {
                const auto& scores = trace.at("fitness");
                REQUIRE(scores.at("guided").get<double>() == Catch::Approx(weightedSum).margin(1e-9));
                const double larger = std::max(weightedSum, scores.at("intended").get<double>());
                REQUIRE(solution->getFitness() == Catch::Approx(larger).margin(1e-9));
            }
            else
            {
                REQUIRE(solution->getFitness() == Catch::Approx(weightedSum).margin(1e-9));
            }
        }
    }
}

TEST_CASE("Every registered task's template file exists under templates/", "[SolutionRegistry]")
{
    // The registry names each task's starting template by filename, but nothing
    // resolved it until neat-dnfs-sol-eval / neat-dnfs-inc-evol ran. Renaming or
    // replacing a template therefore broke both binaries at runtime with the
    // suite still green.
    for (const auto& task : taskEntries())
    {
        const auto templatePath = paths::resourceRoot() / "templates" / std::string(task.templateFile);
        INFO("task '" << task.slug << "' references missing template " << task.templateFile);
        REQUIRE(std::filesystem::exists(templatePath));
    }
}


#include <sstream>
// TEMPORARY DIAGNOSTIC - not for commit. Steps a serial-order replay and summarises the output field (or REPLAY_FIELD) per phase:
// for each event, when its peak first and last stood (after the phase start) and its highest activation; the most
// event peaks up at once; activity away from the events; and when the output last held any activity.
TEST_CASE("DIAG replay phase summary", "[.diag]")
{
    const char* path = std::getenv("REPLAY_DNF");
    REQUIRE(path != nullptr);
    std::vector<double> phaseStarts;
    { std::stringstream list(std::getenv("REPLAY_PHASES")); std::string item; while (std::getline(list, item, ',')) phaseStarts.push_back(std::stod(item)); }
    const std::array<std::pair<const char*, double>, 4> events{ { { "D", 180.0 }, { "C", 120.0 }, { "A", 20.0 }, { "B", 80.0 } } };
    const auto simulation = std::make_shared<dnf_composer::Simulation>("diag", 2.0, 0.0, 0.0);
    const dnf_composer::SimulationFileManager sfm(simulation, path);
    sfm.loadElementsFromJson();
    simulation->setDeltaT(2.0);
    simulation->init();
    const auto output = std::dynamic_pointer_cast<dnf_composer::element::NeuralField>(simulation->getElement(std::getenv("REPLAY_FIELD") ? std::getenv("REPLAY_FIELD") : "nf 2"));
    for (size_t phase = 0; phase + 1 < phaseStarts.size(); ++phase)
    {
        std::array<double, 4> first{ -1, -1, -1, -1 }, last{ -1, -1, -1, -1 }, peak{ -99, -99, -99, -99 };
        int mostAtOnce = 0; double offMax = -99; int offSteps = 0, steps = 0; double lastActive = -1;
        while (simulation->t < phaseStarts[phase + 1])
        {
            simulation->step();
            const double since = simulation->t - phaseStarts[phase];
            const auto& u = output->getComponents()->at("activation");
            int upNow = 0;
            for (size_t e = 0; e < events.size(); ++e)
            {
                double m = -99;
                for (int x = int(events[e].second) - 10; x <= int(events[e].second) + 10; ++x) m = std::max(m, u[size_t(x)]);
                peak[e] = std::max(peak[e], m);
                if (m > 0.0) { if (first[e] < 0) first[e] = since; last[e] = since; ++upNow; }
            }
            mostAtOnce = std::max(mostAtOnce, upNow);
            bool offActive = false; bool anyActive = false;
            for (size_t x = 0; x < u.size(); ++x)
            {
                const bool atEvent = std::ranges::any_of(events, [x](const auto& ev) { return std::abs(double(x) - ev.second) <= 10.0; });
                if (u[x] > 0.0) anyActive = true;
                if (!atEvent) { offMax = std::max(offMax, u[x]); offActive |= u[x] > 0.0; }
            }
            offSteps += offActive; ++steps;
            if (anyActive) lastActive = since;
        }
        std::string line = std::format("phase {} ({:.0f}-{:.0f}):", phase, phaseStarts[phase], phaseStarts[phase + 1]);
        for (size_t e = 0; e < events.size(); ++e)
            line += first[e] < 0 ? std::format(" {} never(max {:.1f})", events[e].first, peak[e])
                                 : std::format(" {} {:.0f}-{:.0f}(max {:.1f})", events[e].first, first[e], last[e], peak[e]);
        line += std::format(" | most at once {} | off-event active {:.0f}% of steps, max {:.1f} | last active {:.0f}",
            mostAtOnce, 100.0 * offSteps / steps, offMax, lastActive);
        std::cout << line << "\n";
    }
}
