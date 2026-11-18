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

// The final task of the serial-order and timing work: from 10 November 2026, six joint goals, from 11 November due at
// the first six times of the ten-event task, and from 12 November on fields of 200.
static const std::string sequenceAndTimingRecall{ "sequence-and-timing-recall" };

// The six events in their order, each with when it is due after the memory recall trigger.
static const std::array<std::pair<std::string, double>, 6> recallEvents{ { { "BSA", 50.0 }, { "BLA", 150.0 },
    { "BSB", 350.0 }, { "BLB", 450.0 }, { "BS", 650.0 }, { "BP", 750.0 } } };

// Where each event lies on the input and output fields, in the same order.
static const std::array<double, 6> recallEventPositions{ 20.0, 100.0, 60.0, 120.0, 160.0, 180.0 };

static SequenceAndTimingRecall loadRecallFixture(const std::string& name)
{
    return loadSequenceTimingFixture<SequenceAndTimingRecall>("sequence-and-timing-recall-" + name + ".dnf");
}

// Every recall of an evaluation, in trace order: each consecutive trial's three.
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

// The hand-built reference (11 November 2026: the ten-event design of 29 October on the first six of its times, with
// thinner peaks and shallower troughs; on fields of 200 from 12 November) recalls all six in order and on time, and lets each go. Its first two events
// come early in their windows: the trigger's lift of the stamps brings BSA's first peak to 60, the edge of its window,
// and BLA's to 120-130, 20-30 early, as it did the first two of the ten events. Its wind-back starts once
// BP is released, and clears the troughs of BS and BP before the recall window ends.
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
        for (const size_t term : std::initializer_list<size_t>{ 0, 1, 3 })
        {
            INFO("evaluation " << evaluation << ", f" << term + 1);
            CHECK(partials[term] > 0.9);
        }
        CHECK(partials[4] > 0.85); // timing: BSA and BLA come early in their windows
        CHECK(partials[6] > 0.85); // released, half for a trough
        CHECK(partials[7] > 0.8); // stopped, half for a trough at every event
        CHECK(partials[2] > 0.2); // memory: the primacy gradient
        CHECK(partials[5] > 0.5); // the events to come
        const auto scores = nlohmann::json::parse(solution.getParameters().evaluationTrace).at("fitness");
        CHECK(scores.at("intended").get<double>() >= 0.9);
        CHECK(solution.getFitness() >= 0.9);
    }
}

// Below threshold the reference holds a primacy gradient: each stored event pre-shaped above the one stored after it.
// The guides score it by its steps, so the gradient earns memory and to-come though its later events sit near the
// output's level: the earliest of them at full height above it would not make a steeper staircase.
TEST_CASE("SequenceAndTimingRecall pays the reference's primacy gradient by its steps", "[Solutions][SequenceAndTimingRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ sequenceAndTimingRecall };
    auto solution = loadRecallFixture("reference");
    solution.evaluate();

    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 8);
    CHECK(partials[2] > 0.6); // memory
    CHECK(partials[5] > 0.85); // the events to come
}

// With no connection into the output, nothing drives a peak there: the output earns the input and quiet terms alone,
// which are an eighth of the intended behaviour.
TEST_CASE("SequenceAndTimingRecall gives an output that never responds the input and quiet terms alone", "[Solutions][SequenceAndTimingRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ sequenceAndTimingRecall };
    SequenceAndTimingRecall solution(sequenceTimingTopology());
    solution.addFieldGene(makeFixedFieldGene(FieldGeneType::INPUT, 1));
    solution.addFieldGene(makeFixedFieldGene(FieldGeneType::OUTPUT, 2));
    solution.evaluate();

    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 8);
    CHECK(partials[1] == Catch::Approx(1.0));
    CHECK(partials[2] < 0.05);
    CHECK(partials[3] == 0.0);
    CHECK(partials[4] == 0.0);
    CHECK(partials[5] < 0.05);
    CHECK(partials[6] == 0.0);
    CHECK(partials[7] == 0.0);
    CHECK(solution.getFitness() == Catch::Approx(0.125).margin(0.02));
}

// The reference with 4-5 at 0: no time stamps are recorded, so all six fire together about 50 after the trigger, as
// one bump across the whole field. None is on time, and a bump that wide earns no peak score.
TEST_CASE("SequenceAndTimingRecall gives a burst no timing credit, and order credit only to the events that fire after the others", "[Solutions][SequenceAndTimingRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ sequenceAndTimingRecall };
    auto solution = loadRecallFixture("burst");
    solution.evaluate();

    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 8);
    CHECK(partials[3] < 0.6);
    CHECK(partials[4] < 0.1);
    CHECK(solution.getFitness() < 0.5);
}

// The reference with 8-5 and 8-10 at 800: the stamps climb four times as fast, and after BSA, which the trigger's lift
// brings on time, it recalls the other five in order 20-50 apart, by about +210. Each goal comes well short of half its
// demonstrated gap after the goals due before it, the spacing that earns full order credit.
TEST_CASE("SequenceAndTimingRecall pays a fast cascade part of the order credit, not the timing", "[Solutions][SequenceAndTimingRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ sequenceAndTimingRecall };
    auto solution = loadRecallFixture("fast-cascade");
    solution.evaluate();

    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 8);
    CHECK(partials[3] > 0.4); // about 0.54: BLA comes soon after BSA, which the trigger's lift brings on time
    CHECK(partials[3] < 0.7);
    CHECK(partials[4] < 0.3); // BSA alone on time: a sixth of the credit
}

// The reference with 8-5 and 8-10 at 1600: the stamps climb eight times as fast, so after every trigger each goal first
// peaks in order 10-20 apart, BSA +40 to BP +120, as batch-8's R07 did (12 November 2026). Full order credit needs each goal at least half its demonstrated gap after the goals due before it, so goals
// crowded together earn little of it, and the release and stop terms, which count only as far as the recall was in
// order, little with it.
TEST_CASE("SequenceAndTimingRecall gives goals crowded together in time little order credit", "[Solutions][SequenceAndTimingRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ sequenceAndTimingRecall };
    auto solution = loadRecallFixture("crowded-cascade");
    solution.evaluate();

    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 8);
    CHECK(partials[3] < 0.5); // about 0.4: any lead counted as fully in order before the spacing rule
    CHECK(solution.getFitness() < 0.55); // about 0.5: its stored staircase still earns the memory and to-come guides
}

// The reference with 9-5 at -200: the wind-back presses the stamps down less, so after a recall in order and on time
// BSB climbs back and peaks again from about +1610, after the recall window, as batch-8's R04 peaked between the goals
// (12 November 2026). A silence probe follows the last recall window of each consecutive trial: its samples count as
// samples after the sequence, so peaks there cost the stop term, while the reference stays quiet through it.
TEST_CASE("SequenceAndTimingRecall probes for silence after the recall window", "[Solutions][SequenceAndTimingRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ sequenceAndTimingRecall };
    auto reference = loadRecallFixture("reference");
    reference.evaluate();
    auto solution = loadRecallFixture("peaks-after-window");
    solution.evaluate();

    for (const auto& recall : recallsOf(timingTrials(reference)))
    {
        CHECK(recall.at("recall").at("afterSequence").at("quiet").get<double>() >= 0.95);
    }
    const auto trials = timingTrials(solution);
    REQUIRE(trials.size() == 4);
    double quietAfter = 0.0;
    for (size_t trial = 1; trial < 4; ++trial)
    {
        const auto& lastRecall = trials[trial].at("recalls").back();
        quietAfter += lastRecall.at("recall").at("afterSequence").at("quiet").get<double>() / 3.0;
    }
    CHECK(quietAfter < 0.9);
}

// The reference with nf 6 at tau 70: each event is released later, so the peaks of events 100 apart are up together
// for longer, each where and when it is due. Overlap is allowed: no event's credit is shared with the peaks up with it.
// Its timing is the reference's: every event within its window.
TEST_CASE("SequenceAndTimingRecall gives ordered peaks that overlap in time full order and timing credit", "[Solutions][SequenceAndTimingRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ sequenceAndTimingRecall };
    auto solution = loadRecallFixture("overlapping-peaks");
    solution.evaluate();

    const auto trials = timingTrials(solution);
    REQUIRE(trials.size() == 4);
    for (size_t trial = 1; trial < 4; ++trial)
    {
        const auto& events = trials[trial].at("recalls").front().at("recall").at("events");
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
        INFO("trial " << trial);
        CHECK(mostUpTogether >= 2);
    }
    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 8);
    CHECK(partials[3] > 0.9);
    CHECK(partials[4] > 0.85);
}

// The reference with 7-8's global part at 0.4: six pending events open the window by themselves, so after the
// demonstration it starts recalling without any trigger.
TEST_CASE("SequenceAndTimingRecall gives an output that fires without the trigger almost no recall credit", "[Solutions][SequenceAndTimingRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ sequenceAndTimingRecall };
    auto solution = loadRecallFixture("fires-without-trigger");
    solution.evaluate();

    const auto trials = timingTrials(solution);
    REQUIRE(trials.front().at("kind") == "probe");
    const double probeQuiet = trials.front().at("quietWithoutCue").get<double>()
        * trials.front().at("eventsQuietWithoutCue").get<double>();
    CHECK(probeQuiet < 0.5);
    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 8);
    for (size_t term = 3; term < partials.size(); ++term)
    {
        INFO("f" << term + 1);
        CHECK(partials[term] < 0.1);
    }
    CHECK(solution.getFitness() < 0.2);
}

// The latest time any event first went above threshold after the trigger, less the earliest, over the events that did.
static double firstPeakSpread(const nlohmann::json& recall)
{
    std::vector<double> afterCue;
    for (const auto& [label, due] : recallEvents)
    {
        const auto& event = recall.at("recall").at("events").at(label);
        if (!event.at("afterCue").is_null())
        {
            afterCue.push_back(event.at("afterCue").get<double>());
        }
    }
    REQUIRE_FALSE(afterCue.empty());
    const auto [earliest, latest] = std::ranges::minmax(afterCue);
    return latest - earliest;
}

// Batch ladder-L5's best of chain A (L5A04, 18 November 2026): by +10 after every trigger the output holds bumps 24-47
// wide over all six places, and only later splits them into narrow peaks, at BLA and BLB about +260 and at BS and BP
// about +590. Counted from when a bump centred on each place, that split read as BLA, BLB, BS and BP recalled later,
// in order (fitness 0.627). An event fires when its place first goes above threshold, whatever bump takes it there.
TEST_CASE("SequenceAndTimingRecall fires an event when its place first goes above threshold, so a burst over every place earns no order credit when it splits", "[Solutions][SequenceAndTimingRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ sequenceAndTimingRecall };
    auto solution = loadRecallFixture("merged-burst");
    solution.evaluate();

    for (const auto& recall : recallsOf(timingTrials(solution)))
    {
        CHECK(firstPeakSpread(recall) <= 20.0);
    }
    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 8);
    CHECK(partials[3] < 0.25); // only BSA in place, and only as far as nothing fired before it
    CHECK(partials[4] < 0.05);
    CHECK(solution.getFitness() < 0.3);
}

// Batch ladder-L6's best of chain D (L6D04, 18 November 2026): narrow bumps jump around the output, with field-wide
// flashes. A bump that jumps more than 10 counts as a new one, still in place, so wherever one landed on an event's
// place that event peaked (fitness 0.596). Its first flash, about +30, puts five of the six places above threshold.
TEST_CASE("SequenceAndTimingRecall gives bumps jumping across the output almost no recall credit", "[Solutions][SequenceAndTimingRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ sequenceAndTimingRecall };
    auto solution = loadRecallFixture("jumping-bumps");
    solution.evaluate();

    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 8);
    CHECK(partials[3] < 0.15);
    CHECK(partials[4] < 0.05);
    CHECK(solution.getFitness() < 0.3);
}

// Bursts and jumping bumps rank below a design that recalls part of the sequence in order with separate narrow peaks
// (batch ladder-L6's best of chain B, L6B04: single order recall, timing drifting), which ranks below one that
// recalls in order and on time after every trigger (batch ladder-L3's best of chain D, L3D03, 18 November 2026).
TEST_CASE("SequenceAndTimingRecall ranks bursts and jumping bumps below a partial recall in order, and that below an evolved recall in order and on time", "[Solutions][SequenceAndTimingRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ sequenceAndTimingRecall };
    const auto fitnessOf = [](const std::string& fixture) {
        auto solution = loadRecallFixture(fixture);
        solution.evaluate();
        return solution.getFitness();
    };
    const double mergedBurst = fitnessOf("merged-burst");
    const double jumpingBumps = fitnessOf("jumping-bumps");
    const double partialOrder = fitnessOf("partial-order");
    const double evolvedRecall = fitnessOf("evolved-recall");

    CHECK(mergedBurst < partialOrder - 0.1);
    CHECK(jumpingBumps < partialOrder - 0.1);
    CHECK(partialOrder < evolvedRecall - 0.1);
    CHECK(evolvedRecall >= 0.9);
}

TEST_CASE("SequenceAndTimingRecall runs the probe, then three consecutive trials whose first waits come one from each third of 300-1500", "[Solutions][SequenceAndTimingRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ sequenceAndTimingRecall };
    SequenceAndTimingRecall solution(sequenceTimingTopology());
    solution.addFieldGene(makeFixedFieldGene(FieldGeneType::INPUT, 1));
    solution.addFieldGene(makeFixedFieldGene(FieldGeneType::OUTPUT, 2));
    const auto firstWaits = [&solution]() {
        solution.evaluate();
        const auto trials = timingTrials(solution);
        REQUIRE(trials.size() == 4);
        CHECK(trials.front().at("kind") == "probe");
        std::vector<double> waits;
        for (size_t trial = 1; trial < 4; ++trial)
        {
            CHECK(trials[trial].at("kind") == "consecutive");
            waits.push_back(trials[trial].at("recalls").front().at("wait").get<double>());
        }
        // The probe waits as long as the longest first wait.
        CHECK(trials.front().at("wait").get<double>() == waits.back());
        return waits;
    };

    const auto first = firstWaits();
    const auto second = firstWaits();
    for (size_t i = 0; i < first.size(); ++i)
    {
        CHECK(first[i] >= 300.0 + 400.0 * static_cast<double>(i));
        CHECK(first[i] <= 700.0 + 400.0 * static_cast<double>(i));
        CHECK(std::fmod(first[i], 10.0) == Catch::Approx(0.0).margin(1e-9));
    }
    CHECK(first != second);
}

// In a population every design of a generation must meet the same waits, or the best score of the generation is the
// luckiest draw: the population gives them one evaluation seed, and the waits follow from it.
TEST_CASE("SequenceAndTimingRecall draws the same waits from the same evaluation seed, and other waits from another", "[Solutions][SequenceAndTimingRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ sequenceAndTimingRecall };
    const auto waitsWithSeed = [](const std::uint64_t seed) {
        SequenceAndTimingRecall solution(sequenceTimingTopology());
        solution.addFieldGene(makeFixedFieldGene(FieldGeneType::INPUT, 1));
        solution.addFieldGene(makeFixedFieldGene(FieldGeneType::OUTPUT, 2));
        solution.setEvaluationSeed(seed);
        solution.evaluate();
        const auto trials = timingTrials(solution);
        REQUIRE(trials.size() == 4);
        std::vector<double> waits{ trials.front().at("wait").get<double>() };
        for (size_t trial = 1; trial < trials.size(); ++trial)
        {
            for (const auto& recall : trials[trial].at("recalls"))
            {
                waits.push_back(recall.at("wait").get<double>());
            }
        }
        return waits;
    };

    const auto waits = waitsWithSeed(7);
    REQUIRE(waits.size() == 10);
    CHECK(waitsWithSeed(7) == waits);
    CHECK(waitsWithSeed(8) != waits);
    // Seeded waits keep the unseeded ones' shape: each consecutive trial holds one wait from each third of 300-1500.
    for (size_t trial = 0; trial < 3; ++trial)
    {
        std::vector<double> trialWaits(waits.begin() + 1 + 3 * trial, waits.begin() + 4 + 3 * trial);
        std::ranges::sort(trialWaits);
        for (size_t i = 0; i < trialWaits.size(); ++i)
        {
            CHECK(trialWaits[i] >= 300.0 + 400.0 * static_cast<double>(i));
            CHECK(trialWaits[i] <= 700.0 + 400.0 * static_cast<double>(i));
        }
    }
}

TEST_CASE("SequenceAndTimingRecall gives each consecutive trial one demonstration and three triggers, one wait from each third of 300-1500, the later two in a shuffled order", "[Solutions][SequenceAndTimingRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ sequenceAndTimingRecall };
    SequenceAndTimingRecall solution(sequenceTimingTopology());
    solution.addFieldGene(makeFixedFieldGene(FieldGeneType::INPUT, 1));
    solution.addFieldGene(makeFixedFieldGene(FieldGeneType::OUTPUT, 2));

    constexpr double demonstrationEnd = 800.0;
    constexpr double recallWindow = 1250.0;
    bool laterEverAscending = false;
    bool laterEverDescending = false;
    for (int evaluation = 0; evaluation < 6; ++evaluation)
    {
        solution.evaluate();
        const auto trials = timingTrials(solution);
        REQUIRE(trials.size() == 4);
        for (size_t trial = 1; trial < 4; ++trial)
        {
            const auto& consecutive = trials[trial];
            REQUIRE(consecutive.at("kind") == "consecutive");
            const auto& recalls = consecutive.at("recalls");
            REQUIRE(recalls.size() == 3);

            std::vector<double> waits;
            // Each wait runs from the end of the demonstration, then from the end of the previous recall window.
            double windowEnd = demonstrationEnd;
            for (const auto& recall : recalls)
            {
                const double wait = recall.at("wait").get<double>();
                CHECK(recall.at("cue").get<double>() == Catch::Approx(windowEnd + wait));
                CHECK(recall.contains("quietBeforeCue"));
                CHECK(recall.at("recall").at("events").size() == 6);
                windowEnd = recall.at("cue").get<double>() + recallWindow;
                waits.push_back(wait);
            }
            (waits[1] < waits[2] ? laterEverAscending : laterEverDescending) = true;
            std::ranges::sort(waits);
            for (size_t i = 0; i < waits.size(); ++i)
            {
                CHECK(waits[i] >= 300.0 + 400.0 * static_cast<double>(i));
                CHECK(waits[i] <= 700.0 + 400.0 * static_cast<double>(i));
                CHECK(std::fmod(waits[i], 10.0) == Catch::Approx(0.0).margin(1e-9));
            }
        }
    }
    CHECK(laterEverAscending);
    CHECK(laterEverDescending);
}

// The reference without its wind-back recalls right after each demonstration and bursts at every later trigger. Its
// three first recalls earn full order credit and its six later ones a burst's, and every recall counts alike.
TEST_CASE("SequenceAndTimingRecall counts every recall alike, the first after a demonstration and the later ones", "[Solutions][SequenceAndTimingRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ sequenceAndTimingRecall };
    auto solution = loadRecallFixture("recalls-once");
    solution.evaluate();

    const auto recalls = recallsOf(timingTrials(solution));
    REQUIRE(recalls.size() == 9);
    double order = 0.0;
    for (const auto& recall : recalls)
    {
        const auto& events = recall.at("recall").at("events");
        for (const auto& [label, due] : recallEvents)
        {
            order += recall.at("quietBeforeCue").get<double>() * events.at(label).at("bestScore").get<double>()
                * events.at(label).at("inPlace").get<double>() / static_cast<double>(recalls.size() * recallEvents.size());
        }
    }
    // The trace rounds each value to two decimals.
    CHECK(solution.getParameters().partialFitness[3] == Catch::Approx(order).margin(0.02));
    CHECK(order < 0.6);
}

TEST_CASE("SequenceAndTimingRecall pays a design that recalls only after the first trigger less than the reference", "[Solutions][SequenceAndTimingRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ sequenceAndTimingRecall };
    auto reference = loadRecallFixture("reference");
    reference.evaluate();
    auto solution = loadRecallFixture("recalls-once");
    solution.evaluate();

    const auto trials = timingTrials(solution);
    REQUIRE(trials.back().at("kind") == "consecutive");
    const auto& recalls = trials.back().at("recalls");
    REQUIRE(recalls.size() == 3);
    REQUIRE(recallsOf(trials).size() == 9);
    const auto eventsOnTime = [](const nlohmann::json& recall) {
        const auto& events = recall.at("recall").at("events");
        return std::ranges::count_if(recallEvents,
            [&events](const auto& event) { return events.at(event.first).at("onTime").get<double>() > 0.5; });
    };
    CHECK(eventsOnTime(recalls[0]) == 6);
    CHECK(eventsOnTime(recalls[1]) <= 2);
    CHECK(eventsOnTime(recalls[2]) <= 2);

    // Six of its nine recalls come after an earlier one, so it loses most of the reference's recall credit.
    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 8);
    CHECK(partials[4] < reference.getParameters().partialFitness[4] - 0.4);
    CHECK(solution.getFitness() < reference.getFitness() - 0.3);
}

// Each recall trial's trace says when each event first peaked after the trigger and how close that came to its time.
TEST_CASE("SequenceAndTimingRecall traces when each event first peaked after the trigger, and its timing credit", "[Solutions][SequenceAndTimingRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ sequenceAndTimingRecall };
    auto solution = loadRecallFixture("reference");
    solution.evaluate();

    const auto trials = timingTrials(solution);
    REQUIRE(trials.size() == 4);
    for (size_t trial = 1; trial < 4; ++trial)
    {
        CHECK(trials[trial].at("snapshots").size() > 2 * recallEvents.size());
    }
    const auto recalls = recallsOf(trials);
    REQUIRE(recalls.size() == 9);
    for (size_t recall = 0; recall < recalls.size(); ++recall)
    {
        const auto& events = recalls[recall].at("recall").at("events");
        for (const auto& [label, due] : recallEvents)
        {
            INFO("recall " << recall << ", " << label);
            const double afterCue = events.at(label).at("afterCue").get<double>();
            CHECK(std::abs(afterCue - due) <= 0.2 * due);
            CHECK(events.at(label).at("onTime").get<double>() > 0.5);
        }
        CHECK(recalls[recall].at("quietBeforeCue").get<double>() == Catch::Approx(1.0));
    }
}

// An event first peaking anywhere within 20% of its demonstrated time counts as on time, so it earns the full timing
// credit: centring an event already in its window earns nothing more. The reference has every event in its window.
TEST_CASE("SequenceAndTimingRecall gives every event within 20% of its time full timing credit", "[Solutions][SequenceAndTimingRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ sequenceAndTimingRecall };
    auto solution = loadRecallFixture("reference");
    solution.evaluate();

    for (const auto& recall : recallsOf(timingTrials(solution)))
    {
        for (const auto& [label, due] : recallEvents)
        {
            const auto& event = recall.at("recall").at("events").at(label);
            const double afterCue = event.at("afterCue").get<double>();
            INFO(label << " at " << afterCue);
            if (std::abs(afterCue - due) <= 0.2 * due)
            {
                CHECK(event.at("onTime").get<double>() == Catch::Approx(1.0));
            }
        }
    }
    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 8);
    CHECK(partials[4] > 0.95);
}

// Past the window the credit falls off steeply from 1, but never to 0 for an event that peaked: a quarter of it is the
// delay over the window's edge (or the edge over the delay), which keeps rising all the way in. The fast cascade
// recalls every event after BSA far early.
TEST_CASE("SequenceAndTimingRecall grades timing credit beyond the window by how far past its edge an event comes", "[Solutions][SequenceAndTimingRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ sequenceAndTimingRecall };
    auto solution = loadRecallFixture("fast-cascade");
    solution.evaluate();

    size_t outside = 0;
    for (const auto& recall : recallsOf(timingTrials(solution)))
    {
        for (const auto& [label, due] : recallEvents)
        {
            const auto& event = recall.at("recall").at("events").at(label);
            if (event.at("afterCue").is_null())
            {
                continue;
            }
            const double afterCue = event.at("afterCue").get<double>();
            const double edge = afterCue < due ? 0.8 * due : 1.2 * due;
            if (std::abs(afterCue - due) <= 0.2 * due || afterCue <= 0.0)
            {
                continue;
            }
            INFO(label << " at " << afterCue);
            const double edgeRatio = std::min(afterCue, edge) / std::max(afterCue, edge);
            CHECK(event.at("onTime").get<double>() == Catch::Approx(0.25 * edgeRatio).margin(0.01));
            ++outside;
        }
    }
    CHECK(outside > 0);
}

// One late event costs its whole recall: a recall's timing credit is its worst event's credit in place times its
// timing, not their mean. The fast cascade, with only BSA on time, keeps far less than the sixth BSA would earn it.
TEST_CASE("SequenceAndTimingRecall scores each recall's timing by its worst event", "[Solutions][SequenceAndTimingRecall]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ sequenceAndTimingRecall };
    for (const std::string fixture : { "reference", "fast-cascade" })
    {
        INFO(fixture);
        auto solution = loadRecallFixture(fixture);
        solution.evaluate();

        const auto trials = timingTrials(solution);
        REQUIRE(trials.front().at("kind") == "probe");
        const double probeQuiet = trials.front().at("quietWithoutCue").get<double>()
            * trials.front().at("eventsQuietWithoutCue").get<double>();
        double worst = 0.0;
        double average = 0.0;
        const auto recalls = recallsOf(trials);
        for (const auto& recall : recalls)
        {
            const auto& events = recall.at("recall").at("events");
            double lowest = 1.0;
            double sum = 0.0;
            for (const auto& [label, due] : recallEvents)
            {
                const auto& event = events.at(label);
                const double credit = event.at("bestScore").get<double>() * event.at("inPlace").get<double>()
                    * event.at("onTime").get<double>();
                lowest = std::min(lowest, credit);
                sum += credit;
            }
            const double quiet = recall.at("quietBeforeCue").get<double>();
            worst += quiet * lowest / static_cast<double>(recalls.size());
            average += quiet * sum / static_cast<double>(recallEvents.size() * recalls.size());
        }
        const auto& partials = solution.getParameters().partialFitness;
        REQUIRE(partials.size() == 8);
        CHECK(partials[4] == Catch::Approx(probeQuiet * worst).margin(0.01));
        if (fixture == "fast-cascade")
        {
            CHECK(partials[4] < probeQuiet * average - 0.1);
        }
    }
}

// A saved solution carries what createPhenotypeEnvironment() adds: a recall trial with a wait of 900, the trial of the
// reference's behaviour table, so a saved design opened in dnf-composer replays it.
TEST_CASE("SequenceAndTimingRecall saves a recall trial with a wait of 900, so a saved design replays it", "[Solutions][SequenceAndTimingRecall]")
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
    // The trigger comes again at 1700; 50 after each event is due, the reference's peak there is up (each first peaks
    // up to 10 after its time and stays up for 140 or more).
    for (size_t event = 0; event < recallEvents.size(); ++event)
    {
        const double time = 1700.0 + recallEvents[event].second + 50.0;
        while (replay->getT() < time - 1e-9)
        {
            replay->step();
        }
        const auto bumps = output->getBumps();
        INFO("t " << time << ", " << recallEvents[event].first);
        REQUIRE_FALSE(bumps.empty());
        const auto nearest = std::ranges::min(bumps | std::views::transform([&](const auto& bump) {
            return std::abs(bump.centroid - recallEventPositions[event]);
        }));
        CHECK(nearest <= 1.0);
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
    }
    for (const auto& gene : solution->getGenome().getConnectionGenes())
    {
        checkConnectionGeneEvolvable(gene);
    }
    // Reachable without output feedback, which the config turns on.
    GenomeMutationConstants::allowOutputAsConnectionSource = false;
    checkOnlyConnectionsEvolutionCanMake(*solution);
    // An added field's connection in starts at the default global and moves by whole steps from there, and every hidden
    // field keeps one such connection unless a rare toggle replaces it. The memory nf 3 has only one, 1-3, and recalls
    // nothing in order a step off its global, so that default must lie on the step.
    CHECK(onConnectionGlobalStep(GaussKernelConstants::amplitudeGlobal));
}

TEST_CASE("SequenceAndTimingRecall is registered with one input and one output, starting from its own template", "[SolutionRegistry]")
{
    const auto* task = findTask(sequenceAndTimingRecall);
    REQUIRE(task != nullptr);
    CHECK(task->templateFile == "sequence-and-timing-recall.dnf");
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
// fitness is then never below the sum, so the stop rule is just as safe.
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
