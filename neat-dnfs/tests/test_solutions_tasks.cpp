#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <filesystem>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>

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
#include "solutions/sequence_timing.h"
#include "solutions/serial_order.h"
#include "solutions/serial_order_fitness_test.h"
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

TEST_CASE("SequenceTiming contract", "[Solutions][SequenceTiming]")
{
    resetGlobalState();
    checkSolutionContract<SequenceTiming>(makeTopology(1, 1));
}

// Like dmts, SequenceTiming places stimuli beyond the global default xSize=100
// (events at 120 and 180), so the topology is built at the live xSize that
// config/solutions/sequence-timing.json sets rather than via makeTopology().
static SolutionTopology sequenceTimingTopology()
{
    using dnf_composer::element::ElementDimensions;
    const ElementDimensions dims{ DimensionConstants::xSize, DimensionConstants::dx };
    return SolutionTopology({
        { FieldGeneType::INPUT, dims },
        { FieldGeneType::OUTPUT, dims }
    });
}

TEST_CASE("SequenceTiming evaluate produces a bounded fitness", "[Solutions][SequenceTiming]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "sequence-timing" };

    SequenceTiming solution(sequenceTimingTopology());
    solution.initialize();

    REQUIRE_NOTHROW(solution.evaluate());
    REQUIRE(solution.getFitness() >= 0.0);
    REQUIRE(solution.getFitness() <= 1.0);
    REQUIRE(solution.getParameters().partialFitness.size() == 7);
}

// With no connection into the output field, nothing can ever drive a peak
// there: the output must earn full credit for staying silent while the
// sequence is demonstrated, and none of the three recall terms.
TEST_CASE("SequenceTiming gives no recall credit to an output field that never peaks", "[Solutions][SequenceTiming]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "sequence-timing" };

    // Fixed genes draw no random numbers, so the output field cannot ignite
    // on its own through a randomly drawn high resting level.
    SequenceTiming solution(sequenceTimingTopology());
    solution.addFieldGene(makeFixedFieldGene(FieldGeneType::INPUT, 1));
    solution.addFieldGene(makeFixedFieldGene(FieldGeneType::OUTPUT, 2));

    solution.evaluate();

    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 7);
    CHECK(partials[2] == Catch::Approx(1.0)); // output silent during demonstration
    CHECK(partials[3] == 0.0);                // recalled events present
    CHECK(partials[4] == 0.0);                // recalled intervals
    CHECK(partials[5] == 0.0);                // recalled peaks one at a time

    const auto trace = nlohmann::json::parse(solution.getParameters().evaluationTrace);
    REQUIRE(trace.at("trials").size() == 4); // three recall trials and the probe
    for (const auto& trial : trace.at("trials"))
    {
        if (trial.at("kind") == "recall")
        {
            CHECK(trial.at("recall").at("risingPeakCount") == 0);
            CHECK(trial.at("recall").at("onsets").at("D").is_null());
        }
    }
}

// Reads, for every trial of an evaluation, when each demonstrated pulse started after the trial began.
static std::vector<std::map<std::string, double>> demonstratedOnsetsPerTrial(const Solution& solution)
{
    std::vector<std::map<std::string, double>> perTrial;
    const auto trace = nlohmann::json::parse(solution.getParameters().evaluationTrace);
    for (const auto& trial : trace.at("trials"))
    {
        perTrial.push_back(trial.at("demonstratedOnsets").get<std::map<std::string, double>>());
    }
    return perTrial;
}

template <typename Task>
static void checkSequenceTimingDemonstration(const char* taskSlug)
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ taskSlug };

    Task solution(sequenceTimingTopology());
    solution.addFieldGene(makeFixedFieldGene(FieldGeneType::INPUT, 1));
    solution.addFieldGene(makeFixedFieldGene(FieldGeneType::OUTPUT, 2));
    solution.evaluate();

    const std::map<std::string, double> demonstrated{ { "X", 0.0 }, { "D", 100.0 }, { "C", 600.0 }, { "A", 800.0 }, { "B", 1800.0 } };
    const auto perTrial = demonstratedOnsetsPerTrial(solution);
    REQUIRE(perTrial.size() == 4);
    for (const auto& onsets : perTrial)
    {
        CHECK(onsets == demonstrated);
    }
}

// Serial order demonstrates its events evenly spaced; the sequence-timing tasks must keep their own intervals.
TEST_CASE("SequenceTiming tasks demonstrate D, C, A and B at 100, 600, 800 and 1800 after X", "[Solutions][SequenceTiming]")
{
    checkSequenceTimingDemonstration<SequenceTiming>("sequence-timing");
    checkSequenceTimingDemonstration<SequenceTimingBoostOrder>("sequence-timing-boost-order");
}

// Fresh gaps on every evaluation stop a solution from tuning its timing to a fixed set of gaps.
// One gap is drawn from each third of 300-1500, so every evaluation still spans short and long gaps.
TEST_CASE("SequenceTiming draws a fresh recall gap from each third of 300-1500 on every evaluation", "[Solutions][SequenceTiming]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "sequence-timing" };

    SequenceTiming solution(sequenceTimingTopology());
    solution.addFieldGene(makeFixedFieldGene(FieldGeneType::INPUT, 1));
    solution.addFieldGene(makeFixedFieldGene(FieldGeneType::OUTPUT, 2));

    const auto recallGaps = [&solution]() {
        solution.evaluate();
        std::vector<double> gaps;
        const auto trace = nlohmann::json::parse(solution.getParameters().evaluationTrace);
        for (const auto& trial : trace.at("trials"))
        {
            if (trial.at("kind") == "recall")
            {
                gaps.push_back(trial.at("gap").get<double>());
            }
        }
        return gaps;
    };

    const auto first = recallGaps();
    const auto second = recallGaps();
    REQUIRE(first.size() == 3);
    REQUIRE(second.size() == 3);
    for (size_t i = 0; i < first.size(); ++i)
    {
        CHECK(first[i] >= 300.0 + 400.0 * static_cast<double>(i));
        CHECK(first[i] <= 700.0 + 400.0 * static_cast<double>(i));
    }
    CHECK(first != second);
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

// Loads a saved solution for a sequence-timing task from tests/fixtures.
template <typename Task>
static Task loadSequenceTimingFixture(const std::string& fileName)
{
    const auto simulation = loadFixtureSimulation(fileName);
    return Task(sequenceTimingTopology(), *simulation);
}

// Saved from a from-scratch run: the output is active almost everywhere, all the time, and after the cue
// travelling waves sweep the field, so each event position sees a peak "rise". Recall credit is gated by how
// silent the output stayed before the cue, so this earns none of it.
TEST_CASE("SequenceTiming gives no recall credit to an output that is never silent before the cue", "[Solutions][SequenceTiming]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "sequence-timing" };

    auto solution = loadSequenceTimingFixture<SequenceTiming>("sequence-timing-wave-exploit.dnf");
    solution.evaluate();

    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 7);
    REQUIRE(partials[2] < 0.1); // the output is active outside recall
    CHECK(partials[3] < 0.05);
    CHECK(partials[4] < 0.05);
    CHECK(partials[5] < 0.05);
}

TEST_CASE("SequenceTimingBoost contract", "[Solutions][SequenceTiming]")
{
    resetGlobalState();
    checkSolutionContract<SequenceTimingBoost>(makeTopology(1, 1));
}

TEST_CASE("SequenceTimingBoost gives no recall credit to an output field that never peaks", "[Solutions][SequenceTiming]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "sequence-timing-boost" };

    SequenceTimingBoost solution(sequenceTimingTopology());
    solution.addFieldGene(makeFixedFieldGene(FieldGeneType::INPUT, 1));
    solution.addFieldGene(makeFixedFieldGene(FieldGeneType::OUTPUT, 2));
    solution.evaluate();

    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 7);
    CHECK(partials[2] == Catch::Approx(1.0));
    CHECK(partials[3] == 0.0);
    CHECK(partials[4] == 0.0);
    CHECK(partials[5] == 0.0);
}

// The hand-built boost-triggered reference: a uniform clamp, set by the first event and reset by the boost,
// freezes the integrator's traces between demonstration and recall, and a latch set by the boost stores the
// time from X to the first event. Every parameter lies inside the task's evolvable ranges. It must solve the
// task in the evaluator too.
TEST_CASE("SequenceTimingBoost scores the hand-built boost-triggered reference above 0.9", "[Solutions][SequenceTiming]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "sequence-timing-boost" };

    auto solution = loadSequenceTimingFixture<SequenceTimingBoost>("sequence-timing-boost-handbuilt.dnf");
    solution.evaluate();

    CHECK(solution.getFitness() > 0.9);
    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 7);
    CHECK(partials[3] > 0.95); // every event recalled
    CHECK(partials[4] > 0.9);  // at the demonstrated time after X
}

// Saved from evolved b2 runs. Run 18's output stays at its resting level between peaks; run 20 recalls the sequence
// at the right times, but its output sits about 650 below rest from the start.
TEST_CASE("SequenceTimingBoost gives full output-level credit to outputs that stay near their resting level", "[Solutions][SequenceTiming]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "sequence-timing-boost" };

    for (const auto* fixture : { "sequence-timing-boost-handbuilt.dnf", "sequence-timing-boost-evolved-b2.dnf" })
    {
        auto solution = loadSequenceTimingFixture<SequenceTimingBoost>(fixture);
        solution.evaluate();

        const auto& partials = solution.getParameters().partialFitness;
        REQUIRE(partials.size() == 7);
        CHECK(partials[6] == Catch::Approx(1.0));
    }
}

TEST_CASE("SequenceTimingBoost gives little output-level credit to an output sunk hundreds below its resting level", "[Solutions][SequenceTiming]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "sequence-timing-boost" };

    auto sunken = loadSequenceTimingFixture<SequenceTimingBoost>("sequence-timing-boost-sunken-output.dnf");
    sunken.evaluate();
    auto atRest = loadSequenceTimingFixture<SequenceTimingBoost>("sequence-timing-boost-evolved-b2.dnf");
    atRest.evaluate();

    const auto& partials = sunken.getParameters().partialFitness;
    REQUIRE(partials.size() == 7);
    CHECK(partials[6] < 0.2);
    CHECK(sunken.getFitness() < atRest.getFitness());
}

// Saved from a from-scratch run (run 23): it recalls D, C, A, B in order at a fixed lag after the demonstration,
// and its output also fires when X never comes.
TEST_CASE("SequenceTiming scales recall credit by how silent the output stays in the probe trial without X", "[Solutions][SequenceTiming]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "sequence-timing-boost-order" };

    auto solution = loadSequenceTimingFixture<SequenceTimingBoostOrder>("sequence-timing-boost-fixed-lag.dnf");
    solution.evaluate();

    const auto trace = nlohmann::json::parse(solution.getParameters().evaluationTrace);
    double probeSilence = 1.0;
    for (const auto& trial : trace.at("trials"))
    {
        if (trial.at("kind") == "probe")
        {
            probeSilence = trial.at("outputSilence").at("withoutCue").get<double>();
        }
    }
    REQUIRE(probeSilence < 0.9);

    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 7);
    CHECK(partials[3] <= probeSilence + 1e-9);
    CHECK(partials[4] <= probeSilence + 1e-9);
    CHECK(partials[5] <= probeSilence + 1e-9);
}

TEST_CASE("SequenceTimingBoost is registered with one input and one output", "[SolutionRegistry]")
{
    const auto* task = findTask("sequence-timing-boost");
    REQUIRE(task != nullptr);
    REQUIRE(task->templateFile == "sequence-timing-boost.dnf");
    REQUIRE(task->inputs == 1);
    REQUIRE(task->outputs == 1);
}

// Order credit of each recall trial, read from the evaluation trace.
static std::vector<double> recallOrderCredits(const Solution& solution)
{
    std::vector<double> credits;
    const auto trace = nlohmann::json::parse(solution.getParameters().evaluationTrace);
    for (const auto& trial : trace.at("trials"))
    {
        if (trial.at("kind") == "recall")
        {
            credits.push_back(trial.at("recall").at("order").get<double>());
        }
    }
    return credits;
}

TEST_CASE("SequenceTimingBoostOrder contract", "[Solutions][SequenceTiming]")
{
    resetGlobalState();
    checkSolutionContract<SequenceTimingBoostOrder>(makeTopology(1, 1));
}

TEST_CASE("SequenceTimingBoostOrder gives full order credit to the hand-built reference", "[Solutions][SequenceTiming]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "sequence-timing-boost-order" };

    auto solution = loadSequenceTimingFixture<SequenceTimingBoostOrder>("sequence-timing-boost-handbuilt.dnf");
    solution.evaluate();

    CHECK(solution.getFitness() > 0.9);
    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 7);
    CHECK(partials[4] > 0.9);
    const auto credits = recallOrderCredits(solution);
    REQUIRE(credits.size() == 3);
    for (const double credit : credits)
    {
        CHECK(credit == Catch::Approx(1.0));
    }
}

// Saved from a from-scratch boost run: a few hundred time units after the cue the output ignites at many
// positions at once, so every event gets a rising peak, but all of them at about the same time.
TEST_CASE("SequenceTimingBoostOrder gives almost no order credit to a burst of simultaneous peaks", "[Solutions][SequenceTiming]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "sequence-timing-boost-order" };

    auto solution = loadSequenceTimingFixture<SequenceTimingBoostOrder>("sequence-timing-boost-burst.dnf");
    solution.evaluate();

    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 7);
    REQUIRE(partials[3] > 0.9); // the burst does put a rising peak on every event
    const auto credits = recallOrderCredits(solution);
    REQUIRE(credits.size() == 3);
    for (const double credit : credits)
    {
        CHECK(credit < 0.3);
    }
    CHECK(partials[4] < 0.3);
}

TEST_CASE("SequenceTimingBoostOrder is registered with the boost template", "[SolutionRegistry]")
{
    const auto* task = findTask("sequence-timing-boost-order");
    REQUIRE(task != nullptr);
    REQUIRE(task->templateFile == "sequence-timing-boost.dnf");
    REQUIRE(task->inputs == 1);
    REQUIRE(task->outputs == 1);
}

TEST_CASE("SequenceTimingBoostOrderOnly contract", "[Solutions][SequenceTiming]")
{
    resetGlobalState();
    checkSolutionContract<SequenceTimingBoostOrderOnly>(makeTopology(1, 1));
}

TEST_CASE("SequenceTimingBoostOrderOnly scores recalled timing as order credit alone", "[Solutions][SequenceTiming]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "sequence-timing-boost-order-only" };

    auto solution = loadSequenceTimingFixture<SequenceTimingBoostOrderOnly>("sequence-timing-boost-handbuilt.dnf");
    solution.evaluate();

    CHECK(solution.getFitness() > 0.9);
    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 7);
    CHECK(partials[4] > 0.95);
}

// The burst stays silent before the cue, so no recall credit is gated away and the timing term is exactly the
// mean order credit of the three recall trials.
TEST_CASE("SequenceTimingBoostOrderOnly gives a burst of simultaneous peaks only its order credit for timing", "[Solutions][SequenceTiming]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "sequence-timing-boost-order-only" };

    auto solution = loadSequenceTimingFixture<SequenceTimingBoostOrderOnly>("sequence-timing-boost-burst.dnf");
    solution.evaluate();

    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 7);
    REQUIRE(partials[2] == Catch::Approx(1.0));
    const auto credits = recallOrderCredits(solution);
    REQUIRE(credits.size() == 3);
    const double meanCredit = std::accumulate(credits.begin(), credits.end(), 0.0) / static_cast<double>(credits.size());
    CHECK(partials[4] == Catch::Approx(meanCredit).margin(1e-9));
    CHECK(partials[4] < 0.3);
}

TEST_CASE("SequenceTimingBoostOrderOnly is registered with the boost template", "[SolutionRegistry]")
{
    const auto* task = findTask("sequence-timing-boost-order-only");
    REQUIRE(task != nullptr);
    REQUIRE(task->templateFile == "sequence-timing-boost.dnf");
    REQUIRE(task->inputs == 1);
    REQUIRE(task->outputs == 1);
}

TEST_CASE("SerialOrder contract", "[Solutions][SerialOrder]")
{
    resetGlobalState();
    checkSolutionContract<SerialOrder>(makeTopology(1, 1));
}

TEST_CASE("SerialOrderFitnessTest contract", "[Solutions][SerialOrderFitnessTest]")
{
    resetGlobalState();
    checkSolutionContract<SerialOrderFitnessTest>(makeTopology(1, 1));
}

// Runs are filed under the solution's name, and a population is built from clone() and copy(), so a copy of
// SerialOrder that kept its name or returned SerialOrder would evolve and save as the original task.
TEST_CASE("SerialOrderFitnessTest names its runs apart from SerialOrder's and clones and copies as itself", "[Solutions][SerialOrderFitnessTest]")
{
    resetGlobalState();

    SerialOrderFitnessTest solution(sequenceTimingTopology());
    solution.initialize();
    solution.buildPhenotype();

    CHECK(solution.getName() == "Serial Order Fitness Test");
    CHECK(solution.getName() != SerialOrder(sequenceTimingTopology()).getName());
    CHECK(std::dynamic_pointer_cast<SerialOrderFitnessTest>(solution.clone()) != nullptr);
    CHECK(std::dynamic_pointer_cast<SerialOrderFitnessTest>(solution.copy()) != nullptr);
}

// The reference is the design whose responses the fitness was written from: the hand-built serial-order design of
// 18 October 2026, with the competition in the output.
TEST_CASE("SerialOrderFitnessTest scores the reference design above 0.9 on every term", "[Solutions][SerialOrderFitnessTest]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order-fitness-test" };
    const auto simulation = loadFixtureSimulation("serial-order-fitness-test-reference.dnf");
    SerialOrderFitnessTest solution(sequenceTimingTopology(), *simulation);

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

// A saved solution carries what createPhenotypeEnvironment() adds, and the evaluation runs on that same schedule, so a
// saved design opened in dnf-composer replays what was scored.
TEST_CASE("SerialOrderFitnessTest saves its schedule with each solution, so a saved design replays its evaluation", "[Solutions][SerialOrderFitnessTest]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order-fitness-test" };
    const auto fixture = loadFixtureSimulation("serial-order-fitness-test-reference.dnf");
    SerialOrderFitnessTest solution(sequenceTimingTopology(), *fixture);
    const auto saveDirectory = std::filesystem::temp_directory_path() / "neat-dnfs-serial-order-fitness-test-replay";

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
TEST_CASE("SerialOrderFitnessTest traces the input and output at every point it scores them", "[Solutions][SerialOrderFitnessTest]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order-fitness-test" };
    auto solution = loadSequenceTimingFixture<SerialOrderFitnessTest>("serial-order-fitness-test-reference.dnf");

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

    // Each cycle's recall summary: every event peaked, first D, then C, A and B, and each was released.
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
            CHECK(!events.at(label).at("release").is_null());
        }
    }
}

// Two weighted from-scratch runs of 20 October 2026 converged on one output peak that rises before any X and is never
// let go: w1's at 138, near no event, and w2's at C. Under the old rule, one bump of the right size earned more than
// half of a recall point wherever it was: f4 0.53 and 0.63.
// - w1's earns nothing: only a peak near an event counts.
// - w2's earns nothing: recall is scored by order, and C counts only after D.
TEST_CASE("SerialOrderFitnessTest gives a peak held through recall no credit off an event or out of order", "[Solutions][SerialOrderFitnessTest]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order-fitness-test" };

    auto offEvent = loadSequenceTimingFixture<SerialOrderFitnessTest>("serial-order-fitness-test-held-peak-off-event.dnf");
    offEvent.evaluate();
    CHECK(offEvent.getParameters().partialFitness[3] < 0.05);

    auto atC = loadSequenceTimingFixture<SerialOrderFitnessTest>("serial-order-fitness-test-held-peak-before-cue.dnf");
    atC.evaluate();
    CHECK(atC.getParameters().partialFitness[3] < 0.05);
}

// The reference design with its done field's inhibition of the output (gk cg 7-2) weakened from -30 to -12. It recalls
// D, C, A and B in order every cycle, each alone, but each peak comes about 100 later, and a recalled event returns to
// about the rest of the field rather than below it. Scored at the reference's own recall times, it earned f4 0.40-0.47
// and f6 0.04. Scored by order, it earns most of the recall credit and part of the release.
TEST_CASE("SerialOrderFitnessTest pays recall in order whatever its pace, and a shallow release in part", "[Solutions][SerialOrderFitnessTest]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order-fitness-test" };

    auto weakRelease = loadSequenceTimingFixture<SerialOrderFitnessTest>("serial-order-fitness-test-weak-release.dnf");
    weakRelease.evaluate();

    const auto& partials = weakRelease.getParameters().partialFitness;
    CHECK(partials[3] > 0.8);
    CHECK(partials[5] > 0.4);
    CHECK(partials[5] < 0.9); // the reference's release, below the field, still earns more
}

// From-scratch run t1 of 20 October 2026: after X, D and C rise together and are held. Rising together is not
// recalling in order, so only D, the first, earns recall credit.
TEST_CASE("SerialOrderFitnessTest gives two events that peak together order credit for the first only", "[Solutions][SerialOrderFitnessTest]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order-fitness-test" };

    auto together = loadSequenceTimingFixture<SerialOrderFitnessTest>("serial-order-fitness-test-held-double-peak.dnf");
    together.evaluate();

    CHECK(together.getParameters().partialFitness[3] < 0.25);
}

// From-scratch run t6 of 20 October 2026 steps D, C, A in turn, a slot later than the reference. Each released event goes
// back to being pre-shaped rather than below the field. That is the start of a release, so it earns some f6, and
// stepping beats t1's held peaks.
TEST_CASE("SerialOrderFitnessTest pays stepping from one event to the next over holding them", "[Solutions][SerialOrderFitnessTest]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order-fitness-test" };

    auto stepping = loadSequenceTimingFixture<SerialOrderFitnessTest>("serial-order-fitness-test-late-stepping.dnf");
    stepping.evaluate();
    auto held = loadSequenceTimingFixture<SerialOrderFitnessTest>("serial-order-fitness-test-held-double-peak.dnf");
    held.evaluate();

    CHECK(stepping.getParameters().partialFitness[5] > 0.1);
    CHECK(stepping.getFitness() > held.getFitness() + 0.05);
}

// From-scratch run o1 of 21 October 2026 holds D, C, A and B as four self-sustained peaks in the input from their pulses
// on, reads them out as graded pre-shapes in the output, and steps through them one peak at a time. The input is a
// detection field: it peaks while an event is shown and lets go after. Pre-shapes now count as memory only as far as the
// input is at rest, so this keeps its recall credit but earns little for its memory, which earned f3 0.60 and f5 0.93.
TEST_CASE("SerialOrderFitnessTest gives little memory credit while the input holds the events", "[Solutions][SerialOrderFitnessTest]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order-fitness-test" };

    auto inputHolds = loadSequenceTimingFixture<SerialOrderFitnessTest>("serial-order-fitness-test-input-holds-items.dnf");
    inputHolds.evaluate();

    const auto& partials = inputHolds.getParameters().partialFitness;
    CHECK(partials[2] < 0.3);
    CHECK(partials[4] < 0.3);
    CHECK(partials[3] > 0.7);
}

// From-scratch run o2 of 21 October 2026: X pushes all four events over threshold within 20, highest pre-shape first,
// and all four fade together. Sampled every 10, they peak in order, so each earned most of a recall point: f4 0.54.
// Recall is one peak at a time, so each sample's recall credit is now shared among the peaks up together. The burst
// keeps a little, a start to build on, but far less than o1's recall one peak at a time.
TEST_CASE("SerialOrderFitnessTest shares recall credit among the peaks up together", "[Solutions][SerialOrderFitnessTest]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order-fitness-test" };

    auto burst = loadSequenceTimingFixture<SerialOrderFitnessTest>("serial-order-fitness-test-burst.dnf");
    burst.evaluate();
    auto oneAtATime = loadSequenceTimingFixture<SerialOrderFitnessTest>("serial-order-fitness-test-input-holds-items.dnf");
    oneAtATime.evaluate();

    const double burstRecall = burst.getParameters().partialFitness[3];
    CHECK(burstRecall < 0.4);
    CHECK(burstRecall > 0.05);
    CHECK(burstRecall < 0.5 * oneAtATime.getParameters().partialFitness[3]);
}

// From-scratch run t6 of 20 October 2026 forms a peak at D during the demonstration, before any X. Memory is pre-shapes
// below threshold, so a demonstration sample's memory now counts only as far as the output has no peak; it earned f3
// 0.82.
TEST_CASE("SerialOrderFitnessTest gives less memory credit while the output peaks during the demonstration", "[Solutions][SerialOrderFitnessTest]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order-fitness-test" };

    auto peaksEarly = loadSequenceTimingFixture<SerialOrderFitnessTest>("serial-order-fitness-test-late-stepping.dnf");
    peaksEarly.evaluate();

    CHECK(peaksEarly.getParameters().partialFitness[2] < 0.7);
}

// From-scratch run g2 of 20 October 2026: the output sits at threshold and crosses it by 0.0-0.1 at A and B exactly when
// they are due. Height was a twentieth of a recall point, so those crossings earned f4 0.30-0.38. A recalled peak now
// earns its credit in proportion to its height, in full from 2, and they earn almost nothing.
TEST_CASE("SerialOrderFitnessTest gives an output flickering at threshold almost no recall credit", "[Solutions][SerialOrderFitnessTest]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order-fitness-test" };

    auto flicker = loadSequenceTimingFixture<SerialOrderFitnessTest>("serial-order-fitness-test-threshold-flicker.dnf");
    flicker.evaluate();

    CHECK(flicker.getParameters().partialFitness[3] < 0.05);
}

// From-scratch run h3 of 20 October 2026 stores D, C and A in order, then from the first X sinks D and C into deep
// troughs without either ever peaking. Troughs at every earlier event were paid whether or not that event had been
// recalled, so this earned f6 0.89 and f7 0.75. A trough now earns credit only for an event that peaked when due in that
// cycle, so these earn nothing: f6 is 0, and f7 keeps only its half for an output with no peak.
TEST_CASE("SerialOrderFitnessTest pays a trough only at an event recalled first", "[Solutions][SerialOrderFitnessTest]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order-fitness-test" };

    auto troughs = loadSequenceTimingFixture<SerialOrderFitnessTest>("serial-order-fitness-test-troughs-without-recall.dnf");
    troughs.evaluate();

    const auto& partials = troughs.getParameters().partialFitness;
    REQUIRE(partials[3] == 0.0); // nothing was recalled
    CHECK(partials[5] < 0.05);
    CHECK(partials[6] == Catch::Approx(0.5).margin(0.05));
}

// With no connection into the output, it stays at rest: quiet through the demonstration, but with nothing stored,
// recalled or suppressed. Noise moves single samples by hundredths, so those terms come out near zero, not exactly.
TEST_CASE("SerialOrderFitnessTest gives no memory or recall credit to an output that never responds", "[Solutions][SerialOrderFitnessTest]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order-fitness-test" };
    SerialOrderFitnessTest solution(sequenceTimingTopology());
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
    CHECK(partials[6] == Catch::Approx(0.5).margin(0.05)); // after recall: quiet, but nothing suppressed
}

TEST_CASE("SerialOrder evaluate produces a bounded fitness", "[Solutions][SerialOrder]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order" };

    SerialOrder solution(sequenceTimingTopology());
    solution.initialize();

    REQUIRE_NOTHROW(solution.evaluate());
    REQUIRE(solution.getFitness() >= 0.0);
    REQUIRE(solution.getFitness() <= 1.0);
    REQUIRE(solution.getParameters().partialFitness.size() == 7);
}

TEST_CASE("SerialOrder gives no recall credit to an output field that never peaks", "[Solutions][SerialOrder]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order" };

    SerialOrder solution(sequenceTimingTopology());
    solution.addFieldGene(makeFixedFieldGene(FieldGeneType::INPUT, 1));
    solution.addFieldGene(makeFixedFieldGene(FieldGeneType::OUTPUT, 2));
    solution.evaluate();

    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 7);
    CHECK(partials[2] == Catch::Approx(1.0));
    CHECK(partials[3] == 0.0);
    CHECK(partials[4] == 0.0);
    CHECK(partials[5] == 0.0);
}

// The recall trial of a serial-order evaluation, read from its trace.
static nlohmann::json serialOrderRecallTrial(const Solution& solution)
{
    const auto trace = nlohmann::json::parse(solution.getParameters().evaluationTrace);
    for (const auto& trial : trace.at("trials"))
    {
        if (trial.at("kind") == "recall")
        {
            return trial;
        }
    }
    throw std::runtime_error("no recall trial in the trace");
}

TEST_CASE("SerialOrder cues three recall cycles of fresh random length after one demonstration", "[Solutions][SerialOrder]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order" };

    const auto cycleLengths = [] {
        SerialOrder solution(sequenceTimingTopology());
        solution.addFieldGene(makeFixedFieldGene(FieldGeneType::INPUT, 1));
        solution.addFieldGene(makeFixedFieldGene(FieldGeneType::OUTPUT, 2));
        solution.evaluate();
        const auto trial = serialOrderRecallTrial(solution);
        CHECK(trial.at("gap").get<double>() >= 300.0);
        CHECK(trial.at("gap").get<double>() <= 1500.0);
        std::vector<double> lengths;
        for (const auto& cycle : trial.at("cycles"))
        {
            lengths.push_back(cycle.at("length").get<double>());
        }
        return lengths;
    };

    const auto first = cycleLengths();
    const auto second = cycleLengths();
    REQUIRE(first.size() == 3);
    for (const double length : first)
    {
        CHECK(length >= 800.0);
        CHECK(length <= 1500.0);
    }
    CHECK(first != second);
}

// Serial order is scored on order alone, so the demonstration spaces X and the events evenly and closely: no interval
// stands out, and every evaluation is shorter than with the sequence-timing intervals.
TEST_CASE("SerialOrder demonstrates X, D, C, A and B evenly spaced, 200 apart", "[Solutions][SerialOrder]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order" };

    SerialOrder solution(sequenceTimingTopology());
    solution.addFieldGene(makeFixedFieldGene(FieldGeneType::INPUT, 1));
    solution.addFieldGene(makeFixedFieldGene(FieldGeneType::OUTPUT, 2));
    solution.evaluate();

    const std::map<std::string, double> demonstrated{ { "X", 0.0 }, { "D", 200.0 }, { "C", 400.0 }, { "A", 600.0 }, { "B", 800.0 } };
    const auto perTrial = demonstratedOnsetsPerTrial(solution);
    REQUIRE(perTrial.size() == 2); // the cued trial and the probe
    for (const auto& onsets : perTrial)
    {
        CHECK(onsets == demonstrated);
    }
}

// The hand-built competitive-queuing design: a slow integrator stores a primacy gradient (earlier events sit higher),
// a hidden selection field picks the highest remaining item and the output copies it, a done field suppresses each
// item once recalled, and every X clears the done field, so the whole sequence plays again on every trigger.
TEST_CASE("SerialOrder scores the hand-built design above 0.95, recalling the order on every trigger", "[Solutions][SerialOrder]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order" };

    auto solution = loadSequenceTimingFixture<SerialOrder>("serial-order-handbuilt.dnf");
    solution.evaluate();

    CHECK(solution.getFitness() > 0.95);
    const auto recallTrial = serialOrderRecallTrial(solution);
    REQUIRE(recallTrial.at("cycles").size() == SerialOrder::recallCycleCount);
    for (const auto& cycle : recallTrial.at("cycles"))
    {
        const auto& recall = cycle.at("recall");
        CHECK(recall.at("order").get<double>() == Catch::Approx(1.0));
        CHECK(recall.at("onsets").size() == 4);
        CHECK(recall.at("activityAtEvents").get<double>() == Catch::Approx(1.0));
        for (const auto& [label, isolation] : recall.at("isolationAtOnset").items())
        {
            INFO("event " << label);
            CHECK(isolation.get<double>() == Catch::Approx(1.0));
            CHECK(recall.at("peakHeightCredit").at(label).get<double>() == Catch::Approx(1.0));
        }
    }
}

// The earlier hand-built design, whose output did the competition itself, with weaker competition, a weaker done field
// and a slower output: in almost every cycle C rises 30-40 before D has gone, though each peak stands alone 50 after
// its own onset, which is all the isolation check used to look at. A peak must only rise once the previous one has
// gone, so this cannot earn full one-at-a-time credit.
TEST_CASE("SerialOrder gives no full one-at-a-time credit to a peak that rises before the previous one has gone", "[Solutions][SerialOrder]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order" };

    auto solution = loadSequenceTimingFixture<SerialOrder>("serial-order-overlapping-peaks.dnf");
    solution.evaluate();

    const auto recallTrial = serialOrderRecallTrial(solution);
    REQUIRE(recallTrial.at("cycles").size() == SerialOrder::recallCycleCount);
    const auto cyclesWithAPeakRisingOverAnother = std::ranges::count_if(recallTrial.at("cycles"), [](const auto& cycle) {
        const auto& isolationAtOnset = cycle.at("recall").at("isolationAtOnset");
        return std::ranges::any_of(isolationAtOnset, [](const auto& isolation) { return isolation.template get<double>() < 1.0; });
    });
    CHECK(cyclesWithAPeakRisingOverAnother >= 2);
    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 7);
    CHECK(partials[3] > 0.9); // the events are still recalled
    CHECK(partials[5] < 0.95);
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

static void checkFieldGeneEvolvable(const FieldGene& gene)
{
    using namespace dnf_composer::element;
    const auto field = gene.getNeuralField()->getParameters();
    INFO("field " << gene.getParameters().id);
    CHECK(withinRange(field.tau, NeuralFieldConstants::tauMinVal, NeuralFieldConstants::tauMaxVal));
    CHECK(withinRange(field.startingRestingLevel, NeuralFieldConstants::restingLevelMinVal, NeuralFieldConstants::restingLevelMaxVal));

    if (const auto gauss = std::dynamic_pointer_cast<GaussKernel>(gene.getKernel()))
    {
        const auto kernel = gauss->getParameters();
        CHECK(withinRange(kernel.width, GaussKernelConstants::widthMinVal, GaussKernelConstants::widthMaxVal));
        CHECK(withinRange(kernel.amplitude, GaussKernelConstants::ampMinVal, GaussKernelConstants::ampMaxVal));
        CHECK(withinRange(kernel.amplitudeGlobal, GaussKernelConstants::ampGlobalMinVal, GaussKernelConstants::ampGlobalMaxVal));
        return;
    }
    const auto mexicanHat = std::dynamic_pointer_cast<MexicanHatKernel>(gene.getKernel());
    REQUIRE(mexicanHat != nullptr);
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

// Evolving serial order from scratch is only a fair test if the search space contains a known solution.
TEST_CASE("SerialOrder's evolvable ranges reach every setting of the hand-built design", "[Solutions][SerialOrder]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order" };

    const auto solution = loadSequenceTimingFixture<SerialOrder>("serial-order-handbuilt.dnf");

    for (const auto& gene : solution.getGenome().getFieldGenes())
    {
        checkFieldGeneEvolvable(gene);
    }
    for (const auto& gene : solution.getGenome().getConnectionGenes())
    {
        checkConnectionGeneEvolvable(gene);
    }
}

// The fitness is written from the reference's responses, so evolving it from scratch is only a fair test if this
// task's ranges reach every setting of that design.
TEST_CASE("SerialOrderFitnessTest's evolvable ranges reach every setting of the reference design", "[Solutions][SerialOrderFitnessTest]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order-fitness-test" };

    const auto solution = loadSequenceTimingFixture<SerialOrderFitnessTest>("serial-order-fitness-test-reference.dnf");

    for (const auto& gene : solution.getGenome().getFieldGenes())
    {
        checkFieldGeneEvolvable(gene);
    }
    for (const auto& gene : solution.getGenome().getConnectionGenes())
    {
        checkConnectionGeneEvolvable(gene);
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

TEST_CASE("SerialOrder's hand-built design uses only connections evolution can make", "[Solutions][SerialOrder]")
{
    OutputSourceRuleGuard guard;
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order" };
    // Reachable without output feedback, which this task's config turns on.
    GenomeMutationConstants::allowOutputAsConnectionSource = false;

    checkOnlyConnectionsEvolutionCanMake(loadSequenceTimingFixture<SerialOrder>("serial-order-handbuilt.dnf"));
}

TEST_CASE("SequenceTimingBoost's hand-built reference uses only connections evolution can make", "[Solutions][SequenceTiming]")
{
    OutputSourceRuleGuard guard;
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "sequence-timing-boost" };
    // Reachable without output feedback, which this task's config turns on.
    GenomeMutationConstants::allowOutputAsConnectionSource = false;

    checkOnlyConnectionsEvolutionCanMake(
        loadSequenceTimingFixture<SequenceTimingBoost>("sequence-timing-boost-handbuilt.dnf"));
}

TEST_CASE("Genome lists the connections evolution cannot make, honouring allowOutputAsConnectionSource", "[genome][legal_connections]")
{
    OutputSourceRuleGuard guard;
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order" };
    GenomeMutationConstants::allowOutputAsConnectionSource = false;

    // Built from the earlier serial-order design, whose output fed its done field (gk cg 2-7).
    const auto outputFedDone = loadSequenceTimingFixture<SerialOrder>("serial-order-overlapping-peaks.dnf");
    const auto handBuilt = loadSequenceTimingFixture<SerialOrder>("serial-order-handbuilt.dnf");

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
    const ScopedTaskConfig taskConfig{ "serial-order" };
    GenomeMutationConstants::allowOutputAsConnectionSource = false;
    const auto* task = findTask("serial-order");
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

// Saved from serial-order run 2: for a few hundred after each X the output holds many broad peaks all over the field,
// which cross the event positions by chance. Re-evaluated it averaged 0.67, with f4 near 0.95. Recall credit is scaled
// by the share of output activity that lies at the event positions, so this earns little of it.
TEST_CASE("SerialOrder gives little recall credit to peaks sweeping all over the output after X", "[Solutions][SerialOrder]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order" };

    auto solution = loadSequenceTimingFixture<SerialOrder>("serial-order-chaotic-sweep.dnf");
    solution.evaluate();

    const auto recallTrial = serialOrderRecallTrial(solution);
    REQUIRE(recallTrial.at("cycles").size() == SerialOrder::recallCycleCount);
    for (const auto& cycle : recallTrial.at("cycles"))
    {
        CHECK(cycle.at("recall").at("activityAtEvents").get<double>() < 0.5);
    }
    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 7);
    CHECK(partials[3] < 0.25);
    CHECK(partials[4] < 0.25);
    CHECK(partials[5] < 0.25);
}

// The same design without the X boost clearing its done field: the sequence plays on the first trigger only.
TEST_CASE("SerialOrder gives a sequence recalled on the first trigger only a third of the recall credit", "[Solutions][SerialOrder]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order" };

    auto solution = loadSequenceTimingFixture<SerialOrder>("serial-order-single-recall.dnf");
    solution.evaluate();

    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 7);
    CHECK(partials[3] == Catch::Approx(1.0 / 3.0).margin(0.02));
    CHECK(partials[4] < 0.35);
}

// Saved from serial-order output-feedback run 2: the output sits just below threshold everywhere and crosses it by
// 0.0-0.2 at the event positions, in a different order each cycle. While any crossing counted as a recalled peak it
// re-evaluated at 0.64, with nearly full recalled-event credit for peaks a fiftieth of the hand-built design's.
TEST_CASE("SerialOrder gives little recall credit to an output that only flickers across threshold", "[Solutions][SerialOrder]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order" };

    auto solution = loadSequenceTimingFixture<SerialOrder>("serial-order-threshold-flicker.dnf");
    solution.evaluate();

    const auto recallTrial = serialOrderRecallTrial(solution);
    REQUIRE(recallTrial.at("cycles").size() == SerialOrder::recallCycleCount);
    for (const auto& cycle : recallTrial.at("cycles"))
    {
        for (const auto& [label, height] : cycle.at("recall").at("peakHeight").items())
        {
            INFO("event " << label);
            CHECK(height.get<double>() < 0.5);
        }
    }
    const auto& partials = solution.getParameters().partialFitness;
    REQUIRE(partials.size() == 7);
    CHECK(partials[3] < 0.25);
    CHECK(partials[4] < 0.1);
    CHECK(partials[5] < 0.1);
}

// Saved from output-feedback run 3: on every trigger D rises to about 2.4 and then B to about 1.3, one at a time;
// A only touches threshold and C never rises. Credit grows with height, so a low peak still earns some.
TEST_CASE("SerialOrder credits each recalled peak in proportion to its height, in full from a height of 2", "[Solutions][SerialOrder]")
{
    resetGlobalState();
    const ScopedTaskConfig taskConfig{ "serial-order" };

    auto solution = loadSequenceTimingFixture<SerialOrder>("serial-order-low-peaks.dnf");
    solution.evaluate();

    const auto recallTrial = serialOrderRecallTrial(solution);
    REQUIRE(recallTrial.at("cycles").size() == SerialOrder::recallCycleCount);
    bool sawPartialCredit = false;
    for (const auto& cycle : recallTrial.at("cycles"))
    {
        const auto& recall = cycle.at("recall");
        for (const auto& [label, height] : recall.at("peakHeight").items())
        {
            INFO("event " << label << ", peak height " << height.get<double>());
            const double credit = recall.at("peakHeightCredit").at(label).get<double>();
            CHECK(credit == Catch::Approx(std::min(1.0, height.get<double>() / 2.0)));
            sawPartialCredit = sawPartialCredit || (credit > 0.0 && credit < 1.0);
        }
    }
    CHECK(sawPartialCredit);
}

TEST_CASE("SerialOrder is registered with one input and one output", "[SolutionRegistry]")
{
    const auto* task = findTask("serial-order");
    REQUIRE(task != nullptr);
    REQUIRE(task->templateFile == "serial-order.dnf");
    REQUIRE(task->inputs == 1);
    REQUIRE(task->outputs == 1);
}

TEST_CASE("SerialOrderFitnessTest is registered with one input and one output, starting from serial order's template", "[SolutionRegistry]")
{
    const auto* task = findTask("serial-order-fitness-test");
    REQUIRE(task != nullptr);
    REQUIRE(task->templateFile == "serial-order.dnf");
    REQUIRE(task->inputs == 1);
    REQUIRE(task->outputs == 1);
}

TEST_CASE("SequenceTiming is registered with one input and one output", "[SolutionRegistry]")
{
    const auto* task = findTask("sequence-timing");
    REQUIRE(task != nullptr);
    REQUIRE(task->templateFile == "sequence-timing.dnf");
    REQUIRE(task->inputs == 1);
    REQUIRE(task->outputs == 1);
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

            REQUIRE(solution->getFitness() == Catch::Approx(weightedSum).margin(1e-9));
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


