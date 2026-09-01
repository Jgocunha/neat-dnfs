#include <catch2/catch_test_macros.hpp>

#include <dnf_composer/exceptions/exception.h>

#include "solutions/memory_instability.h"
#include "solutions/selection_instability.h"
#include "solutions/memory_trace.h"
#include "solutions/delayed_match_to_sample.h"
#include "solutions/inhibition_of_return.h"
#include "solutions/and.h"
#include "solutions/xor.h"
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
