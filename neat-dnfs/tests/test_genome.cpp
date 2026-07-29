#include <catch2/catch_test_macros.hpp>

#include <thread>
#include <vector>
#include <algorithm>
#include <map>

#include "neat/genome.h"
#include "test_helpers.h"

using namespace neat_dnfs;
using namespace neat_dnfs::test;
using namespace dnf_composer::element;

static const ElementDimensions kDim{100, 1.0};

TEST_CASE("Genome Initialization", "[Genome]")
{
    const Genome genome;

    REQUIRE(genome.getFieldGenes().empty());
    REQUIRE(genome.getConnectionGenes().empty());
    REQUIRE(genome.isEmpty());
}

TEST_CASE("Genome::isEmpty is false once any gene is added", "[Genome]")
{
    Genome genome;
    REQUIRE(genome.isEmpty());

    genome.addInputGene(kDim);
    REQUIRE_FALSE(genome.isEmpty());
}

TEST_CASE("Add Field Genes", "[Genome]")
{
    Genome genome;

    genome.addInputGene(kDim);
    genome.addOutputGene(kDim);
    genome.addHiddenGene(FieldGene({FieldGeneType::HIDDEN, 3}));

    const auto fieldGenes = genome.getFieldGenes();
    REQUIRE(fieldGenes.size() == 3);
    REQUIRE(fieldGenes[0].getParameters().type == FieldGeneType::INPUT);
    REQUIRE(fieldGenes[1].getParameters().type == FieldGeneType::OUTPUT);
    REQUIRE(fieldGenes[2].getParameters().type == FieldGeneType::HIDDEN);
}

TEST_CASE("Add Connection Gene", "[Genome]")
{
    Genome genome;
    genome.addInputGene(kDim);
    genome.addOutputGene(kDim);

    const auto fieldGenes = genome.getFieldGenes();
    const int inId  = fieldGenes[0].getParameters().id;
    const int outId = fieldGenes[1].getParameters().id;

    const ConnectionTuple tuple(inId, outId);
    ConnectionGene connectionGene(tuple, 1);
    genome.addConnectionGene(connectionGene);

    const auto connectionGenes = genome.getConnectionGenes();
    REQUIRE(connectionGenes.size() == 1);
    REQUIRE(connectionGenes[0] == connectionGene);
}

TEST_CASE("Excess Genes", "[Genome]")
{
    Genome genome1;
    genome1.addInputGene(kDim);
    genome1.addOutputGene(kDim);
    genome1.addConnectionGene(ConnectionGene(ConnectionTuple(1, 2), 1));

    Genome genome2;
    genome2.addInputGene(kDim);
    genome2.addOutputGene(kDim);
    genome2.addConnectionGene(ConnectionGene(ConnectionTuple(1, 2), 1));
    genome2.addHiddenGene(FieldGene({FieldGeneType::HIDDEN, 3}));
    genome2.addConnectionGene(ConnectionGene(ConnectionTuple(1, 3), 2));
    genome2.addConnectionGene(ConnectionGene(ConnectionTuple(3, 2), 3));

    // genome2 has innov 2 and 3 beyond genome1's max of 1 → excess = 2
    const int excess = genome1.excessGenes(genome2);
    REQUIRE(excess == 2);
}

TEST_CASE("Disjoint Genes", "[Genome]")
{
    Genome genome1;
    genome1.addInputGene(kDim);
    genome1.addOutputGene(kDim);
    genome1.addConnectionGene(ConnectionGene(ConnectionTuple(1, 2), 1));
    genome1.addConnectionGene(ConnectionGene(ConnectionTuple(1, 2), 3));

    Genome genome2;
    genome2.addInputGene(kDim);
    genome2.addOutputGene(kDim);
    genome2.addConnectionGene(ConnectionGene(ConnectionTuple(1, 2), 2));
    genome2.addConnectionGene(ConnectionGene(ConnectionTuple(1, 2), 3));

    // innov 1 is in genome1 only, innov 2 is in genome2 only → disjoint = 2
    const int disjoint = genome1.disjointGenes(genome2);
    REQUIRE(disjoint == 2);
}

TEST_CASE("Average Connection Difference", "[Genome]")
{
    Genome genome1;
    genome1.addInputGene(kDim);
    genome1.addOutputGene(kDim);
    genome1.addConnectionGene(ConnectionGene(ConnectionTuple(1, 2), 1));

    Genome genome2;
    genome2.addInputGene(kDim);
    genome2.addOutputGene(kDim);
    genome2.addConnectionGene(ConnectionGene(ConnectionTuple(1, 2), 1));

    const double avgDiff = genome1.averageConnectionDifference(genome2);
    REQUIRE(avgDiff >= 0.0);
}

TEST_CASE("Add Field Gene (public overload)", "[Genome]")
{
    Genome genome;

    FieldGene fieldGene({ FieldGeneType::HIDDEN, 1 });
    genome.addFieldGene(fieldGene);

    auto fieldGenes = genome.getFieldGenes();
    REQUIRE(fieldGenes.size() == 1);
    REQUIRE(fieldGenes[0] == fieldGene);
}

TEST_CASE("Contains Field Gene", "[Genome]")
{
    Genome genome;

    const FieldGene fieldGene({ FieldGeneType::HIDDEN, 1 });
    genome.addFieldGene(fieldGene);

    REQUIRE(genome.containsFieldGene(fieldGene) == true);
}

TEST_CASE("Contains Connection Gene", "[Genome]")
{
    Genome genome;
    genome.addInputGene(kDim);
    genome.addOutputGene(kDim);

    const ConnectionTuple tuple(1, 2);
    const ConnectionGene connectionGene(tuple, 1);
    genome.addConnectionGene(connectionGene);

    REQUIRE(genome.containsConnectionGene(connectionGene) == true);
}

TEST_CASE("Get Connection Gene by Innovation Number", "[Genome]")
{
    Genome genome;
    genome.addInputGene(kDim);
    genome.addOutputGene(kDim);

    const ConnectionTuple tuple(1, 2);
    const ConnectionGene connectionGene(tuple, 42);
    genome.addConnectionGene(connectionGene);

    auto retrievedGene = genome.getConnectionGeneByInnovationNumber(42);
    REQUIRE(retrievedGene == connectionGene);
}

TEST_CASE("Clear Generational Innovations", "[Genome]")
{
    // After clearing, newly added connections should get fresh innovation numbers
    Genome::clearGenerationalInnovations();
    Genome genome;
    genome.addInputGene(kDim);
    genome.addOutputGene(kDim);
    genome.addConnectionGene(ConnectionGene(ConnectionTuple(1, 2), 1));
    REQUIRE(genome.getConnectionGenes().size() == 1);

    Genome::clearGenerationalInnovations();
    // No assert on private map — verify via observable state (genes still present)
    REQUIRE(genome.getConnectionGenes().size() == 1);
}

TEST_CASE("Mutate Genome", "[Genome]")
{
    constexpr uint16_t attempts = 1000;

    for (uint16_t i = 0; i < attempts; ++i)
    {
        Genome genome;
        genome.addInputGene(kDim);
        genome.addOutputGene(kDim);
        genome.addHiddenGene(FieldGene({FieldGeneType::HIDDEN, 3}));
        genome.addHiddenGene(FieldGene({FieldGeneType::HIDDEN, 4}));
        genome.addHiddenGene(FieldGene({FieldGeneType::HIDDEN, 5}));
        genome.addConnectionGene(ConnectionGene(ConnectionTuple(1, 3), 1));
        genome.addConnectionGene(ConnectionGene(ConnectionTuple(3, 2), 2));

        REQUIRE_NOTHROW(genome.mutate());
    }
}

// Regression test for issue #3 (innovation number thread safety). The
// statics globalInnovationNumber and connectionTupleAndInnovationNumberWithinGeneration
// are guarded by Genome::innovationMutex (see genome.cpp addConnectionGene/addGene).
//
// The dedup table is shared across all genomes by design: identical structural
// mutations (same ConnectionTuple) occurring in different genomes within the same
// generation are SUPPOSED to receive the same innovation number — that isn't a
// race, it's the NEAT innovation-number contract. The actual invariant a lock
// failure would violate is narrower: (a) a given tuple must always map to the
// SAME innovation number everywhere it appears, and (b) two DIFFERENT tuples must
// never be assigned the same innovation number (which is what a lost
// globalInnovationNumber++ from an unsynchronised read-modify-write would cause).
//
// This passes with or without the mutex on MSVC (a lost increment needs unlucky
// timing this test cannot force). Its value is making CI's ThreadSanitizer job
// (Linux/macOS only) flag a data race if the lock is ever removed.
TEST_CASE("Genome::globalInnovationNumber is consistent under concurrent mutation", "[Genome]")
{
    // Other TEST_CASEs in this process may have left stale tuple -> innovation
    // entries in the shared generational map (it is a process-global static,
    // not reset between TEST_CASEs). Clear it so this test's own tuples dedupe
    // only against each other, not against unrelated genomes from earlier tests.
    Genome::clearGenerationalInnovations();

    const unsigned threadCount = std::max(2u, std::thread::hardware_concurrency());
    constexpr int mutationsPerThread = 50;

    std::vector<std::vector<ConnectionGene>> connectionGenesByThread(threadCount);
    std::vector<std::thread> threads;
    threads.reserve(threadCount);

    for (unsigned t = 0; t < threadCount; ++t)
    {
        threads.emplace_back([&connectionGenesByThread, t]()
            {
                Genome genome;
                genome.addInputGene(kDim);
                genome.addOutputGene(kDim);
                genome.addHiddenGene(FieldGene({ FieldGeneType::HIDDEN, 3 }));
                genome.addHiddenGene(FieldGene({ FieldGeneType::HIDDEN, 4 }));
                genome.addHiddenGene(FieldGene({ FieldGeneType::HIDDEN, 5 }));

                for (int i = 0; i < mutationsPerThread; ++i)
                    genome.mutate();

                connectionGenesByThread[t] = genome.getConnectionGenes();
            });
    }

    for (auto& thread : threads)
        thread.join();

    std::map<int, ConnectionTuple> tupleByInnovation; // innovation -> the tuple that first claimed it
    std::map<ConnectionTuple, int> innovationByTuple; // tuple -> its assigned innovation
    for (const auto& connectionGenes : connectionGenesByThread)
    {
        for (const auto& connectionGene : connectionGenes)
        {
            const auto params = connectionGene.getParameters();
            const auto [it, inserted] = tupleByInnovation.try_emplace(params.innovationNumber, params.connectionTuple);
            if (!inserted)
            {
                // Same innovation number seen again — must be for the same tuple.
                REQUIRE(it->second == params.connectionTuple);
            }

            const auto [tupleIt, tupleInserted] =
                innovationByTuple.try_emplace(params.connectionTuple, params.innovationNumber);
            if (!tupleInserted)
            {
                // Same tuple seen again — must have been assigned the same innovation number.
                REQUIRE(tupleIt->second == params.innovationNumber);
            }
        }
    }
}

TEST_CASE("Genome::removeConnectionGene removes the gene by innovation number", "[Genome]")
{
    Genome genome;
    genome.addInputGene(kDim);
    genome.addOutputGene(kDim);
    genome.addConnectionGene(ConnectionGene(ConnectionTuple(1, 2), 7));
    REQUIRE(genome.getConnectionGenes().size() == 1);

    genome.removeConnectionGene(7);

    REQUIRE(genome.getConnectionGenes().empty());
}

TEST_CASE("Genome::removeConnectionGene throws for an unknown innovation number", "[Genome]")
{
    Genome genome;
    genome.addInputGene(kDim);
    genome.addOutputGene(kDim);
    genome.addConnectionGene(ConnectionGene(ConnectionTuple(1, 2), 7));

    REQUIRE_THROWS_AS(genome.removeConnectionGene(999), std::invalid_argument);
}

TEST_CASE("Genome metrics for two identical genomes are all zero", "[Genome]")
{
    // ConnectionGene(tuple, innov) randomizes its kernel parameters, so two
    // separately constructed genes with the same tuple/innov are NOT
    // parameter-identical. Use the explicit-kernel constructor so both
    // genomes truly match, isolating averageConnectionDifference at 0.0.
    const dnf_composer::element::GaussKernelParameters gkp{ 5.0, 3.0, false, false };

    Genome genome1;
    genome1.addInputGene(kDim);
    genome1.addOutputGene(kDim);
    genome1.addConnectionGene(ConnectionGene(ConnectionTuple(1, 2), 1, gkp));

    Genome genome2;
    genome2.addInputGene(kDim);
    genome2.addOutputGene(kDim);
    genome2.addConnectionGene(ConnectionGene(ConnectionTuple(1, 2), 1, gkp));

    REQUIRE(genome1.excessGenes(genome2) == 0);
    REQUIRE(genome1.disjointGenes(genome2) == 0);
    REQUIRE(genome1.averageConnectionDifference(genome2) == 0.0);
}

TEST_CASE("Genome metrics for two completely disjoint genomes", "[Genome]")
{
    Genome genome1;
    genome1.addInputGene(kDim);
    genome1.addOutputGene(kDim);
    genome1.addConnectionGene(ConnectionGene(ConnectionTuple(1, 2), 1));

    Genome genome2;
    genome2.addInputGene(kDim);
    genome2.addOutputGene(kDim);
    genome2.addConnectionGene(ConnectionGene(ConnectionTuple(1, 2), 2));

    // No shared innovation numbers: genome1's {1} is entirely within genome2's
    // range [0, max=2], so it's disjoint, not excess; genome2's {2} exceeds
    // genome1's max of 1, so it's excess.
    REQUIRE(genome1.disjointGenes(genome2) == 1);
    REQUIRE(genome2.excessGenes(genome1) == 1);
    // No matching innovation numbers between the genomes to average over.
    REQUIRE(genome1.averageConnectionDifference(genome2) == 0.0);
}

TEST_CASE("Genome::clearLastMutations empties getMutationsInLastGeneration", "[Genome]")
{
    Genome genome;
    genome.addInputGene(kDim);
    genome.addOutputGene(kDim);
    genome.addConnectionGene(ConnectionGene(ConnectionTuple(1, 2), 1));
    genome.mutate();

    genome.clearLastMutations();

    REQUIRE(genome.getMutationsInLastGeneration().empty());
}

TEST_CASE("Genome::mutate over many iterations always produces valid gene ids", "[Genome]")
{
    // Covers the private random-id pickers (getRandomGeneId, getRandomGeneIdByType(s))
    // indirectly, since they have no public access path and no friend declaration.
    resetGlobalState();
    Genome genome;
    genome.addInputGene(kDim);
    genome.addOutputGene(kDim);
    genome.addHiddenGene(FieldGene({ FieldGeneType::HIDDEN, 3 }));

    for (int i = 0; i < 200; ++i)
    {
        genome.mutate();
        for (const auto& connectionGene : genome.getConnectionGenes())
        {
            const auto params = connectionGene.getParameters();
            REQUIRE(genome.containsFieldGene(genome.getFieldGeneById(params.connectionTuple.inFieldGeneId)));
            REQUIRE(genome.containsFieldGene(genome.getFieldGeneById(params.connectionTuple.outFieldGeneId)));
        }
    }
}
