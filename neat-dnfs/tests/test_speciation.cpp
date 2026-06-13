#include <catch2/catch_test_macros.hpp>

#include "neat/genome.h"
#include "neat/species.h"
#include "neat/population.h"
#include "solutions/detection_instability.h"
#include "test_helpers.h"

using namespace neat_dnfs;
using namespace neat_dnfs::test;
using namespace dnf_composer::element;

static const ElementDimensions kDim{100, 1.0};

TEST_CASE("Speciation: identical empty genomes have zero excess and disjoint genes", "[Speciation]")
{
    const Genome g1;
    const Genome g2;
    REQUIRE(g1.excessGenes(g2) == 0);
    REQUIRE(g1.disjointGenes(g2) == 0);
    REQUIRE(g1.averageConnectionDifference(g2) == 0.0);
}

TEST_CASE("Speciation: excessGenes counts correctly", "[Speciation]")
{
    Genome gSmall;
    gSmall.addInputGene(kDim);
    gSmall.addOutputGene(kDim);
    gSmall.addConnectionGene(ConnectionGene(ConnectionTuple(1, 2), 1));

    Genome gLarge;
    gLarge.addInputGene(kDim);
    gLarge.addOutputGene(kDim);
    gLarge.addConnectionGene(ConnectionGene(ConnectionTuple(1, 2), 1));
    gLarge.addConnectionGene(ConnectionGene(ConnectionTuple(1, 2), 2));
    gLarge.addConnectionGene(ConnectionGene(ConnectionTuple(1, 2), 3));

    // excessGenes returns thisExcessCount + otherExcessCount — result is symmetric
    // innov 2 and 3 in gLarge are beyond gSmall's max of 1 → total excess = 2
    REQUIRE(gSmall.excessGenes(gLarge) == 2);
    REQUIRE(gLarge.excessGenes(gSmall) == 2);
}

TEST_CASE("Speciation: disjointGenes counts correctly", "[Speciation]")
{
    Genome g1;
    g1.addInputGene(kDim);
    g1.addOutputGene(kDim);
    g1.addConnectionGene(ConnectionGene(ConnectionTuple(1, 2), 1));
    g1.addConnectionGene(ConnectionGene(ConnectionTuple(1, 2), 3));

    Genome g2;
    g2.addInputGene(kDim);
    g2.addOutputGene(kDim);
    g2.addConnectionGene(ConnectionGene(ConnectionTuple(1, 2), 2));
    g2.addConnectionGene(ConnectionGene(ConnectionTuple(1, 2), 3));

    // innov 1 only in g1, innov 2 only in g2 — both within overlapping range [1,3] → disjoint = 2
    REQUIRE(g1.disjointGenes(g2) == 2);
}

TEST_CASE("Speciation: averageConnectionDifference is non-negative", "[Speciation]")
{
    Genome g1;
    g1.addInputGene(kDim);
    g1.addOutputGene(kDim);
    g1.addConnectionGene(ConnectionGene(ConnectionTuple(1, 2), 1));

    Genome g2;
    g2.addInputGene(kDim);
    g2.addOutputGene(kDim);
    g2.addConnectionGene(ConnectionGene(ConnectionTuple(1, 2), 1));

    REQUIRE(g1.averageConnectionDifference(g2) >= 0.0);
}

TEST_CASE("Speciation: identical solutions are compatible with their species", "[Speciation]")
{
    const auto topology = makeTopology(3, 1);
    const auto rep = std::make_shared<DetectionInstability>(topology);
    const auto sol = std::make_shared<DetectionInstability>(topology);

    Species species;
    species.setRepresentative(rep);

    REQUIRE(species.isCompatible(sol));
}
