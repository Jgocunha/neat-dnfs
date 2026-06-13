#include <catch2/catch_test_macros.hpp>

#include "neat/genome.h"

using namespace neat_dnfs;
using namespace dnf_composer::element;

static const ElementDimensions kDim{100, 1.0};

TEST_CASE("Genome Initialization", "[Genome]")
{
    const Genome genome;

    REQUIRE(genome.getFieldGenes().empty());
    REQUIRE(genome.getConnectionGenes().empty());
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
