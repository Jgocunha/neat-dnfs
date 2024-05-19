#include <catch2/catch_test_macros.hpp>

#include "neat/genome.h"

using namespace neat_dnfs;
using namespace dnf_composer::tools::utils;

TEST_CASE("Genome Initialization", "[Genome]")
{
    const Genome genome;

    REQUIRE(genome.getFieldGenes().empty());
    REQUIRE(genome.getConnectionGenes().empty());
}

TEST_CASE("Add Genes", "[Genome]")
{
    Genome genome;

    genome.addInputGene();
    genome.addOutputGene();
    genome.addHiddenGene();

    const auto fieldGenes = genome.getFieldGenes();
    REQUIRE(fieldGenes.size() == 3);
    REQUIRE(fieldGenes[0].getParameters().type == FieldGeneType::INPUT);
    REQUIRE(fieldGenes[1].getParameters().type == FieldGeneType::OUTPUT);
    REQUIRE(fieldGenes[2].getParameters().type == FieldGeneType::HIDDEN);
}

TEST_CASE("Add Random Initial Connection Gene", "[Genome]")
{
    constexpr uint8_t attempts = 100;

    for (uint8_t i = 0; i < attempts; ++i)
	{
		Genome genome;

		genome.addInputGene();
		genome.addOutputGene();

		genome.addRandomInitialConnectionGene();

		const auto connectionGenes = genome.getConnectionGenes();
		REQUIRE(connectionGenes.size() == 1);
		REQUIRE(connectionGenes[0].getParameters().connectionTuple.inFieldGeneId != 0);
		REQUIRE(connectionGenes[0].getParameters().connectionTuple.outFieldGeneId != 0);
	}
}

TEST_CASE("Mutate Genome", "[Genome]")
{
    Genome genome;

    genome.addInputGene();
    genome.addOutputGene();
    genome.addHiddenGene();
    genome.addRandomInitialConnectionGene();

    REQUIRE_NOTHROW(genome.mutate());
}

TEST_CASE("Excess Genes", "[Genome]")
{
    Genome genome1;
    Genome genome2;

    genome1.addInputGene();
    genome1.addOutputGene();
    genome1.addRandomInitialConnectionGene(); // Innovation number 1

    genome2.addInputGene();
    genome2.addOutputGene();
    genome2.addRandomInitialConnectionGene(); // Innovation number 2

    genome2.addHiddenGene();
    genome2.addRandomInitialConnectionGene(); // Innovation number 3

    // E = 2 && 3
    const int excess = genome1.excessGenes(genome2);
    REQUIRE(excess == 2);
}

TEST_CASE("Disjoint Genes", "[Genome]")
{
    Genome genome1;
    Genome genome2;

    genome1.addInputGene();
    genome1.addOutputGene();
    genome1.addRandomInitialConnectionGene(); // Innovation number 1
    genome1.addHiddenGene();
    genome1.addRandomInitialConnectionGene(); // Innovation number 2

    genome2.addInputGene();
    genome2.addOutputGene();
    genome2.addRandomInitialConnectionGene(); // Innovation number 3

    genome2.addHiddenGene();
    genome2.addRandomInitialConnectionGene(); // Innovation number 4

    // D = 1 && 2
    const int disjoint = genome1.disjointGenes(genome2);
    REQUIRE(disjoint == 2);
}

TEST_CASE("Average Connection Difference", "[Genome]")
{
    Genome genome1;
    Genome genome2;

    genome1.addInputGene();
    genome1.addOutputGene();
    genome1.addRandomInitialConnectionGene();

    genome2.addInputGene();
    genome2.addOutputGene();
    genome2.addRandomInitialConnectionGene();

    const double avgDiff = genome1.averageConnectionDifference(genome2);
    REQUIRE(avgDiff >= 0.0);
}

TEST_CASE("Add Field Gene", "[Genome]")
{
    Genome genome;

    FieldGene fieldGene(FieldGeneType::HIDDEN);
    genome.addFieldGene(fieldGene);

    auto fieldGenes = genome.getFieldGenes();
    REQUIRE(fieldGenes.size() == 1);
    REQUIRE(fieldGenes[0] == fieldGene);
}

TEST_CASE("Add Connection Gene", "[Genome]")
{
    Genome genome;

    genome.addInputGene();
    genome.addOutputGene();

    const ConnectionTuple tuple(1, 2);
    ConnectionGene connectionGene(tuple);
    genome.addConnectionGene(connectionGene);

    auto connectionGenes = genome.getConnectionGenes();
    REQUIRE(connectionGenes.size() == 1);
    REQUIRE(connectionGenes[0] == connectionGene);
}

TEST_CASE("Contains Field Gene", "[Genome]")
{
    Genome genome;

    const FieldGene fieldGene(FieldGeneType::HIDDEN);
    genome.addFieldGene(fieldGene);

    REQUIRE(genome.containsFieldGene(fieldGene) == true);
}

TEST_CASE("Contains Connection Gene", "[Genome]") {
    Genome genome;

    genome.addInputGene();
    genome.addOutputGene();

    const ConnectionTuple tuple(1, 2);
    const ConnectionGene connectionGene(tuple);
    genome.addConnectionGene(connectionGene);

    REQUIRE(genome.containsConnectionGene(connectionGene) == true);
}

TEST_CASE("Get Connection Gene by Innovation Number", "[Genome]") {
    Genome genome;

    genome.addInputGene();
    genome.addOutputGene();

    const ConnectionTuple tuple(1, 2);
    ConnectionGene connectionGene(tuple);
    genome.addConnectionGene(connectionGene);

    auto retrievedGene = genome.getConnectionGeneByInnovationNumber(connectionGene.getInnovationNumber());
    REQUIRE(retrievedGene == connectionGene);
}

TEST_CASE("Clear Generational Innovations", "[Genome]")
{
    Genome genome;

    genome.addInputGene();
    genome.addOutputGene();
    genome.addRandomInitialConnectionGene();

    Genome::clearGenerationalInnovations();

    REQUIRE(Genome::getConnectionToInnovationNumberMap().empty());
}