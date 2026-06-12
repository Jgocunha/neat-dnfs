#include <catch2/catch_test_macros.hpp>

#include "neat/solution.h"
#include "solutions/detection_instability.h"
#include "test_helpers.h"

using namespace neat_dnfs;
using namespace neat_dnfs::test;
using namespace dnf_composer::element;

TEST_CASE("Solution Initialization", "[Solution]")
{
    SECTION("Valid Initialization")
    {
        const auto topology = makeTopology(1, 1);
        DetectionInstability solution(topology);

        REQUIRE(solution.getGenome().getFieldGenes().empty());
        REQUIRE(solution.getGenome().getConnectionGenes().empty());
        REQUIRE(solution.getParameters().fitness == 0.0);
        REQUIRE(solution.getParameters().adjustedFitness == 0.0);
        REQUIRE(solution.getParameters().age == 0);
    }

    SECTION("Invalid Initialization - Not enough input genes")
    {
        REQUIRE_THROWS_AS(
            DetectionInstability(SolutionTopology{ {
                {FieldGeneType::OUTPUT, dnf_composer::element::ElementDimensions{DimensionConstants::xSize, DimensionConstants::dx}},
            }}),
            std::invalid_argument
        );
    }

    SECTION("Invalid Initialization - Not enough output genes")
    {
        REQUIRE_THROWS_AS(
            DetectionInstability(SolutionTopology{ {
                {FieldGeneType::INPUT, dnf_composer::element::ElementDimensions{DimensionConstants::xSize, DimensionConstants::dx}},
            }}),
            std::invalid_argument
        );
    }
}

TEST_CASE("Solution Initialize Method", "[Solution]")
{
    const auto topology = makeTopology(1, 1);
    DetectionInstability solution(topology);
    solution.initialize();

    const auto fieldGenes = solution.getGenome().getFieldGenes();
    const long inputs  = std::count_if(fieldGenes.begin(), fieldGenes.end(), [](const auto& g){ return g.getParameters().type == FieldGeneType::INPUT; });
    const long outputs = std::count_if(fieldGenes.begin(), fieldGenes.end(), [](const auto& g){ return g.getParameters().type == FieldGeneType::OUTPUT; });

    REQUIRE(inputs  == 1);
    REQUIRE(outputs == 1);
}

TEST_CASE("Solution Mutate Method", "[Solution]")
{
    static constexpr uint16_t attempts = 1000;

    for (uint16_t i = 0; i < attempts; ++i)
    {
        auto topology = makeTopology(1, 1);
        DetectionInstability solution(topology);
        solution.initialize();
        const size_t initialGenomeSize = solution.getGenomeSize();

        REQUIRE_NOTHROW(solution.mutate());
        REQUIRE(solution.getGenomeSize() >= initialGenomeSize);
    }
}

TEST_CASE("Solution Getters", "[Solution]")
{
    auto topology = makeTopology(1, 1);
    DetectionInstability solution(topology);

    REQUIRE(solution.getGenome().getFieldGenes().empty());
    REQUIRE(solution.getGenome().getConnectionGenes().empty());
    REQUIRE(solution.getFitness() == 0.0);
    REQUIRE(solution.getGenomeSize() == 0);
    REQUIRE(solution.getInnovationNumbers().empty());
}

TEST_CASE("Solution Build Phenotype", "[Solution]")
{
    auto topology = makeTopology(1, 1);
    DetectionInstability solution(topology);
    solution.initialize();
    const auto genome = solution.getGenome();
    const auto fieldGenes = genome.getFieldGenes();

    const FieldGene& firstInput  = fieldGenes[0]; // input
    const FieldGene& firstOutput = fieldGenes[1]; // output

    REQUIRE(firstInput.getParameters().type == FieldGeneType::INPUT);
    REQUIRE(firstOutput.getParameters().type == FieldGeneType::OUTPUT);

    solution.addConnectionGene(ConnectionGene(
        ConnectionTuple(firstInput.getParameters().id, firstOutput.getParameters().id), 99));

    solution.buildPhenotype();

    auto phenotype = solution.getPhenotype();
    phenotype.init();
    REQUIRE(phenotype.getNumberOfElements() > 0);
}

TEST_CASE("Solution Age Increment", "[Solution]")
{
    const auto topology = makeTopology(1, 1);
    DetectionInstability solution(topology);

    const int initialAge = solution.getParameters().age;

    solution.incrementAge();
    REQUIRE(solution.getParameters().age == initialAge + 1);
}

TEST_CASE("Solution Fitness Management", "[Solution]")
{
    const auto topology = makeTopology(1, 1);
    DetectionInstability solution(topology);
    solution.initialize();

    SECTION("Initial fitness is zero")
    {
        REQUIRE(solution.getFitness() == 0.0);
    }

    SECTION("Set adjusted fitness")
    {
        constexpr double adjustedFitness = 0.75;
        solution.setAdjustedFitness(adjustedFitness);
        REQUIRE(solution.getParameters().adjustedFitness == adjustedFitness);
    }
}

TEST_CASE("Solution Add Field Gene", "[Solution]")
{
    auto topology = makeTopology(1, 1);
    DetectionInstability solution(topology);
    solution.initialize();

    const FieldGene newGene({ FieldGeneType::HIDDEN, 999 });
    solution.addFieldGene(newGene);

    auto fieldGenes = solution.getGenome().getFieldGenes();
    REQUIRE(std::ranges::find(fieldGenes, newGene) != fieldGenes.end());
}

TEST_CASE("Solution Add Connection Gene", "[Solution]")
{
    auto topology = makeTopology(1, 1);
    DetectionInstability solution(topology);
    solution.initialize();

    const auto fieldGenes = solution.getGenome().getFieldGenes();
    const int id1 = fieldGenes[0].getParameters().id; // input
    const int id2 = fieldGenes[1].getParameters().id; // output

    const ConnectionTuple tuple(id1, id2);
    const ConnectionGene newGene(tuple, 0);
    solution.addConnectionGene(newGene);

    auto connectionGenes = solution.getGenome().getConnectionGenes();
    REQUIRE(std::ranges::find(connectionGenes, newGene) != connectionGenes.end());
}

TEST_CASE("Solution Contains Connection Gene", "[Solution]")
{
    auto topology = makeTopology(1, 1);
    DetectionInstability solution(topology);
    solution.initialize();

    const auto fieldGenes = solution.getGenome().getFieldGenes();
    const int id1 = fieldGenes[0].getParameters().id; // input
    const int id2 = fieldGenes[1].getParameters().id; // output

    const ConnectionTuple tuple(id1, id2);
    const ConnectionGene newGene(tuple, 0);
    solution.addConnectionGene(newGene);

    REQUIRE(solution.containsConnectionGene(newGene) == true);
}

TEST_CASE("Solution Evaluation", "[Solution]")
{
    const auto topology = makeTopology(1, 1);
    DetectionInstability solution(topology);
    solution.initialize();

    REQUIRE_NOTHROW(solution.evaluate());
}

TEST_CASE("Solution Crossover", "[Solution]")
{
    auto topology = makeTopology(1, 1);
    const auto parent1 = std::make_shared<DetectionInstability>(topology);
    const auto parent2 = std::make_shared<DetectionInstability>(topology);
    parent1->initialize();
    parent2->initialize();

    auto offspring = parent1->crossover(parent2);

    REQUIRE(offspring != nullptr);
    REQUIRE(!offspring->getGenome().getFieldGenes().empty());
}

TEST_CASE("Solution Phenotype Translation", "[Solution]")
{
    const auto topology = makeTopology(1, 1);
    DetectionInstability solution(topology);
    solution.initialize();

    SECTION("Build phenotype")
    {
        REQUIRE_NOTHROW(solution.buildPhenotype());
    }

    SECTION("Clear phenotype")
    {
        REQUIRE_NOTHROW(solution.clearPhenotype());
    }
}
