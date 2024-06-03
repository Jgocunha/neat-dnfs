#include <catch2/catch_test_macros.hpp>


#include "neat/solution.h"
#include "solutions/empty_solution.h"


using namespace neat_dnfs;
using namespace dnf_composer::element;

TEST_CASE("Solution Initialization", "[Solution]")
{
    const SolutionTopology topology(3, 1, 1, 0);

    SECTION("Valid Initialization")
    {
        EmptySolution solution(topology);

        REQUIRE(solution.getGenome().getFieldGenes().empty());
        REQUIRE(solution.getGenome().getConnectionGenes().empty());
        REQUIRE(solution.getParameters().fitness == 0.0);
        REQUIRE(solution.getParameters().adjustedFitness == 0.0);
        REQUIRE(solution.getParameters().age == 0);
    }

    SECTION("Invalid Initialization - Not enough input genes")
    {
        const SolutionTopology invalidTopology(0, 1, 1, 0);
        REQUIRE_THROWS_AS(EmptySolution(invalidTopology), std::invalid_argument);
    }

    SECTION("Invalid Initialization - Not enough output genes")
    {
        const SolutionTopology invalidTopology(3, 0, 1, 0);
        REQUIRE_THROWS_AS(EmptySolution(invalidTopology), std::invalid_argument);
    }
}

TEST_CASE("Solution Initialize Method", "[Solution]")
{
    SolutionTopology topology(3, 1, 1);
    EmptySolution solution(topology);
    solution.initialize();

    SECTION("Correct number of input genes")
    {
        REQUIRE(solution.getGenome().getFieldGenes().size() >= topology.numInputGenes);
    }

    SECTION("Correct number of output genes")
    {
        REQUIRE(solution.getGenome().getFieldGenes().size() >= topology.numOutputGenes);
    }

    SECTION("Correct number of hidden genes")
    {
        REQUIRE(solution.getGenome().getFieldGenes().size() >= topology.numHiddenGenes);
    }

    SECTION("Correct number of connection genes")
    {
        if constexpr (SolutionConstants::initialConnectionProbability == 0.0)
        {
            auto connectionGenes = solution.getGenome().getConnectionGenes();
            REQUIRE(connectionGenes.empty());
        }
	}
}

TEST_CASE("Solution Mutate Method", "[Solution]")
{
    static constexpr uint16_t attempts = 1000;

    for (uint16_t i = 0; i < attempts; ++i)
	{
		SolutionTopology topology(3, 1);
		EmptySolution solution(topology);
		solution.initialize();
		const size_t initialGenomeSize = solution.getGenomeSize();

		auto initialFieldGenes = solution.getGenome().getFieldGenes();
		auto initialConnectionGenes = solution.getGenome().getConnectionGenes();

        REQUIRE_NOTHROW(solution.mutate());
        // Mutation does not decrease genome size.
        REQUIRE(solution.getGenomeSize() >= initialGenomeSize);
	}
}

TEST_CASE("Solution Getters", "[Solution]")
{
    SolutionTopology topology(3, 1);
    EmptySolution solution(topology);

    REQUIRE(solution.getGenome().getFieldGenes().empty());
    REQUIRE(solution.getGenome().getConnectionGenes().empty());
    REQUIRE(solution.getFitness() == 0.0);
    REQUIRE(solution.getGenomeSize() == 0);
    REQUIRE(solution.getInnovationNumbers().empty());
}

TEST_CASE("Solution Build Phenotype", "[Solution]")
{
    SolutionTopology topology(3, 1, 1);
    EmptySolution solution(topology);
    solution.initialize();
    const auto genome = solution.getGenome();
    const auto fieldGenes = genome.getFieldGenes();
    const FieldGene& fieldGeneFirst = fieldGenes[0];
    REQUIRE(fieldGeneFirst.getParameters().type == FieldGeneType::INPUT);
    const FieldGene& fieldGeneSecondToLast = fieldGenes[3];
    REQUIRE(fieldGeneSecondToLast.getParameters().type == FieldGeneType::OUTPUT);
    solution.addConnectionGene(ConnectionGene(ConnectionTuple(fieldGeneFirst.getParameters().id, fieldGeneSecondToLast.getParameters().id)));

    solution.buildPhenotype();

    auto phenotype = solution.getPhenotype();
    phenotype.init();
    // 3 input genes = 3 * (1 neural field, 1 self-excitation kernel)
    // 1 output gene = 1 * (1 neural field, 1 self-excitation kernel)
    // 1 hidden gene = 1 * (1 neural field, 1 self-excitation kernel)
    // 1 connection gene = 1 * 1 interaction-kernel
    // total num. elements = (3 * 2) + (1 * 2) + (1 * 2) + (1 * 1) = 11

    REQUIRE(phenotype.getNumberOfElements() == 11);

    const auto elements = phenotype.getElements();
    // "nf 0" is input to "nf 3" and its own self-excitation kernel
    REQUIRE(phenotype.getElementsThatHaveSpecifiedElementAsInput(elements[0]->getUniqueName()).size() == 2);
}

TEST_CASE("Solution Age Increment", "[Solution]")
{
    const SolutionTopology topology(3, 1, 1, 0);
    EmptySolution solution(topology);

    const int initialAge = solution.getParameters().age;

    SECTION("Increment age") {
        solution.incrementAge();
        REQUIRE(solution.getParameters().age == initialAge + 1);
    }
}

TEST_CASE("Solution Fitness Management", "[Solution]")
{
    const SolutionTopology topology(3, 1, 1, 0);
    EmptySolution solution(topology);
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
    const SolutionTopology topology(3, 1);
    EmptySolution solution(topology);
    solution.initialize();

    const FieldGene newGene({ FieldGeneType::HIDDEN, 1 });
    solution.addFieldGene(newGene);

    auto fieldGenes = solution.getGenome().getFieldGenes();
    REQUIRE(std::ranges::find(fieldGenes, newGene) != fieldGenes.end());
}

TEST_CASE("Solution Add Connection Gene", "[Solution]")
{
    const SolutionTopology topology(3, 1);
    EmptySolution solution(topology);
    solution.initialize();

    const ConnectionTuple tuple(1, 2);
    const ConnectionGene newGene(tuple);
    solution.addConnectionGene(newGene);

    auto connectionGenes = solution.getGenome().getConnectionGenes();
    REQUIRE(std::ranges::find(connectionGenes, newGene) != connectionGenes.end());
}

TEST_CASE("Solution Contains Connection Gene", "[Solution]")
{
    const SolutionTopology topology(3, 1);
    EmptySolution solution(topology);
    solution.initialize();

    const ConnectionTuple tuple(1, 2);
    const ConnectionGene newGene(tuple);
    solution.addConnectionGene(newGene);

    REQUIRE(solution.containsConnectionGene(newGene) == true);
}

TEST_CASE("Solution Evaluation", "[Solution]")
{
    const SolutionTopology topology(3, 1, 1, 0);
    EmptySolution solution(topology);
    solution.initialize();

    SECTION("Evaluate without errors")
    {
        REQUIRE_NOTHROW(solution.evaluate());
    }
}

TEST_CASE("Solution Crossover", "[Solution]")
{
    SolutionTopology topology(3, 1, 1);
    const std::shared_ptr<EmptySolution> parent1 = std::make_shared<EmptySolution>(topology);
    const std::shared_ptr<EmptySolution> parent2 = std::make_shared<EmptySolution>(topology);
    parent1->initialize();
    parent2->initialize();

    auto offspring = parent1->crossover(parent2);

    SECTION("Offspring is not null")
    {
        REQUIRE(offspring != nullptr);
    }

    SECTION("Offspring has non-zero genome size")
    {
        REQUIRE(!offspring->getGenome().getFieldGenes().empty());
    }
}

TEST_CASE("Solution Phenotype Translation", "[Solution]")
{
    const SolutionTopology topology(3, 1, 1, 0);
    EmptySolution solution(topology);
    solution.initialize();

    SECTION("Build phenotype") {
        REQUIRE_NOTHROW(solution.buildPhenotype());
    }

    SECTION("Clear phenotype") {
        REQUIRE_NOTHROW(solution.clearPhenotype());
    }
}
