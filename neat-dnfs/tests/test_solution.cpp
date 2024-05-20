#include <catch2/catch_test_macros.hpp>


#include "neat/solution.h"
#include "solutions/empty_solution.h"


using namespace neat_dnfs;

TEST_CASE("Solution Initialization", "[Solution]")
{
    SolutionTopology topology(3, 1);
    EmptySolution solution(topology);

    REQUIRE(solution.getGenome().getFieldGenes().empty());
    REQUIRE(solution.getGenome().getConnectionGenes().empty());
    REQUIRE(solution.getParameters().fitness == 0.0);
    REQUIRE(solution.getParameters().adjustedFitness == 0.0);
    REQUIRE(solution.getParameters().age == 0);
}

TEST_CASE("Solution Initialize Method", "[Solution]")
{
    SolutionTopology topology(3, 1);
    EmptySolution solution(topology);
    solution.initialize();

    auto fieldGenes = solution.getGenome().getFieldGenes();
    REQUIRE(fieldGenes.size() == 4); // 3 input genes + 1 output gene

    if constexpr (SolutionConstants::initialConnectionProbability == 0.0)
    {
		auto connectionGenes = solution.getGenome().getConnectionGenes();
		REQUIRE(connectionGenes.empty()); 
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

		auto initialFieldGenes = solution.getGenome().getFieldGenes();
		auto initialConnectionGenes = solution.getGenome().getConnectionGenes();

        REQUIRE_NOTHROW(solution.mutate());
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

TEST_CASE("Solution Increment Age", "[Solution]")
{
    const SolutionTopology topology(3, 1);
    EmptySolution solution(topology);
    solution.incrementAge();

    REQUIRE(solution.getParameters().age == 1);
}

TEST_CASE("Solution Set Adjusted Fitness", "[Solution]")
{
    const SolutionTopology topology(3, 1);
    EmptySolution solution(topology);
    solution.setAdjustedFitness(5.0);

    REQUIRE(solution.getParameters().adjustedFitness == 5.0);
}

TEST_CASE("Solution Add Field Gene", "[Solution]")
{
    const SolutionTopology topology(3, 1);
    EmptySolution solution(topology);
    solution.initialize();

    const FieldGene newGene(FieldGeneType::HIDDEN);
    solution.addFieldGene(newGene);

    auto fieldGenes = solution.getGenome().getFieldGenes();
    REQUIRE(std::ranges::find(fieldGenes, newGene) != fieldGenes.end());
}

TEST_CASE("Solution Add Connection Gene", "[Solution]")
{
    SolutionTopology topology(3, 1);
    EmptySolution solution(topology);
    solution.initialize();

    ConnectionTuple tuple(1, 2);
    ConnectionGene newGene(tuple);
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
