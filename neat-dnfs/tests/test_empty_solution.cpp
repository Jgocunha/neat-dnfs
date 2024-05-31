#include <catch2/catch_test_macros.hpp>

#include "solutions/empty_solution.h"

using namespace neat_dnfs;

TEST_CASE("EmptySolution::evaluate", "[EmptySolution]")
{
    static constexpr uint16_t attempts = 100;

    for (uint16_t i = 0; i < attempts; ++i)
    {
	    SolutionTopology topology(3, 1); // 3 input genes, 1 output gene
		EmptySolution solution(topology);

		solution.evaluate();

		const double fitness = solution.getParameters().fitness;
		REQUIRE(fitness >= 0.0);
		REQUIRE(fitness <= 1.0);
	}
}

TEST_CASE("EmptySolution::clone", "[EmptySolution]")
{
    SolutionTopology topology(3, 1); // 3 input genes, 1 output gene
    EmptySolution solution(topology);

    solution.evaluate();

    SolutionPtr clonedSolution = solution.clone();

    REQUIRE(clonedSolution != nullptr);
    REQUIRE(clonedSolution->getParameters().fitness == 0.0);
    REQUIRE(clonedSolution->getGenome().getFieldGenes().size() == solution.getGenome().getFieldGenes().size());
    REQUIRE(clonedSolution->getGenome().getConnectionGenes().size() == solution.getGenome().getConnectionGenes().size());
}

TEST_CASE("EmptySolution::crossover", "[EmptySolution]")
{
    static constexpr uint16_t attempts = 100;

    for (uint16_t i = 0; i < attempts; ++i)
    {
        SolutionTopology topology(3, 1); // 3 input genes, 1 output gene
        const std::shared_ptr<EmptySolution> solution1 = std::make_shared<EmptySolution>(topology);
        const std::shared_ptr<EmptySolution> solution2 = std::make_shared<EmptySolution>(topology);

        solution1->initialize();
        solution2->initialize();
        solution1->evaluate();
        solution2->evaluate();

        SolutionPtr offspring = solution1->crossover(solution2);

        REQUIRE(offspring != nullptr);

        auto fitness1 = solution1->getParameters().fitness;
        auto fitness2 = solution2->getParameters().fitness;
        auto offspringFitness = offspring->getParameters().fitness;

        // Ensure offspring has inherited genes from either parent
        const auto& offspringFieldGenes = offspring->getGenome().getFieldGenes();
        const auto& parent1FieldGenes = solution1->getGenome().getFieldGenes();
        const auto& parent2FieldGenes = solution2->getGenome().getFieldGenes();

        for (const auto& gene : offspringFieldGenes) {
            bool foundInParent1 = std::any_of(parent1FieldGenes.begin(), parent1FieldGenes.end(),
                [&gene](const FieldGene& parentGene) {
                    return parentGene.getParameters().id == gene.getParameters().id;
                });
            bool foundInParent2 = std::any_of(parent2FieldGenes.begin(), parent2FieldGenes.end(),
                [&gene](const FieldGene& parentGene) {
                    return parentGene.getParameters().id == gene.getParameters().id;
                });
            const bool foundInEitherParents = foundInParent1 || foundInParent2;
            REQUIRE(foundInEitherParents);
        }

        const auto& offspringConnectionGenes = offspring->getGenome().getConnectionGenes();
        const auto& parent1ConnectionGenes = solution1->getGenome().getConnectionGenes();
        const auto& parent2ConnectionGenes = solution2->getGenome().getConnectionGenes();

        for (const auto& gene : offspringConnectionGenes) {
            bool foundInParent1 = std::any_of(parent1ConnectionGenes.begin(), parent1ConnectionGenes.end(),
                [&gene](const ConnectionGene& parentGene) {
                    return parentGene.getInnovationNumber() == gene.getInnovationNumber();
                });
            bool foundInParent2 = std::any_of(parent2ConnectionGenes.begin(), parent2ConnectionGenes.end(),
                [&gene](const ConnectionGene& parentGene) {
                    return parentGene.getInnovationNumber() == gene.getInnovationNumber();
                });
            const bool foundInEitherParents = foundInParent1 || foundInParent2;
            REQUIRE(foundInEitherParents);
        }
	}
}
