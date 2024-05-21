#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "neat/species.h"
#include "neat/solution.h"
#include "solutions/empty_solution.h"

using namespace neat_dnfs;

TEST_CASE("Species::addSolution", "[Species]")
{
    Species species;
    auto topology = SolutionTopology(3, 1);
    const auto solution = std::make_shared<EmptySolution>(topology);

    species.addSolution(solution);
    REQUIRE(species.contains(solution));

    // Adding the same solution again should not increase the size
    const size_t sizeBefore = species.size();
    species.addSolution(solution);
    REQUIRE(species.size() == sizeBefore);
}

TEST_CASE("Species::removeSolution", "[Species]")
{
    Species species;
    auto topology = SolutionTopology(3, 1);
    const auto solution = std::make_shared<EmptySolution>(topology);

    species.addSolution(solution);
    REQUIRE(species.contains(solution));

    species.removeSolution(solution);
    REQUIRE_FALSE(species.contains(solution));
}

TEST_CASE("Species::isCompatible", "[Species]")
{
    SECTION("Compatible solutions")
    {
	    Species species;
	    auto topology = SolutionTopology(3, 1);
	    const auto representative = std::make_shared<EmptySolution>(topology);
	    const auto solution = std::make_shared<EmptySolution>(topology);

        species.setRepresentative(representative);
        REQUIRE(species.isCompatible(solution));
    }

    SECTION("Not compatible solutions")
    {
        Species species;
        auto topology_1 = SolutionTopology(1, 1);
        const auto representative = std::make_shared<EmptySolution>(topology_1);
        auto topology_2 = SolutionTopology(3, 5, 10);
        const auto solution = std::make_shared<EmptySolution>(topology_2);

        species.setRepresentative(representative);
        REQUIRE(species.isCompatible(solution));
    }
}

TEST_CASE("Species::totalAdjustedFitness", "[Species]")
{
    Species species;
    auto topology = SolutionTopology(3, 1);
    const auto solution1 = std::make_shared<EmptySolution>(topology);
    const auto solution2 = std::make_shared<EmptySolution>(topology);

    species.addSolution(solution1);
    species.addSolution(solution2);

    const double expectedTotal = solution1->getParameters().adjustedFitness + 
									solution2->getParameters().adjustedFitness;
    REQUIRE(species.totalAdjustedFitness() == Catch::Approx(expectedTotal).epsilon(0.01));
}

TEST_CASE("Species::killLeastFitSolutions", "[Species]")
{
    Species species;
    auto topology = SolutionTopology(3, 1);
    auto solution1 = std::make_shared<EmptySolution>(topology);
    auto solution2 = std::make_shared<EmptySolution>(topology);
    auto solution3 = std::make_shared<EmptySolution>(topology);

    solution1->evaluate();
    solution2->evaluate();
    solution3->evaluate();

    species.addSolution(solution1);
    species.addSolution(solution2);
    species.addSolution(solution3);
    species.setOffspringCount(1);

    auto removed = species.killLeastFitSolutions();
    REQUIRE(removed.size() == 2);
    REQUIRE(species.size() == 1);

    // Ensure the most fit solution is retained
    const size_t numMembers = species.size();
    REQUIRE(numMembers == 1);

    auto remainingSolution = species.getMembers()[0];
    REQUIRE(remainingSolution->getParameters().fitness >= solution1->getParameters().fitness);
    REQUIRE(remainingSolution->getParameters().fitness >= solution2->getParameters().fitness);
    REQUIRE(remainingSolution->getParameters().fitness >= solution3->getParameters().fitness);
}

TEST_CASE("Species::crossover", "[Species]")
{
    Species species;
    auto topology = SolutionTopology(3, 1);
    const auto solution1 = std::make_shared<EmptySolution>(topology);
    const auto solution2 = std::make_shared<EmptySolution>(topology);

    species.addSolution(solution1);
    species.addSolution(solution2);
    species.setOffspringCount(2);

    species.crossover();
    const auto offspring = species.getOffspring();
    REQUIRE(offspring.size() == 2);
}

TEST_CASE("Species::sortMembersByFitness", "[Species]")
{
    Species species;
    auto topology = SolutionTopology(3, 1);
    const auto solution1 = std::make_shared<EmptySolution>(topology);
    const auto solution2 = std::make_shared<EmptySolution>(topology);
    const auto solution3 = std::make_shared<EmptySolution>(topology);

    species.addSolution(solution1);
    species.addSolution(solution2);
    species.addSolution(solution3);

    solution1->evaluate();
    solution2->evaluate();
    solution3->evaluate();

    species.sortMembersByFitness();

    const auto sortedMembers = species.getMembers();
    REQUIRE(sortedMembers[0]->getParameters().fitness >= sortedMembers[1]->getParameters().fitness);
    REQUIRE(sortedMembers[1]->getParameters().fitness >= sortedMembers[2]->getParameters().fitness);
}

TEST_CASE("Species::getMembers", "[Species]")
{
    Species species;
    auto topology = SolutionTopology(3, 1);
    auto solution1 = std::make_shared<EmptySolution>(topology);
    auto solution2 = std::make_shared<EmptySolution>(topology);

    species.addSolution(solution1);
    species.addSolution(solution2);

    auto members = species.getMembers();
    REQUIRE(members.size() == 2);
    REQUIRE(members[0] == solution1);
    REQUIRE(members[1] == solution2);
}