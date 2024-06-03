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
    REQUIRE(species.size() == 1);

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
    REQUIRE(species.size() == 0);
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
        REQUIRE_FALSE(!species.isCompatible(solution));
    }
}

TEST_CASE("Species::totalAdjustedFitness", "[Species]")
{
    Species species;
    auto topology = SolutionTopology(3, 1);
    const auto solution1 = std::make_shared<EmptySolution>(topology);
    const auto solution2 = std::make_shared<EmptySolution>(topology);

    solution1->evaluate();
    solution2->evaluate();

    species.addSolution(solution1);
    species.addSolution(solution2);

    const double expectedTotal = solution1->getParameters().adjustedFitness +
        solution2->getParameters().adjustedFitness;
    REQUIRE(species.totalAdjustedFitness() == Catch::Approx(expectedTotal).epsilon(0.01));
}

TEST_CASE("Species::crossover", "[Species]")
{
    Species species;
    auto topology = SolutionTopology(3, 1);
    const auto solution1 = std::make_shared<EmptySolution>(topology);
    const auto solution2 = std::make_shared<EmptySolution>(topology);

    solution1->initialize();
    solution2->initialize();
    solution1->evaluate();
    solution2->evaluate();

    species.addSolution(solution1);
    species.addSolution(solution2);
    species.setOffspringCount(2);

    species.crossover();
    const auto offspring = species.getOffspring();
    REQUIRE(offspring.size() == 2);

    for (const auto& child : offspring) 
    {
        REQUIRE(child != nullptr);
        REQUIRE(!child->getGenome().getFieldGenes().empty());
    }
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

TEST_CASE("Species::selectElitesAndLeastFit", "[Species]")
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

    species.selectElitesAndLeastFit();

    auto elites = species.getElites();
    auto leastFit = species.getLeastFit();

    REQUIRE(!elites.empty());
    REQUIRE(!leastFit.empty());
    REQUIRE(elites.size() + leastFit.size() == 3);

    for (const auto& elite : elites) {
        REQUIRE(std::find(leastFit.begin(), leastFit.end(), elite) == leastFit.end());
    }
}

TEST_CASE("Species::updateMembers", "[Species]")
{
    Species species;
    auto topology = SolutionTopology(3, 1);
    const auto solution1 = std::make_shared<EmptySolution>(topology);
    const auto solution2 = std::make_shared<EmptySolution>(topology);
    const auto solution3 = std::make_shared<EmptySolution>(topology);

    solution1->initialize();
    solution2->initialize();
    solution3->initialize();

    solution1->evaluate();
    solution2->evaluate();
    solution3->evaluate();

    species.addSolution(solution1);
    species.addSolution(solution2);
    species.addSolution(solution3);

    species.selectElitesAndLeastFit();
    const auto numElites = static_cast<size_t>(std::ceil((1 - PopulationConstants::killRatio) * static_cast<double>(species.size())));
    const size_t numLeastFit = species.size() - numElites;
    const size_t remaining = species.size() - numLeastFit;
	species.crossover();
    species.updateMembers();

    const auto members = species.getMembers();
    REQUIRE(members.size() == remaining);
}

TEST_CASE("Species::clearOffspring", "[Species]")
{
    Species species;
    auto topology = SolutionTopology(3, 1);
    const auto solution1 = std::make_shared<EmptySolution>(topology);
    const auto solution2 = std::make_shared<EmptySolution>(topology);

    species.addSolution(solution1);
    species.addSolution(solution2);
    species.setOffspringCount(2);

    species.crossover();
    REQUIRE(species.getOffspring().size() == 2);

    species.clearOffspring();
    REQUIRE(species.getOffspring().empty());
}

TEST_CASE("Species::setRepresentative and getRepresentative", "[Species]")
{
    Species species;
    auto topology = SolutionTopology(3, 1);
    const auto representative = std::make_shared<EmptySolution>(topology);

    species.setRepresentative(representative);
    REQUIRE(species.getRepresentative() == representative);
}

TEST_CASE("Species::setOffspringCount and getOffspringCount", "[Species]")
{
    Species species;
    constexpr uint16_t count = 5;
    species.setOffspringCount(count);
    REQUIRE(species.getOffspringCount() == count);
}
