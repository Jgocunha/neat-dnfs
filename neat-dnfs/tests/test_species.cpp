#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "neat/species.h"
#include "neat/solution.h"
#include "solutions/detection_instability.h"
#include "test_helpers.h"

using namespace neat_dnfs;
using namespace neat_dnfs::test;

TEST_CASE("Species::addSolution", "[Species]")
{
    Species species;
    const auto solution = std::make_shared<DetectionInstability>(makeTopology(1, 1));

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
    const auto solution = std::make_shared<DetectionInstability>(makeTopology(1, 1));

    species.addSolution(solution);
    REQUIRE(species.contains(solution));

    species.removeSolution(solution);
    REQUIRE_FALSE(species.contains(solution));
    REQUIRE(species.size() == 0);
}

TEST_CASE("Species::isCompatible", "[Species]")
{
    SECTION("Compatible solutions — identical small topology")
    {
        Species species;
        const auto representative = std::make_shared<DetectionInstability>(makeTopology(1, 1));
        const auto solution       = std::make_shared<DetectionInstability>(makeTopology(1, 1));

        species.setRepresentative(representative);
        REQUIRE(species.isCompatible(solution));
    }
}

TEST_CASE("Species::totalAdjustedFitness", "[Species]")
{
    Species species;
    const auto solution1 = std::make_shared<DetectionInstability>(makeTopology(1, 1));
    const auto solution2 = std::make_shared<DetectionInstability>(makeTopology(1, 1));

    solution1->initialize();
    solution2->initialize();

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
    const auto solution1 = std::make_shared<DetectionInstability>(makeTopology(1, 1));
    const auto solution2 = std::make_shared<DetectionInstability>(makeTopology(1, 1));

    solution1->initialize();
    solution2->initialize();
    solution1->evaluate();
    solution2->evaluate();

    species.addSolution(solution1);
    species.addSolution(solution2);
    species.setOffspringCount(2);

    REQUIRE_NOTHROW(species.crossover());
}

TEST_CASE("Species::sortMembersByFitness", "[Species]")
{
    Species species;
    const auto solution1 = std::make_shared<DetectionInstability>(makeTopology(1, 1));
    const auto solution2 = std::make_shared<DetectionInstability>(makeTopology(1, 1));
    const auto solution3 = std::make_shared<DetectionInstability>(makeTopology(1, 1));

    solution1->initialize();
    solution2->initialize();
    solution3->initialize();

    solution1->evaluate();
    solution2->evaluate();
    solution3->evaluate();

    species.addSolution(solution1);
    species.addSolution(solution2);
    species.addSolution(solution3);

    species.sortMembersByFitness();

    const auto sortedMembers = species.getMembers();
    REQUIRE(sortedMembers[0]->getParameters().fitness >= sortedMembers[1]->getParameters().fitness);
    REQUIRE(sortedMembers[1]->getParameters().fitness >= sortedMembers[2]->getParameters().fitness);
}

TEST_CASE("Species::getMembers", "[Species]")
{
    Species species;
    auto solution1 = std::make_shared<DetectionInstability>(makeTopology(1, 1));
    auto solution2 = std::make_shared<DetectionInstability>(makeTopology(1, 1));

    species.addSolution(solution1);
    species.addSolution(solution2);

    auto members = species.getMembers();
    REQUIRE(members.size() == 2);
    REQUIRE(members[0] == solution1);
    REQUIRE(members[1] == solution2);
}

TEST_CASE("Species::pruneWorsePerformingMembers", "[Species]")
{
    Species species;
    for (int i = 0; i < 10; ++i)
    {
        auto sol = std::make_shared<DetectionInstability>(makeTopology(1, 1));
        sol->initialize();
        sol->evaluate();
        species.addSolution(sol);
    }

    const size_t before = species.size();
    species.pruneWorsePerformingMembers(0.5);
    REQUIRE(species.size() < before);
    REQUIRE(species.size() > 0);
}

TEST_CASE("Species::replaceMembersWithOffspring", "[Species]")
{
    Species species;
    const auto solution1 = std::make_shared<DetectionInstability>(makeTopology(1, 1));
    const auto solution2 = std::make_shared<DetectionInstability>(makeTopology(1, 1));

    solution1->initialize();
    solution2->initialize();
    solution1->evaluate();
    solution2->evaluate();

    species.addSolution(solution1);
    species.addSolution(solution2);
    species.setOffspringCount(2);
    species.crossover();

    REQUIRE_NOTHROW(species.replaceMembersWithOffspring());
}

TEST_CASE("Species::setRepresentative and getRepresentative", "[Species]")
{
    Species species;
    const auto representative = std::make_shared<DetectionInstability>(makeTopology(1, 1));

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
