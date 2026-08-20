#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <type_traits>

#include "neat/species.h"
#include "neat/solution.h"
#include "solutions/detection_instability.h"
#include "test_helpers.h"
#include "test_stub_solution.h"

using namespace neat_dnfs;
using namespace neat_dnfs::test;

TEST_CASE("Species is movable", "[Species]")
{
    // Locks in move semantics: a user-declared no-op destructor would suppress
    // the implicit move ctor/assign and silently downgrade Species to copy-only.
    static_assert(std::is_move_constructible_v<Species>);
    static_assert(std::is_move_assignable_v<Species>);
    SUCCEED();
}

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

TEST_CASE("Species::pruneWorsePerformingMembers removes exactly ratio * size members", "[Species]")
{
    Species species;
    for (int i = 0; i < 10; ++i)
    {
        auto sol = std::make_shared<DetectionInstability>(makeTopology(1, 1));
        sol->initialize();
        sol->evaluate();
        species.addSolution(sol);
    }

    REQUIRE(species.size() == 10);

    species.pruneWorsePerformingMembers(0.8);

    // 10 members pruned at ratio 0.8 must leave exactly 2, not
    // floor(10 * 0.8 / (1 + 0.8)) = 4 as a shrinking-size-during-loop bug would produce.
    REQUIRE(species.size() == 2);
}

TEST_CASE("Species::pruneWorsePerformingMembers reassigns a pruned representative", "[Species]")
{
    // FixedFitnessSolution makes the worst member deterministic, so the test can
    // point the representative at a solution it knows pruning will remove.
    Species species;
    const auto worst = std::make_shared<FixedFitnessSolution>(makeTopology(1, 1), 0.1);
    const auto best = std::make_shared<FixedFitnessSolution>(makeTopology(1, 1), 0.9);
    worst->initialize();
    best->initialize();
    worst->evaluate();
    best->evaluate();

    species.addSolution(worst);
    species.addSolution(best);
    species.setRepresentative(worst);

    species.pruneWorsePerformingMembers(0.5);

    // The representative was pruned, so it must not survive as a stale pointer;
    // isCompatible() would otherwise measure against a non-member.
    const auto members = species.getMembers();
    REQUIRE(members.size() == 1);
    REQUIRE(members[0] == best);
    REQUIRE(species.getRepresentative() == best);
}

TEST_CASE("Species::pruneWorsePerformingMembers keeps a surviving representative", "[Species]")
{
    Species species;
    const auto worst = std::make_shared<FixedFitnessSolution>(makeTopology(1, 1), 0.1);
    const auto best = std::make_shared<FixedFitnessSolution>(makeTopology(1, 1), 0.9);
    worst->initialize();
    best->initialize();
    worst->evaluate();
    best->evaluate();

    species.addSolution(worst);
    species.addSolution(best);
    species.setRepresentative(best);

    species.pruneWorsePerformingMembers(0.5);

    // A representative that survived pruning must be left untouched.
    REQUIRE(species.getRepresentative() == best);
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

TEST_CASE("Species::randomlyAssignRepresentative picks a current member", "[Species]")
{
    Species species;
    const auto solution1 = std::make_shared<DetectionInstability>(makeTopology(1, 1));
    const auto solution2 = std::make_shared<DetectionInstability>(makeTopology(1, 1));
    species.addSolution(solution1);
    species.addSolution(solution2);

    species.randomlyAssignRepresentative();

    const auto members = species.getMembers();
    const auto representative = species.getRepresentative();
    REQUIRE(std::ranges::find(members, representative) != members.end());
}

TEST_CASE("Species::assignChampion tracks fitness improvement across generations", "[Species]")
{
    // hasFitnessImprovedOverTheLastGenerations() reads generationsSinceFitnessImproved,
    // which assignChampion() resets to 0 on improvement and increments otherwise.
    // FixedFitnessSolution reports a known positive fitness deterministically,
    // rather than depending on DetectionInstability::evaluate()'s task- and
    // seed-dependent outcome.
    Species species;
    const auto solution = std::make_shared<FixedFitnessSolution>(makeTopology(1, 1), 0.5);
    solution->initialize();
    species.addSolution(solution);

    solution->evaluate();
    species.assignChampion();
    REQUIRE(species.hasFitnessImprovedOverTheLastGenerations());

    // Champion's stored fitness (from the first assignChampion call) cannot be
    // exceeded without re-evaluating higher; repeated calls with no better
    // member increment generationsSinceFitnessImproved.
    for (int i = 0; i < PopulationConstants::generationsWithoutImprovementThresholdInSpecies + 1; ++i)
        species.assignChampion();

    REQUIRE_FALSE(species.hasFitnessImprovedOverTheLastGenerations());
}

TEST_CASE("Species::isExtinct is true after crossover with no members", "[Species]")
{
    Species species;
    const auto solution = std::make_shared<DetectionInstability>(makeTopology(1, 1));
    species.addSolution(solution);
    REQUIRE_FALSE(species.isExtinct());

    species.removeSolution(solution);
    species.setOffspringCount(0); // avoid the FATAL log path for members.empty() && offspringCount > 0
    species.crossover();

    REQUIRE(species.isExtinct());
}

TEST_CASE("Species::copyChampionToNextGeneration replaces the last member with the champion", "[Species]")
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
    species.assignChampion();
    const auto champion = species.getChampion();
    REQUIRE(champion != nullptr);

    species.copyChampionToNextGeneration();

    const auto members = species.getMembers();
    REQUIRE(std::ranges::find(members, champion) != members.end());
}

TEST_CASE("Species::crossover with a single member self-pairs without throwing", "[Species]")
{
    // crossover() has a distinct code path for members.size() == 1: it crosses
    // the lone member with itself (species.cpp:202-211) rather than skipping.
    Species species;
    const auto solution = std::make_shared<DetectionInstability>(makeTopology(1, 1));
    solution->initialize();
    solution->evaluate();

    species.addSolution(solution);
    species.setOffspringCount(3);

    REQUIRE_NOTHROW(species.crossover());
    REQUIRE_FALSE(species.isExtinct());
}
