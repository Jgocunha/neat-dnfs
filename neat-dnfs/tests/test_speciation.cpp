#include <catch2/catch_test_macros.hpp>

#include "neat/genome.h"
#include "neat/species.h"
#include "neat/population.h"
#include "solutions/detection_instability.h"
#include "test_helpers.h"
#include "test_population_access.h"

using namespace neat_dnfs;
using namespace neat_dnfs::test;
using namespace dnf_composer::element;

static const ElementDimensions kDim{100, 1.0};

TEST_CASE("Speciation: identical empty genomes have zero excess and disjoint genes", "[Speciation]")
{
    const Genome g1;
    const Genome g2;
    REQUIRE(g1.excessGenes(g2) == 0);
    REQUIRE(g1.disjointGenes(g2) == 0);
    REQUIRE(g1.averageConnectionDifference(g2) == 0.0);
}

TEST_CASE("Speciation: excessGenes counts correctly", "[Speciation]")
{
    Genome gSmall;
    gSmall.addInputGene(kDim);
    gSmall.addOutputGene(kDim);
    gSmall.addConnectionGene(ConnectionGene(ConnectionTuple(1, 2), 1));

    Genome gLarge;
    gLarge.addInputGene(kDim);
    gLarge.addOutputGene(kDim);
    gLarge.addConnectionGene(ConnectionGene(ConnectionTuple(1, 2), 1));
    gLarge.addConnectionGene(ConnectionGene(ConnectionTuple(1, 2), 2));
    gLarge.addConnectionGene(ConnectionGene(ConnectionTuple(1, 2), 3));

    // excessGenes returns thisExcessCount + otherExcessCount — result is symmetric
    // innov 2 and 3 in gLarge are beyond gSmall's max of 1 → total excess = 2
    REQUIRE(gSmall.excessGenes(gLarge) == 2);
    REQUIRE(gLarge.excessGenes(gSmall) == 2);
}

TEST_CASE("Speciation: disjointGenes counts correctly", "[Speciation]")
{
    Genome g1;
    g1.addInputGene(kDim);
    g1.addOutputGene(kDim);
    g1.addConnectionGene(ConnectionGene(ConnectionTuple(1, 2), 1));
    g1.addConnectionGene(ConnectionGene(ConnectionTuple(1, 2), 3));

    Genome g2;
    g2.addInputGene(kDim);
    g2.addOutputGene(kDim);
    g2.addConnectionGene(ConnectionGene(ConnectionTuple(1, 2), 2));
    g2.addConnectionGene(ConnectionGene(ConnectionTuple(1, 2), 3));

    // innov 1 only in g1, innov 2 only in g2 — both within overlapping range [1,3] → disjoint = 2
    REQUIRE(g1.disjointGenes(g2) == 2);
}

TEST_CASE("Speciation: averageConnectionDifference is non-negative", "[Speciation]")
{
    Genome g1;
    g1.addInputGene(kDim);
    g1.addOutputGene(kDim);
    g1.addConnectionGene(ConnectionGene(ConnectionTuple(1, 2), 1));

    Genome g2;
    g2.addInputGene(kDim);
    g2.addOutputGene(kDim);
    g2.addConnectionGene(ConnectionGene(ConnectionTuple(1, 2), 1));

    REQUIRE(g1.averageConnectionDifference(g2) >= 0.0);
}

TEST_CASE("Speciation: identical solutions are compatible with their species", "[Speciation]")
{
    const auto topology = makeTopology(3, 1);
    const auto rep = std::make_shared<DetectionInstability>(topology);
    const auto sol = std::make_shared<DetectionInstability>(topology);

    Species species;
    species.setRepresentative(rep);

    REQUIRE(species.isCompatible(sol));
}

// Regression test for a bug where two Species objects ended up with the same
// representative Solution: removeSolution() erased a departing member from
// `members` but left `representative` pointing at it if it happened to be the
// representative, even though that solution could no longer belong to this
// species (an individual can only represent the species it is a member of).
// Population::assignToSpecies() calls removeSolution() on a solution's old
// species every time speciation moves it elsewhere, so this is not an edge
// case -- it is the normal path every generation.
TEST_CASE("Speciation: removing the representative from a species invalidates it", "[Speciation]")
{
    const auto topology = makeTopology(3, 1);
    const auto representative = std::make_shared<DetectionInstability>(topology);
    const auto otherMember = std::make_shared<DetectionInstability>(topology);

    Species species;
    species.addSolution(representative);
    species.addSolution(otherMember);
    species.setRepresentative(representative);

    species.removeSolution(representative);

    const auto newRepresentative = species.getRepresentative();
    REQUIRE(newRepresentative != representative);
    REQUIRE(species.contains(newRepresentative));
}

// Same scenario, but the removed representative was the species' only member:
// there is nothing left to reassign from, so the representative must not be
// left dangling on a solution that just left the species.
TEST_CASE("Speciation: removing the sole representative from a species clears it", "[Speciation]")
{
    const auto topology = makeTopology(3, 1);
    const auto representative = std::make_shared<DetectionInstability>(topology);

    Species species;
    species.addSolution(representative);
    species.setRepresentative(representative);

    species.removeSolution(representative);

    REQUIRE(species.size() == 0);
    REQUIRE(species.getRepresentative() != representative);
}

// Reproduces the exact cross-species collision: a solution transferring from
// speciesA to speciesB (as Population::assignToSpecies does every generation)
// must not leave speciesA's representative pointing at a solution that no
// longer belongs to speciesA -- even though the very same solution is now a
// legitimate member/representative of speciesB.
TEST_CASE("Speciation: a solution moving between species does not leave two species sharing a representative", "[Speciation]")
{
    const auto topology = makeTopology(3, 1);
    const auto migrating = std::make_shared<DetectionInstability>(topology);
    const auto otherMember = std::make_shared<DetectionInstability>(topology);

    Species speciesA;
    speciesA.addSolution(migrating);
    speciesA.addSolution(otherMember);
    speciesA.setRepresentative(migrating);

    Species speciesB;
    speciesA.removeSolution(migrating); // migrating leaves A for B, as assignToSpecies() does
    speciesB.addSolution(migrating);
    speciesB.setRepresentative(migrating);

    const auto repA = speciesA.getRepresentative();
    const auto repB = speciesB.getRepresentative();
    REQUIRE(repA != repB);
}

// Regression test for issue #59 (moving representative): Population::assignToSpecies()
// called species->randomlyAssignRepresentative() after every addSolution,
// including when a solution joined an ALREADY EXISTING species -- so the
// compatibility reference moved mid-pass instead of staying fixed for the
// whole assignment pass, a departure from standard NEAT (compatibility is
// measured against the previous generation's representative). A brand-new
// species may still pick an initial representative; only re-randomizing on
// top of an existing one is the bug.
TEST_CASE("Speciation: an existing species' representative stays fixed for the whole assignment pass", "[Speciation][Population]")
{
    resetGlobalState();
    const auto topology = makeTopology(1, 1);
    // All solutions share the exact same connection-less topology, so every
    // one of them is mutually compatible (zero excess/disjoint/connection-
    // difference) -- they must all land in a single species within one
    // Population::speciate() pass.
    const PopulationParameters parameters(11, 5, 1.1, false);
    Population population(parameters, std::make_shared<DetectionInstability>(topology), false);
    population.initialize();

    PopulationTestAccess::speciate(population);

    const auto speciesList = population.getSpeciesList();
    REQUIRE(speciesList.size() == 1);
    REQUIRE(speciesList[0]->size() == 11);

    const auto representative = speciesList[0]->getRepresentative();
    REQUIRE(representative != nullptr);

    // The first solution processed becomes the new species' initial
    // representative (picking one when a species is brand-new is fine); it
    // must not have been re-randomized as the other 10 solutions were folded
    // into that same, already-existing species during this single pass.
    const auto solutions = population.getSolutions();
    REQUIRE(representative == solutions.front());
}
