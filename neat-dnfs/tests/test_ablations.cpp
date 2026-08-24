#include <catch2/catch_test_macros.hpp>

#include <set>
#include <utility>

#include "constants.h"
#include "neat_tools/ablation_presets.h"
#include "neat/genome.h"
#include "neat/species.h"
#include "solutions/and.h"
#include "test_helpers.h"

using namespace neat_dnfs;
using namespace neat_dnfs::test;

namespace
{
    struct AblationFlagGuard
    {
        ~AblationFlagGuard()
        {
            AblationConstants::reset();
        }
    };

    // sources = 2 inputs + hiddenCount; targets = hiddenCount + 1 output;
    // hiddenCount self-loops (hidden id as both source and target) are illegal.
    int maxLegalConnectionsFor(int hiddenCount)
    {
        return (2 + hiddenCount) * (hiddenCount + 1) - hiddenCount;
    }

    Genome buildGenomeWithHidden(int hiddenCount)
    {
        using namespace dnf_composer::element;
        const ElementDimensions dims{ DimensionConstants::xSize, DimensionConstants::dx };
        Genome genome;
        genome.addInputGene(dims);
        genome.addInputGene(dims);
        genome.addOutputGene(dims);
        for (int i = 0; i < hiddenCount; ++i)
            genome.addHiddenGene(dims);
        return genome;
    }

    std::set<std::pair<int, int>> connectionTupleSet(const Genome& genome)
    {
        std::set<std::pair<int, int>> tuples;
        for (const auto& cg : genome.getConnectionGenes())
            tuples.emplace(cg.getInFieldGeneId(), cg.getOutFieldGeneId());
        return tuples;
    }

    bool allConnectionsEnabled(const Genome& genome)
    {
        for (const auto& cg : genome.getConnectionGenes())
            if (!cg.isEnabled())
                return false;
        return true;
    }
}

TEST_CASE("Ablation test harness is wired up", "[ablations]")
{
    REQUIRE(true);
}

TEST_CASE("Genome::legalConnectionTuples() enumerates every legal src->tgt pair exactly once", "[genome][legal_connections]")
{
    for (int hiddenCount = 0; hiddenCount <= 5; ++hiddenCount)
    {
        resetGlobalState();
        const Genome genome = buildGenomeWithHidden(hiddenCount);
        const auto tuples = genome.legalConnectionTuples();

        REQUIRE(static_cast<int>(tuples.size()) == maxLegalConnectionsFor(hiddenCount));
        REQUIRE(genome.maxLegalConnectionCount() == maxLegalConnectionsFor(hiddenCount));

        std::set<std::pair<int, int>> seen;
        for (const auto& tuple : tuples)
        {
            REQUIRE(tuple.inFieldGeneId != tuple.outFieldGeneId);

            const auto srcType = genome.getFieldGeneById(tuple.inFieldGeneId).getParameters().type;
            const auto tgtType = genome.getFieldGeneById(tuple.outFieldGeneId).getParameters().type;
            REQUIRE((srcType == FieldGeneType::HIDDEN || srcType == FieldGeneType::INPUT));
            REQUIRE((tgtType == FieldGeneType::HIDDEN || tgtType == FieldGeneType::OUTPUT));

            REQUIRE(seen.insert({ tuple.inFieldGeneId, tuple.outFieldGeneId }).second);
        }
    }
}

TEST_CASE("Seeded random connections reach both the minimum and the full legal maximum", "[genome][seed_connections]")
{
    AblationFlagGuard guard;
    AblationConstants::seedRandomHiddenFields = true;
    AblationConstants::seedHiddenFieldsMin = 3;
    AblationConstants::seedHiddenFieldsMax = 3;
    AblationConstants::seedRandomConnections = true;

    const int legalMax = maxLegalConnectionsFor(3);
    std::set<int> observedCounts;

    for (int trial = 0; trial < 200; ++trial)
    {
        resetGlobalState();
        AND solution{ makeTopology(2, 1) };
        solution.initialize();

        REQUIRE(solution.getNumFieldGenes() == 6); // 2 inputs + 1 output + 3 hidden
        const int count = static_cast<int>(solution.getNumConnectionGenes());
        REQUIRE(count >= 1);
        REQUIRE(count <= legalMax);
        observedCounts.insert(count);

        std::set<std::pair<int, int>> seen;
        for (const auto& cg : solution.getGenome().getConnectionGenes())
            REQUIRE(seen.insert({ cg.getInFieldGeneId(), cg.getOutFieldGeneId() }).second);
    }

    REQUIRE(observedCounts.contains(1));
    REQUIRE(observedCounts.contains(legalMax));
}

TEST_CASE("Baseline parity: default flags leave initialize() and mutate() unchanged", "[ablations][baseline]")
{
    AblationFlagGuard guard;
    resetGlobalState();

    AND solution{ makeTopology(2, 1) };
    solution.initialize();

    REQUIRE(solution.getNumFieldGenes() == 3);
    REQUIRE(solution.getNumConnectionGenes() == 0);

    bool fieldGenesGrew = false;
    bool connectionGenesGrew = false;
    for (int i = 0; i < 20000 && !(fieldGenesGrew && connectionGenesGrew); ++i)
    {
        solution.mutate();
        if (solution.getNumFieldGenes() > 3)
            fieldGenesGrew = true;
        if (solution.getNumConnectionGenes() > 0)
            connectionGenesGrew = true;
    }

    REQUIRE(fieldGenesGrew);
    REQUIRE(connectionGenesGrew);
}

TEST_CASE("No Growth IO Only: initialize() seeds exactly the 2 legal I/O connections", "[ablations][no_growth_io_only]")
{
    AblationFlagGuard guard;
    resetGlobalState();
    AblationPresets::noGrowthIOOnly();

    AND solution{ makeTopology(2, 1) };
    solution.initialize();

    REQUIRE(solution.getNumFieldGenes() == 3);
    REQUIRE(solution.getNumConnectionGenes() == 2);

    const std::set<std::pair<int, int>> expected{ {1, 3}, {2, 3} };
    REQUIRE(connectionTupleSet(solution.getGenome()) == expected);
    REQUIRE(allConnectionsEnabled(solution.getGenome()));
}

TEST_CASE("No Growth IO Only: rule agreement - exhaustive seeder and mutation legality rule agree", "[ablations][no_growth_io_only]")
{
    AblationFlagGuard guard;
    resetGlobalState();
    AblationPresets::noGrowthIOOnly();
    AblationConstants::disableAddConnectionGene = false; // leave the rule live

    AND solution{ makeTopology(2, 1) };
    solution.initialize();
    REQUIRE(solution.getNumConnectionGenes() == 2);

    for (int i = 0; i < 1000; ++i)
        solution.mutate();

    REQUIRE(solution.getNumConnectionGenes() == 2);
}

TEST_CASE("No Growth IO Only: full flags freeze structure across 1000 mutations", "[ablations][no_growth_io_only]")
{
    AblationFlagGuard guard;
    resetGlobalState();
    AblationPresets::noGrowthIOOnly();

    AND solution{ makeTopology(2, 1) };
    solution.initialize();

    // solution.mutate() appends to an internal per-gene mutation-log string on every
    // call, unbounded here since nothing clears it outside a real generational cycle.
    // getGenome() deep-copies that string, so sample state every 100th mutation
    // instead of every mutation - checking every iteration turns this loop O(n^2).
    bool sawDisabledConnection = false;
    for (int i = 0; i < 3000; ++i)
    {
        solution.mutate();
        if (!sawDisabledConnection && i % 100 == 0)
            sawDisabledConnection = !allConnectionsEnabled(solution.getGenome());
    }

    REQUIRE(solution.getNumFieldGenes() == 3);
    REQUIRE(solution.getNumConnectionGenes() == 2);
    REQUIRE(sawDisabledConnection); // gene count is frozen, but toggle can still prune a connection
}

TEST_CASE("No Growth IO Only: parameters still evolve while structure is frozen", "[ablations][no_growth_io_only]")
{
    AblationFlagGuard guard;
    resetGlobalState();
    AblationPresets::noGrowthIOOnly();

    AND solution{ makeTopology(2, 1) };
    solution.initialize();

    const auto initialAmplitudes = [&]
    {
        std::vector<double> amps;
        for (const auto& cg : solution.getGenome().getConnectionGenes())
            amps.push_back(cg.getKernelAmplitude());
        return amps;
    }();

    bool parametersMoved = false;
    for (int i = 0; i < 1000 && !parametersMoved; ++i)
    {
        solution.mutate();
        const auto genes = solution.getGenome().getConnectionGenes();
        if (genes.size() != initialAmplitudes.size())
            continue;
        for (size_t j = 0; j < genes.size(); ++j)
        {
            if (genes[j].getKernelAmplitude() != initialAmplitudes[j])
            {
                parametersMoved = true;
                break;
            }
        }
    }

    REQUIRE(parametersMoved);
}

TEST_CASE("No Growth Reference Hidden Field Count: initialize() seeds the reference hidden field count and exactly the 5 legal connections", "[ablations][no_growth_reference_hidden_field_count]")
{
    AblationFlagGuard guard;
    resetGlobalState();
    AblationPresets::noGrowthReferenceHiddenFieldCount();

    AND solution{ makeTopology(2, 1) };
    solution.initialize();

    REQUIRE(solution.getNumFieldGenes() == 4);
    REQUIRE(solution.getNumConnectionGenes() == 5);

    const std::set<std::pair<int, int>> expected{ {1, 3}, {1, 4}, {2, 3}, {2, 4}, {4, 3} };
    REQUIRE(connectionTupleSet(solution.getGenome()) == expected);
    REQUIRE(allConnectionsEnabled(solution.getGenome()));
}

TEST_CASE("No Growth Reference Hidden Field Count: rule agreement and structure freeze across 1000 mutations", "[ablations][no_growth_reference_hidden_field_count]")
{
    AblationFlagGuard guard;
    resetGlobalState();
    AblationPresets::noGrowthReferenceHiddenFieldCount();
    AblationConstants::disableAddConnectionGene = false; // leave the rule live

    AND solution{ makeTopology(2, 1) };
    solution.initialize();
    REQUIRE(solution.getNumConnectionGenes() == 5);

    bool sawDisabledConnection = false;
    for (int i = 0; i < 3000; ++i)
    {
        solution.mutate();
        if (!sawDisabledConnection && i % 100 == 0)
            sawDisabledConnection = !allConnectionsEnabled(solution.getGenome());
    }

    REQUIRE(solution.getNumFieldGenes() == 4);
    REQUIRE(solution.getNumConnectionGenes() == 5);
    REQUIRE(sawDisabledConnection); // gene count is frozen, but toggle can still prune a connection
}

TEST_CASE("No Growth Reference Hidden Field Count: field ordering - inputs, then output, then hidden", "[ablations][no_growth_reference_hidden_field_count]")
{
    AblationFlagGuard guard;
    resetGlobalState();
    AblationPresets::noGrowthReferenceHiddenFieldCount();

    AND solution{ makeTopology(2, 1) };
    solution.initialize();

    const auto& genome = solution.getGenome();
    REQUIRE(genome.getFieldGeneById(1).getParameters().type == FieldGeneType::INPUT);
    REQUIRE(genome.getFieldGeneById(2).getParameters().type == FieldGeneType::INPUT);
    REQUIRE(genome.getFieldGeneById(3).getParameters().type == FieldGeneType::OUTPUT);
    REQUIRE(genome.getFieldGeneById(4).getParameters().type == FieldGeneType::HIDDEN);
}

TEST_CASE("No Crossover: crossover disabled - every offspring is an exact single-parent clone", "[ablations][no_crossover]")
{
    AblationFlagGuard guard;
    resetGlobalState();
    AblationPresets::noCrossover();

    const auto parentA = std::make_shared<AND>(makeTopology(2, 1));
    parentA->initialize();
    const auto parentB = std::make_shared<AND>(makeTopology(2, 1));
    parentB->initialize();
    for (int i = 0; i < 50; ++i)
        parentB->mutate();

    REQUIRE_FALSE(parentA->hasTheSameGenome(parentB));

    Species species;
    species.addSolution(parentA);
    species.addSolution(parentB);
    species.setOffspringCount(20);
    species.crossover();
    species.replaceMembersWithOffspring();

    const auto offspring = species.getMembers();
    REQUIRE(offspring.size() == 20);
    for (const auto& child : offspring)
    {
        const auto parents = child->getParents();
        REQUIRE(std::get<0>(parents) == std::get<1>(parents));
        const auto& parent = (std::get<0>(parents) == parentA->getId()) ? parentA : parentB;
        REQUIRE(parent->getId() == std::get<0>(parents));
        REQUIRE(child->hasTheSameGenome(parent));
    }
}

TEST_CASE("No Crossover: control - crossover enabled produces genuine mixing", "[ablations][no_crossover]")
{
    AblationFlagGuard guard;
    resetGlobalState();

    const auto parentA = std::make_shared<AND>(makeTopology(2, 1));
    parentA->initialize();

    // Deterministic divergence: seed every legal connection instead of relying on
    // rare addConnectionGeneProbability mutations. FieldGene::operator==/
    // ConnectionGene::operator== compare only (type, id) and innovationNumber, not
    // mutated kernel/neural-field values, so "genuine mixing" here can only be
    // observed through the connection-gene set. A 50x-mutate() divergence
    // occasionally leaves parentB with only 0 or 1 of AND's 2 legal connections
    // instead of both, making a genuine partial mix structurally impossible for
    // the whole trial batch below -- see the identical seeding idiom in the
    // "No Speciation" test above for the same reasoning.
    AblationConstants::seedAllLegalConnections = true;
    const auto parentB = std::make_shared<AND>(makeTopology(2, 1));
    parentB->initialize();
    AblationConstants::seedAllLegalConnections = false;

    REQUIRE_FALSE(parentA->hasTheSameGenome(parentB));

    Species species;
    species.addSolution(parentA);
    species.addSolution(parentB);
    species.setOffspringCount(200);
    species.crossover();
    species.replaceMembersWithOffspring();

    const auto offspring = species.getMembers();
    bool sawMixing = false;
    for (const auto& child : offspring)
    {
        if (!child->hasTheSameGenome(parentA) && !child->hasTheSameGenome(parentB))
        {
            sawMixing = true;
            break;
        }
    }

    REQUIRE(sawMixing);
}

TEST_CASE("No Speciation: isCompatible() ignores genetic distance when speciation is disabled", "[ablations][no_speciation]")
{
    AblationFlagGuard guard;
    resetGlobalState();

    const auto representative = std::make_shared<AND>(makeTopology(2, 1));
    representative->initialize();

    // Build a structurally divergent genome deterministically (No-Growth-One-Hidden-
    // style seeding) rather than relying on rare growth mutations (addFieldGeneProbability
    // is 0.0005/call, and genome growth makes repeated mutate() calls expensive).
    AblationConstants::seedAllLegalConnections = true;
    AblationConstants::seedRandomHiddenFields = true;
    AblationConstants::seedHiddenFieldsMin = 1;
    AblationConstants::seedHiddenFieldsMax = 1;
    const auto divergent = std::make_shared<AND>(makeTopology(2, 1));
    divergent->initialize();
    AblationConstants::seedAllLegalConnections = false;
    AblationConstants::seedRandomHiddenFields = false;
    AblationConstants::seedHiddenFieldsMin = 0;
    AblationConstants::seedHiddenFieldsMax = 0;

    REQUIRE_FALSE(representative->hasTheSameGenome(divergent));

    Species species;
    species.setRepresentative(representative);

    REQUIRE_FALSE(species.isCompatible(divergent));

    AblationConstants::disableSpeciation = true;
    REQUIRE(species.isCompatible(divergent));
}

TEST_CASE("No Speciation: initialize() starts from a random population, not a minimal one", "[ablations][no_speciation]")
{
    // Paper (Stanley & Miikkulainen 2002, S5.5): nonspeciated NEAT must start from a
    // random initial population, or no structural innovation survives and the
    // population is stuck in minimal form. Random start and single-species pooling
    // must hold simultaneously.
    AblationFlagGuard guard;
    resetGlobalState();
    AblationPresets::noSpeciation();

    AND solutionA{ makeTopology(2, 1) };
    solutionA.initialize();
    REQUIRE(solutionA.getNumFieldGenes() > 3);
    REQUIRE(solutionA.getNumConnectionGenes() > 0);

    resetGlobalState();
    AND solutionB{ makeTopology(2, 1) };
    solutionB.initialize();

    Species species;
    species.setRepresentative(std::make_shared<AND>(solutionA));
    REQUIRE(species.isCompatible(std::make_shared<AND>(solutionB)));
}

TEST_CASE("Random Initial Topology: initialize() seeds hidden field count within [1,5] and it varies across trials", "[ablations][random_initial_topology]")
{
    AblationFlagGuard guard;
    AblationPresets::randomInitialTopology();

    std::set<int> observedHiddenCounts;
    for (int trial = 0; trial < 200; ++trial)
    {
        resetGlobalState();
        AND solution{ makeTopology(2, 1) };
        solution.initialize();

        const int hiddenCount = static_cast<int>(solution.getNumFieldGenes()) - 3;
        REQUIRE(hiddenCount >= 1);
        REQUIRE(hiddenCount <= 5);
        observedHiddenCounts.insert(hiddenCount);
    }

    REQUIRE(observedHiddenCounts.size() > 1);
}

TEST_CASE("Random Initial Topology: seeded connections are legal, non-duplicate, and within count bounds", "[ablations][random_initial_topology]")
{
    AblationFlagGuard guard;
    AblationPresets::randomInitialTopology();

    for (int trial = 0; trial < 200; ++trial)
    {
        resetGlobalState();
        AND solution{ makeTopology(2, 1) };
        solution.initialize();

        const auto& genome = solution.getGenome();
        const int hiddenCount = static_cast<int>(solution.getNumFieldGenes()) - 3;
        const int legalMax = maxLegalConnectionsFor(hiddenCount);

        REQUIRE(solution.getNumConnectionGenes() >= 1);
        REQUIRE(static_cast<int>(solution.getNumConnectionGenes()) <= legalMax);

        std::set<std::pair<int, int>> seen;
        for (const auto& cg : genome.getConnectionGenes())
        {
            const int a = cg.getInFieldGeneId();
            const int b = cg.getOutFieldGeneId();
            REQUIRE(a != b);

            const auto srcType = genome.getFieldGeneById(a).getParameters().type;
            const auto tgtType = genome.getFieldGeneById(b).getParameters().type;
            REQUIRE((srcType == FieldGeneType::HIDDEN || srcType == FieldGeneType::INPUT));
            REQUIRE((tgtType == FieldGeneType::HIDDEN || tgtType == FieldGeneType::OUTPUT));

            REQUIRE(seen.insert({ a, b }).second);
        }
        REQUIRE(allConnectionsEnabled(genome));
    }
}

TEST_CASE("Random Initial Topology: field ordering - inputs, then output, then hidden, regardless of hidden count", "[ablations][random_initial_topology]")
{
    AblationFlagGuard guard;
    resetGlobalState();
    AblationPresets::randomInitialTopology();

    AND solution{ makeTopology(2, 1) };
    solution.initialize();

    const auto& genome = solution.getGenome();
    REQUIRE(genome.getFieldGeneById(1).getParameters().type == FieldGeneType::INPUT);
    REQUIRE(genome.getFieldGeneById(2).getParameters().type == FieldGeneType::INPUT);
    REQUIRE(genome.getFieldGeneById(3).getParameters().type == FieldGeneType::OUTPUT);
    for (int id = 4; id <= static_cast<int>(solution.getNumFieldGenes()); ++id)
        REQUIRE(genome.getFieldGeneById(id).getParameters().type == FieldGeneType::HIDDEN);
}

TEST_CASE("Random Initial Topology: structure is not frozen - mutate() can still grow the genome beyond the initial seed", "[ablations][random_initial_topology]")
{
    AblationFlagGuard guard;
    resetGlobalState();
    AblationPresets::randomInitialTopology();

    AND solution{ makeTopology(2, 1) };
    solution.initialize();
    const int initialFieldGenes = static_cast<int>(solution.getNumFieldGenes());

    bool grew = false;
    for (int i = 0; i < 20000 && !grew; ++i)
    {
        solution.mutate();
        if (static_cast<int>(solution.getNumFieldGenes()) > initialFieldGenes)
            grew = true;
    }

    REQUIRE(grew);
}
