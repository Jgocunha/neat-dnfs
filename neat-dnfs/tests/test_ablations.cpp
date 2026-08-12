#include <catch2/catch_test_macros.hpp>

#include <set>
#include <utility>

#include "constants.h"
#include "neat/genome.h"
#include "neat/species.h"
#include "solutions/hri_packaging_task.h"

namespace
{
    using namespace neat_dnfs;

    struct AblationFlagGuard
    {
        ~AblationFlagGuard()
        {
            AblationConstants::label = "";
            AblationConstants::disableAddFieldGene = false;
            AblationConstants::disableAddConnectionGene = false;
            AblationConstants::disableToggleConnectionGene = false;
            AblationConstants::seedAllLegalConnections = false;
            AblationConstants::seedRandomHiddenFields = false;
            AblationConstants::seedHiddenFieldsMin = 0;
            AblationConstants::seedHiddenFieldsMax = 0;
            AblationConstants::seedRandomConnections = false;
            AblationConstants::seedConnectionsMin = 1;
            AblationConstants::seedConnectionsMax = 8;
            AblationConstants::disableSpeciation = false;
            AblationConstants::disableCrossover = false;
        }
    };

    void resetGlobals()
    {
        Genome::resetGlobalInnovationNumber();
        Genome::clearGenerationalInnovations();
        Solution::resetUniqueIdentifier();
        Species::resetUniqueIdentifier();
    }

    SolutionTopology hriTopology()
    {
        using namespace dnf_composer::element;
        const ElementDimensions dims{ DimensionConstants::xSize, DimensionConstants::dx };
        return SolutionTopology{ {
            {FieldGeneType::INPUT, dims},
            {FieldGeneType::INPUT, dims},
            {FieldGeneType::INPUT, dims},
            {FieldGeneType::OUTPUT, dims},
        } };
    }

    void setNoGrowthIOOnlyFlags()
    {
        AblationConstants::disableAddFieldGene = true;
        AblationConstants::disableAddConnectionGene = true;
        AblationConstants::disableToggleConnectionGene = true;
        AblationConstants::seedAllLegalConnections = true;
    }

    void setNoGrowthOneHiddenFlags()
    {
        setNoGrowthIOOnlyFlags();
        AblationConstants::seedRandomHiddenFields = true;
        AblationConstants::seedHiddenFieldsMin = 1;
        AblationConstants::seedHiddenFieldsMax = 1;
    }

    void setRandomInitialTopologyFlags()
    {
        AblationConstants::seedRandomHiddenFields = true;
        AblationConstants::seedHiddenFieldsMin = 0;
        AblationConstants::seedHiddenFieldsMax = 5;
        AblationConstants::seedRandomConnections = true;
        AblationConstants::seedConnectionsMin = 1;
        AblationConstants::seedConnectionsMax = 8;
    }

    // sources = 3 inputs + hiddenCount; targets = hiddenCount + 1 output;
    // hiddenCount self-loops (hidden id as both source and target) are illegal.
    int maxLegalConnectionsFor(int hiddenCount)
    {
        return (3 + hiddenCount) * (hiddenCount + 1) - hiddenCount;
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

TEST_CASE("Baseline parity: default flags leave initialize() and mutate() unchanged", "[ablations][baseline]")
{
    AblationFlagGuard guard;
    resetGlobals();

    HRIPackagingTask solution{ hriTopology() };
    solution.initialize();

    REQUIRE(solution.getNumFieldGenes() == 4);
    REQUIRE(solution.getNumConnectionGenes() == 0);

    bool fieldGenesGrew = false;
    bool connectionGenesGrew = false;
    for (int i = 0; i < 20000 && !(fieldGenesGrew && connectionGenesGrew); ++i)
    {
        solution.mutate();
        if (solution.getNumFieldGenes() > 4)
            fieldGenesGrew = true;
        if (solution.getNumConnectionGenes() > 0)
            connectionGenesGrew = true;
    }

    REQUIRE(fieldGenesGrew);
    REQUIRE(connectionGenesGrew);
}

TEST_CASE("No Growth IO Only: initialize() seeds exactly the 3 legal I/O connections", "[ablations][no_growth_io_only]")
{
    AblationFlagGuard guard;
    resetGlobals();
    setNoGrowthIOOnlyFlags();

    HRIPackagingTask solution{ hriTopology() };
    solution.initialize();

    REQUIRE(solution.getNumFieldGenes() == 4);
    REQUIRE(solution.getNumConnectionGenes() == 3);

    const std::set<std::pair<int, int>> expected{ {1, 4}, {2, 4}, {3, 4} };
    REQUIRE(connectionTupleSet(solution.getGenome()) == expected);
    REQUIRE(allConnectionsEnabled(solution.getGenome()));
}

TEST_CASE("No Growth IO Only: rule agreement - exhaustive seeder and mutation legality rule agree", "[ablations][no_growth_io_only]")
{
    AblationFlagGuard guard;
    resetGlobals();
    setNoGrowthIOOnlyFlags();
    AblationConstants::disableAddConnectionGene = false; // leave the rule live

    HRIPackagingTask solution{ hriTopology() };
    solution.initialize();
    REQUIRE(solution.getNumConnectionGenes() == 3);

    for (int i = 0; i < 1000; ++i)
        solution.mutate();

    REQUIRE(solution.getNumConnectionGenes() == 3);
}

TEST_CASE("No Growth IO Only: full flags freeze structure across 1000 mutations", "[ablations][no_growth_io_only]")
{
    AblationFlagGuard guard;
    resetGlobals();
    setNoGrowthIOOnlyFlags();

    HRIPackagingTask solution{ hriTopology() };
    solution.initialize();

    for (int i = 0; i < 1000; ++i)
        solution.mutate();

    REQUIRE(solution.getNumFieldGenes() == 4);
    REQUIRE(solution.getNumConnectionGenes() == 3);
    REQUIRE(allConnectionsEnabled(solution.getGenome()));
}

TEST_CASE("No Growth IO Only: parameters still evolve while structure is frozen", "[ablations][no_growth_io_only]")
{
    AblationFlagGuard guard;
    resetGlobals();
    setNoGrowthIOOnlyFlags();

    HRIPackagingTask solution{ hriTopology() };
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

TEST_CASE("No Growth One Hidden: initialize() seeds one hidden field and exactly the 7 legal connections", "[ablations][no_growth_one_hidden]")
{
    AblationFlagGuard guard;
    resetGlobals();
    setNoGrowthOneHiddenFlags();

    HRIPackagingTask solution{ hriTopology() };
    solution.initialize();

    REQUIRE(solution.getNumFieldGenes() == 5);
    REQUIRE(solution.getNumConnectionGenes() == 7);

    const std::set<std::pair<int, int>> expected{ {1, 4}, {1, 5}, {2, 4}, {2, 5}, {3, 4}, {3, 5}, {5, 4} };
    REQUIRE(connectionTupleSet(solution.getGenome()) == expected);
    REQUIRE(allConnectionsEnabled(solution.getGenome()));
}

TEST_CASE("No Growth One Hidden: rule agreement and structure freeze across 1000 mutations", "[ablations][no_growth_one_hidden]")
{
    AblationFlagGuard guard;
    resetGlobals();
    setNoGrowthOneHiddenFlags();
    AblationConstants::disableAddConnectionGene = false; // leave the rule live

    HRIPackagingTask solution{ hriTopology() };
    solution.initialize();
    REQUIRE(solution.getNumConnectionGenes() == 7);

    for (int i = 0; i < 1000; ++i)
        solution.mutate();

    REQUIRE(solution.getNumFieldGenes() == 5);
    REQUIRE(solution.getNumConnectionGenes() == 7);
    REQUIRE(allConnectionsEnabled(solution.getGenome()));
}

TEST_CASE("No Growth One Hidden: field ordering - inputs, then output, then hidden", "[ablations][no_growth_one_hidden]")
{
    AblationFlagGuard guard;
    resetGlobals();
    setNoGrowthOneHiddenFlags();

    HRIPackagingTask solution{ hriTopology() };
    solution.initialize();

    const auto genome = solution.getGenome();
    REQUIRE(genome.getFieldGeneById(1).getParameters().type == FieldGeneType::INPUT);
    REQUIRE(genome.getFieldGeneById(2).getParameters().type == FieldGeneType::INPUT);
    REQUIRE(genome.getFieldGeneById(3).getParameters().type == FieldGeneType::INPUT);
    REQUIRE(genome.getFieldGeneById(4).getParameters().type == FieldGeneType::OUTPUT);
    REQUIRE(genome.getFieldGeneById(5).getParameters().type == FieldGeneType::HIDDEN);
}

TEST_CASE("No Crossover: crossover disabled - every offspring is an exact single-parent clone", "[ablations][no_crossover]")
{
    AblationFlagGuard guard;
    resetGlobals();
    AblationConstants::disableCrossover = true;

    const auto parentA = std::make_shared<HRIPackagingTask>(hriTopology());
    parentA->initialize();
    const auto parentB = std::make_shared<HRIPackagingTask>(hriTopology());
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
    resetGlobals();

    const auto parentA = std::make_shared<HRIPackagingTask>(hriTopology());
    parentA->initialize();
    const auto parentB = std::make_shared<HRIPackagingTask>(hriTopology());
    parentB->initialize();
    for (int i = 0; i < 50; ++i)
        parentB->mutate();

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
    resetGlobals();

    const auto representative = std::make_shared<HRIPackagingTask>(hriTopology());
    representative->initialize();

    // Build a structurally divergent genome deterministically (No-Growth-One-Hidden-
    // style seeding) rather than relying on rare growth mutations (addFieldGeneProbability
    // is 0.0005/call, and genome growth makes repeated mutate() calls expensive).
    AblationConstants::seedAllLegalConnections = true;
    AblationConstants::seedRandomHiddenFields = true;
    AblationConstants::seedHiddenFieldsMin = 1;
    AblationConstants::seedHiddenFieldsMax = 1;
    const auto divergent = std::make_shared<HRIPackagingTask>(hriTopology());
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

TEST_CASE("Random Initial Topology: initialize() seeds hidden field count within [0,5] and it varies across trials", "[ablations][random_initial_topology]")
{
    AblationFlagGuard guard;
    setRandomInitialTopologyFlags();

    std::set<int> observedHiddenCounts;
    for (int trial = 0; trial < 200; ++trial)
    {
        resetGlobals();
        HRIPackagingTask solution{ hriTopology() };
        solution.initialize();

        const int hiddenCount = solution.getNumFieldGenes() - 4;
        REQUIRE(hiddenCount >= 0);
        REQUIRE(hiddenCount <= 5);
        observedHiddenCounts.insert(hiddenCount);
    }

    REQUIRE(observedHiddenCounts.size() > 1);
}

TEST_CASE("Random Initial Topology: seeded connections are legal, non-duplicate, and within count bounds", "[ablations][random_initial_topology]")
{
    AblationFlagGuard guard;
    setRandomInitialTopologyFlags();

    bool sawNonZeroConnections = false;
    for (int trial = 0; trial < 200; ++trial)
    {
        resetGlobals();
        HRIPackagingTask solution{ hriTopology() };
        solution.initialize();

        const auto genome = solution.getGenome();
        const int hiddenCount = solution.getNumFieldGenes() - 4;
        const int legalMax = maxLegalConnectionsFor(hiddenCount);

        REQUIRE(solution.getNumConnectionGenes() >= 0);
        REQUIRE(solution.getNumConnectionGenes() <= AblationConstants::seedConnectionsMax);
        REQUIRE(solution.getNumConnectionGenes() <= legalMax);
        if (solution.getNumConnectionGenes() > 0)
            sawNonZeroConnections = true;

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

    REQUIRE(sawNonZeroConnections);
}

TEST_CASE("Random Initial Topology: field ordering - inputs, then output, then hidden, regardless of hidden count", "[ablations][random_initial_topology]")
{
    AblationFlagGuard guard;
    resetGlobals();
    setRandomInitialTopologyFlags();

    HRIPackagingTask solution{ hriTopology() };
    solution.initialize();

    const auto genome = solution.getGenome();
    REQUIRE(genome.getFieldGeneById(1).getParameters().type == FieldGeneType::INPUT);
    REQUIRE(genome.getFieldGeneById(2).getParameters().type == FieldGeneType::INPUT);
    REQUIRE(genome.getFieldGeneById(3).getParameters().type == FieldGeneType::INPUT);
    REQUIRE(genome.getFieldGeneById(4).getParameters().type == FieldGeneType::OUTPUT);
    for (int id = 5; id <= solution.getNumFieldGenes(); ++id)
        REQUIRE(genome.getFieldGeneById(id).getParameters().type == FieldGeneType::HIDDEN);
}

TEST_CASE("Random Initial Topology: structure is not frozen - mutate() can still grow the genome beyond the initial seed", "[ablations][random_initial_topology]")
{
    AblationFlagGuard guard;
    resetGlobals();
    setRandomInitialTopologyFlags();

    HRIPackagingTask solution{ hriTopology() };
    solution.initialize();
    const int initialFieldGenes = solution.getNumFieldGenes();

    bool grew = false;
    for (int i = 0; i < 20000 && !grew; ++i)
    {
        solution.mutate();
        if (solution.getNumFieldGenes() > initialFieldGenes)
            grew = true;
    }

    REQUIRE(grew);
}
