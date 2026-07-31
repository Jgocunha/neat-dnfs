#include <catch2/catch_test_macros.hpp>

#include "neat/solution.h"
#include "solutions/detection_instability.h"
#include "test_helpers.h"
#include "test_stub_solution.h"

using namespace neat_dnfs;
using namespace neat_dnfs::test;
using namespace dnf_composer::element;

TEST_CASE("Solution Initialization", "[Solution]")
{
    SECTION("Valid Initialization")
    {
        const auto topology = makeTopology(1, 1);
        DetectionInstability solution(topology);

        REQUIRE(solution.getGenome().getFieldGenes().empty());
        REQUIRE(solution.getGenome().getConnectionGenes().empty());
        REQUIRE(solution.getParameters().fitness == 0.0);
        REQUIRE(solution.getParameters().adjustedFitness == 0.0);
        REQUIRE(solution.getParameters().age == 0);
    }

    SECTION("Invalid Initialization - Not enough input genes")
    {
        REQUIRE_THROWS_AS(
            DetectionInstability(SolutionTopology{ {
                {FieldGeneType::OUTPUT, dnf_composer::element::ElementDimensions{DimensionConstants::xSize, DimensionConstants::dx}},
            }}),
            std::invalid_argument
        );
    }

    SECTION("Invalid Initialization - Not enough output genes")
    {
        REQUIRE_THROWS_AS(
            DetectionInstability(SolutionTopology{ {
                {FieldGeneType::INPUT, dnf_composer::element::ElementDimensions{DimensionConstants::xSize, DimensionConstants::dx}},
            }}),
            std::invalid_argument
        );
    }
}

TEST_CASE("Solution Initialize Method", "[Solution]")
{
    const auto topology = makeTopology(1, 1);
    DetectionInstability solution(topology);
    solution.initialize();

    const auto fieldGenes = solution.getGenome().getFieldGenes();
    const long inputs  = std::count_if(fieldGenes.begin(), fieldGenes.end(), [](const auto& g){ return g.getParameters().type == FieldGeneType::INPUT; });
    const long outputs = std::count_if(fieldGenes.begin(), fieldGenes.end(), [](const auto& g){ return g.getParameters().type == FieldGeneType::OUTPUT; });

    REQUIRE(inputs  == 1);
    REQUIRE(outputs == 1);
}

TEST_CASE("Solution Mutate Method", "[Solution]")
{
    static constexpr uint16_t attempts = 1000;

    for (uint16_t i = 0; i < attempts; ++i)
    {
        auto topology = makeTopology(1, 1);
        DetectionInstability solution(topology);
        solution.initialize();
        const size_t initialGenomeSize = solution.getGenomeSize();

        REQUIRE_NOTHROW(solution.mutate());
        REQUIRE(solution.getGenomeSize() >= initialGenomeSize);
    }
}

TEST_CASE("Solution Getters", "[Solution]")
{
    auto topology = makeTopology(1, 1);
    DetectionInstability solution(topology);

    REQUIRE(solution.getGenome().getFieldGenes().empty());
    REQUIRE(solution.getGenome().getConnectionGenes().empty());
    REQUIRE(solution.getFitness() == 0.0);
    REQUIRE(solution.getGenomeSize() == 0);
    REQUIRE(solution.getInnovationNumbers().empty());
}

TEST_CASE("Solution Build Phenotype", "[Solution]")
{
    auto topology = makeTopology(1, 1);
    DetectionInstability solution(topology);
    solution.initialize();
    const auto genome = solution.getGenome();
    const auto fieldGenes = genome.getFieldGenes();

    const FieldGene& firstInput  = fieldGenes[0]; // input
    const FieldGene& firstOutput = fieldGenes[1]; // output

    REQUIRE(firstInput.getParameters().type == FieldGeneType::INPUT);
    REQUIRE(firstOutput.getParameters().type == FieldGeneType::OUTPUT);

    solution.addConnectionGene(ConnectionGene(
        ConnectionTuple(firstInput.getParameters().id, firstOutput.getParameters().id), 99));

    solution.buildPhenotype();

    auto phenotype = solution.getPhenotype();
    phenotype.init();
    REQUIRE(phenotype.getNumberOfElements() > 0);
}

TEST_CASE("Solution Age Increment", "[Solution]")
{
    const auto topology = makeTopology(1, 1);
    DetectionInstability solution(topology);

    const int initialAge = solution.getParameters().age;

    solution.incrementAge();
    REQUIRE(solution.getParameters().age == initialAge + 1);
}

TEST_CASE("Solution Fitness Management", "[Solution]")
{
    const auto topology = makeTopology(1, 1);
    DetectionInstability solution(topology);
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
    auto topology = makeTopology(1, 1);
    DetectionInstability solution(topology);
    solution.initialize();

    const FieldGene newGene({ FieldGeneType::HIDDEN, 999 });
    solution.addFieldGene(newGene);

    auto fieldGenes = solution.getGenome().getFieldGenes();
    REQUIRE(std::ranges::find(fieldGenes, newGene) != fieldGenes.end());
}

TEST_CASE("Solution Add Connection Gene", "[Solution]")
{
    auto topology = makeTopology(1, 1);
    DetectionInstability solution(topology);
    solution.initialize();

    const auto fieldGenes = solution.getGenome().getFieldGenes();
    const int id1 = fieldGenes[0].getParameters().id; // input
    const int id2 = fieldGenes[1].getParameters().id; // output

    const ConnectionTuple tuple(id1, id2);
    const ConnectionGene newGene(tuple, 0);
    solution.addConnectionGene(newGene);

    auto connectionGenes = solution.getGenome().getConnectionGenes();
    REQUIRE(std::ranges::find(connectionGenes, newGene) != connectionGenes.end());
}

TEST_CASE("Solution Contains Connection Gene", "[Solution]")
{
    auto topology = makeTopology(1, 1);
    DetectionInstability solution(topology);
    solution.initialize();

    const auto fieldGenes = solution.getGenome().getFieldGenes();
    const int id1 = fieldGenes[0].getParameters().id; // input
    const int id2 = fieldGenes[1].getParameters().id; // output

    const ConnectionTuple tuple(id1, id2);
    const ConnectionGene newGene(tuple, 0);
    solution.addConnectionGene(newGene);

    REQUIRE(solution.containsConnectionGene(newGene) == true);
}

TEST_CASE("Solution Evaluation", "[Solution]")
{
    const auto topology = makeTopology(1, 1);
    DetectionInstability solution(topology);
    solution.initialize();

    REQUIRE_NOTHROW(solution.evaluate());
}

// Regression test: fitness helpers (closenessToRestingLevel, noBumps, etc.)
// all resolve their field through the shared Solution::getNeuralFieldOrThrow,
// which must raise an indicative error rather than silently returning 0.0 --
// a missing/wrong-type field means the solution is misconfigured, not that it
// legitimately scored the worst possible fitness.
TEST_CASE("Solution fitness helpers throw on a field name that doesn't exist", "[Solution]")
{
    const auto topology = makeTopology(1, 1);
    MissingFieldSolution solution(topology);
    solution.initialize();

    REQUIRE_THROWS_AS(solution.evaluate(), std::invalid_argument);
}

// Same guard, but the name resolves to a real element (a GaussStimulus)
// instead of no element at all -- the dynamic_pointer_cast<NeuralField>
// failure path, not just the phenotype.getElement() miss path.
TEST_CASE("Solution fitness helpers throw when the named element isn't a NeuralField", "[Solution]")
{
    const auto topology = makeTopology(1, 1);
    WrongElementTypeSolution solution(topology);
    solution.initialize();

    REQUIRE_THROWS_AS(solution.evaluate(), std::invalid_argument);
}

TEST_CASE("Solution Crossover", "[Solution]")
{
    auto topology = makeTopology(1, 1);
    const auto parent1 = std::make_shared<DetectionInstability>(topology);
    const auto parent2 = std::make_shared<DetectionInstability>(topology);
    parent1->initialize();
    parent2->initialize();

    auto offspring = parent1->crossover(parent2);

    REQUIRE(offspring != nullptr);
    REQUIRE(!offspring->getGenome().getFieldGenes().empty());
}

TEST_CASE("Solution Phenotype Translation", "[Solution]")
{
    const auto topology = makeTopology(1, 1);
    DetectionInstability solution(topology);
    solution.initialize();

    SECTION("Build phenotype")
    {
        REQUIRE_NOTHROW(solution.buildPhenotype());
    }

    SECTION("Clear phenotype")
    {
        REQUIRE_NOTHROW(solution.clearPhenotype());
    }
}

TEST_CASE("Solution::hasTheSameTopology", "[Solution]")
{
    resetGlobalState();
    const auto topology = makeTopology(1, 1);
    const auto solutionA = std::make_shared<DetectionInstability>(topology);
    const auto solutionB = std::make_shared<DetectionInstability>(topology);

    REQUIRE(solutionA->hasTheSameTopology(solutionB));

    const auto differentTopology = makeTopology(1, 1, 1); // extra hidden gene
    const auto solutionC = std::make_shared<DetectionInstability>(differentTopology);
    REQUIRE_FALSE(solutionA->hasTheSameTopology(solutionC));
}

TEST_CASE("Solution::hasTheSameParameters", "[Solution]")
{
    resetGlobalState();
    const auto topology = makeTopology(1, 1);
    const auto solutionA = std::make_shared<DetectionInstability>(topology);
    const auto solutionB = std::make_shared<DetectionInstability>(topology);

    REQUIRE(solutionA->hasTheSameParameters(solutionB));

    solutionB->setAdjustedFitness(0.5);
    REQUIRE_FALSE(solutionA->hasTheSameParameters(solutionB));
}

TEST_CASE("Solution::hasTheSameGenome", "[Solution]")
{
    resetGlobalState();
    const auto topology = makeTopology(1, 1);
    const auto solutionA = std::make_shared<DetectionInstability>(topology);
    solutionA->initialize();

    // hasTheSameGenome compares Genome::operator== directly, so use a genome
    // that is actually identical -- a fresh solution built from the same
    // topology and initialize() sequence, not copy() (which reconstructs the
    // genome from the phenotype via translatePhenotypeToGenome() and is not
    // guaranteed to produce a byte-identical genome; see test_solutions_tasks.cpp).
    const auto solutionB = std::make_shared<DetectionInstability>(topology);
    solutionB->initialize();
    REQUIRE(solutionA->hasTheSameGenome(solutionB));

    solutionB->addFieldGene(FieldGene({ FieldGeneType::HIDDEN, 999 }));
    REQUIRE_FALSE(solutionA->hasTheSameGenome(solutionB));
}

TEST_CASE("Solution::clearGenome", "[Solution]")
{
    resetGlobalState();
    const auto topology = makeTopology(1, 1);
    DetectionInstability solution(topology);
    solution.initialize();
    // getGenomeSize() counts connection genes only; a freshly initialized
    // solution has field genes but no connections yet, so check field genes.
    REQUIRE(!solution.getGenome().getFieldGenes().empty());

    solution.clearGenome();
    REQUIRE(solution.getGenomeSize() == 0);
    REQUIRE(solution.getGenome().getFieldGenes().empty());
    REQUIRE(solution.getGenome().getConnectionGenes().empty());
}

TEST_CASE("Solution::clearPhenotype allows rebuilding the phenotype", "[Solution]")
{
    resetGlobalState();
    const auto topology = makeTopology(1, 1);
    DetectionInstability solution(topology);
    solution.initialize();

    solution.buildPhenotype();
    solution.clearPhenotype();

    REQUIRE_NOTHROW(solution.buildPhenotype());
    REQUIRE(solution.getPhenotype().getNumberOfElements() > 0);
}

TEST_CASE("Solution::translatePhenotypeToGenome produces a non-empty genome from a built phenotype", "[Solution]")
{
    // translatePhenotypeToGenome() infers field gene type (INPUT/OUTPUT/HIDDEN)
    // from each neural field's connection pattern in the phenotype (see
    // solution.cpp translatePhenotypeToGenome). With zero connection genes,
    // as here, that inference does not reproduce the original field count
    // exactly -- so this only asserts the documented, guaranteed behaviour:
    // it does not throw and the resulting genome is non-empty.
    resetGlobalState();
    const auto topology = makeTopology(1, 1);
    DetectionInstability solution(topology);
    solution.initialize();
    solution.buildPhenotype();

    REQUIRE_NOTHROW(solution.translatePhenotypeToGenome());
    REQUIRE(!solution.getGenome().getFieldGenes().empty());
}

TEST_CASE("Solution::crossover with identical parents yields a matching-topology child", "[Solution]")
{
    resetGlobalState();
    const auto topology = makeTopology(1, 1);
    const auto parent1 = std::make_shared<DetectionInstability>(topology);
    const auto parent2 = std::make_shared<DetectionInstability>(topology);
    parent1->initialize();
    parent2->initialize();

    const auto offspring = parent1->crossover(parent2);

    REQUIRE(offspring != nullptr);
    REQUIRE(offspring->hasTheSameTopology(parent1));
    REQUIRE(offspring->getGenome().getFieldGenes().size() == parent1->getGenome().getFieldGenes().size());
}

TEST_CASE("Solution::crossover inherits field genes from the fitter parent", "[Solution]")
{
    resetGlobalState();
    const auto topology = makeTopology(1, 1);
    // FixedFitnessSolution::evaluate() reports a fitness set directly by the
    // test rather than one derived from a real (task- and seed-dependent)
    // DNF simulation -- crossover() compares getFitness(), not adjustedFitness,
    // and Solution has no direct fitness setter otherwise.
    const auto parent1 = std::make_shared<FixedFitnessSolution>(topology, 0.9);
    const auto parent2 = std::make_shared<FixedFitnessSolution>(topology, 0.1);
    parent1->initialize();
    parent2->initialize();
    parent1->addFieldGene(FieldGene({ FieldGeneType::HIDDEN, 999 }));

    parent1->evaluate();
    parent2->evaluate();
    REQUIRE(parent1->getFitness() != parent2->getFitness());

    const auto fitterParent = parent1->getFitness() > parent2->getFitness() ? parent1 : parent2;
    const auto offspring = parent1->crossover(parent2);

    REQUIRE(offspring != nullptr);
    REQUIRE(!offspring->getGenome().getFieldGenes().empty());
    // Every offspring field gene must trace back to the fitter parent.
    for (const auto& gene : offspring->getGenome().getFieldGenes())
        REQUIRE(fitterParent->getGenome().containsFieldGene(gene));
}
