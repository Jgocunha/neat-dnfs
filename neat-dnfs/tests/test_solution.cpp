#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>

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
    REQUIRE(solution.getPhenotype().getNumberOfElements() == 0);
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
    REQUIRE(solution.getPhenotype().getNumberOfElements() == 0);
}

// Same guard, checked directly against twoBumpsAtPositionWithAmplitudeAndWidth
// -- the multi-bump fitness primitive named explicitly in issue #56's location
// list -- rather than only closenessToRestingLevel.
TEST_CASE("Solution twoBumpsAtPositionWithAmplitudeAndWidth throws on a field name that doesn't exist", "[Solution]")
{
    const auto topology = makeTopology(1, 1);
    MissingFieldTwoBumpsSolution solution(topology);
    solution.initialize();

    REQUIRE_THROWS_AS(solution.evaluate(), std::invalid_argument);
    REQUIRE(solution.getPhenotype().getNumberOfElements() == 0);
}

// Issue #56: preShapednessAtPosition converts a spatial position to an index
// with `static_cast<int>(position / d_x)` and indexes the "activation"
// component directly. A position at the field's own upper bound (xSize) --
// which occurs in practice, e.g. a stimulus placed at the last valid
// position -- previously produced an index one past the last valid sample.
// This must not read out of bounds and must produce a well-defined, bounded
// result instead.
TEST_CASE("Solution preShapednessAtPosition does not read past the end of the field at the upper boundary", "[Solution]")
{
    const auto topology = makeTopology(1, 1);
    BoundaryPositionPreShapednessSolution solution(topology);
    solution.initialize();

    REQUIRE_NOTHROW(solution.evaluate());
    REQUIRE(std::isfinite(solution.getFitness()));
    REQUIRE(solution.getFitness() >= 0.0);
    REQUIRE(solution.getFitness() <= 1.0);
}

// Issue #53: unlike oneBumpAtPositionWithAmplitudeAndWidth (which returns 0.0
// immediately when the field has zero bumps), twoBumpsAtPositionWithAmplitudeAndWidth
// had no such guard -- on an empty field the bump-matching loop never runs and
// fitness was computed against a default-constructed NeuralFieldBump (position/
// amplitude/width all 0.0), crediting a bump that was never there. This is a
// real red/green check: before the fix this asserts a non-zero inflated value;
// after the fix it must equal the same zero baseline oneBump uses.
TEST_CASE("Solution twoBumpsAtPositionWithAmplitudeAndWidth yields no bump credit for an empty field", "[Solution]")
{
    const auto topology = makeTopology(1, 1);
    EmptyFieldTwoBumpsSolution solution(topology);
    solution.initialize();

    REQUIRE_NOTHROW(solution.evaluate());
    REQUIRE(solution.getFitness() == 0.0);
}

// Same as above, for threeBumpsAtPositionWithAmplitudeAndWidth.
TEST_CASE("Solution threeBumpsAtPositionWithAmplitudeAndWidth yields no bump credit for an empty field", "[Solution]")
{
    const auto topology = makeTopology(1, 1);
    EmptyFieldThreeBumpsSolution solution(topology);
    solution.initialize();

    REQUIRE_NOTHROW(solution.evaluate());
    REQUIRE(solution.getFitness() == 0.0);
}

// Issue #53 (sibling bug flagged in the author's own comment): twoBumpsAtPositionWithAmplitudeAndWidth
// independently searched for "the bump closest to position1" and "the bump
// closest to position2" without removing a matched bump from the candidate
// pool, so a single real bump could be matched -- and credited -- against
// both target slots at once, inflating fitness as though two distinct bumps
// existed. SingleBumpTwoBumpsSolution drives one field with a single Gaussian
// stimulus (so exactly one real bump forms) and queries the helper with both
// target positions pointing at that same bump, proving the fix matches
// injectively (a matched bump is removed from the pool before the next slot
// is matched) instead of double-counting.
TEST_CASE("Solution twoBumpsAtPositionWithAmplitudeAndWidth does not credit the same bump for both target positions", "[Solution]")
{
    const auto topology = makeTopology(1, 1);
    SingleBumpTwoBumpsSolution solution(topology);
    solution.initialize();

    REQUIRE_NOTHROW(solution.evaluate());
    // A single, sufficiently strong, well-localized stimulus reliably forms
    // exactly one bump -- this is the same "one stimulus -> one bump" premise
    // oneBumpAtPositionWithAmplitudeAndWidth's production callers rely on
    // (e.g. DetectionInstability). If this ever legitimately varies, the
    // stimulus parameters below need revisiting, not this assertion.
    REQUIRE(solution.observedBumps.size() == 1);

    // Same weights as Solution::twoBumpsAtPositionWithAmplitudeAndWidth.
    static constexpr int targetNumberOfBumps = 2;
    static constexpr double weightBumps = 0.70;
    static constexpr double weightPos   = 0.20 / targetNumberOfBumps;
    static constexpr double weightAmp   = 0.05 / targetNumberOfBumps;
    static constexpr double weightWidth = 0.05 / targetNumberOfBumps;

    const auto& bump = solution.observedBumps.front();
    const double bumpsTerm = weightBumps / (1.0 + std::abs(targetNumberOfBumps - 1));
    const double matchedBumpTerm =
        weightPos / (1.0 + std::abs(bump.centroid - SingleBumpTwoBumpsSolution::targetPosition)) +
        weightAmp / (1.0 + std::abs(bump.amplitude - SingleBumpTwoBumpsSolution::targetAmplitude)) +
        weightWidth / (1.0 + std::abs(bump.width - SingleBumpTwoBumpsSolution::targetWidth));

    // Buggy (pre-fix) behaviour: the same closest bump matched -- and
    // credited -- for both target slots.
    const double doubleCountedFitness = bumpsTerm + 2.0 * matchedBumpTerm;
    // Correct (post-fix) behaviour: the bump is consumed by the first target
    // slot; with no bumps left, the second slot contributes nothing.
    const double expectedFitness = bumpsTerm + matchedBumpTerm;

    REQUIRE(solution.getFitness() == Catch::Approx(expectedFitness).margin(1e-9));
    REQUIRE(solution.getFitness() < doubleCountedFitness - 1e-9);
}

// Issue #68: oneBumpAtPositionWithAmplitudeAndWidth's own zero-bump case had
// never been directly asserted -- only twoBumps/threeBumps had. A field that
// never receives a stimulus must score 0.0, same as the multi-bump helpers.
TEST_CASE("Solution oneBumpAtPositionWithAmplitudeAndWidth yields no bump credit for an empty field", "[Solution]")
{
    const auto topology = makeTopology(1, 1);
    EmptyFieldOneBumpSolution solution(topology);
    solution.initialize();

    REQUIRE_NOTHROW(solution.evaluate());
    REQUIRE(solution.getFitness() == 0.0);
}

// Issue #68: the missing-field-name guard, checked directly against
// oneBumpAtPositionWithAmplitudeAndWidth rather than only closenessToRestingLevel
// or twoBumpsAtPositionWithAmplitudeAndWidth.
TEST_CASE("Solution oneBumpAtPositionWithAmplitudeAndWidth throws on a field name that doesn't exist", "[Solution]")
{
    const auto topology = makeTopology(1, 1);
    MissingFieldOneBumpSolution solution(topology);
    solution.initialize();

    REQUIRE_THROWS_AS(solution.evaluate(), std::invalid_argument);
    REQUIRE(solution.getPhenotype().getNumberOfElements() == 0);
}

// Issue #68: exact match should score at (or extremely near) the theoretical
// maximum. SingleBumpOneBumpSolution queries oneBumpAtPositionWithAmplitudeAndWidth
// with the exact position/amplitude/width of the one real bump that formed, so
// bump count matches (1 == 1) and every distance term is exactly zero -- every
// weighted component (0.45 + 0.45 + 0.05 + 0.05) is credited in full.
TEST_CASE("Solution oneBumpAtPositionWithAmplitudeAndWidth scores the theoretical maximum on an exact match", "[Solution]")
{
    const auto topology = makeTopology(1, 1);
    SingleBumpOneBumpSolution solution(topology);
    solution.initialize();

    REQUIRE_NOTHROW(solution.evaluate());
    REQUIRE(solution.observedBumps.size() == 1);
    REQUIRE(solution.getFitness() == Catch::Approx(1.0).margin(1e-9));
}

// Issue #68: the missing-field-name guard, checked directly against
// threeBumpsAtPositionWithAmplitudeAndWidth.
TEST_CASE("Solution threeBumpsAtPositionWithAmplitudeAndWidth throws on a field name that doesn't exist", "[Solution]")
{
    const auto topology = makeTopology(1, 1);
    MissingFieldThreeBumpsSolution solution(topology);
    solution.initialize();

    REQUIRE_THROWS_AS(solution.evaluate(), std::invalid_argument);
    REQUIRE(solution.getPhenotype().getNumberOfElements() == 0);
}

// Issue #68 / #53: the injective-matching fix must also prevent a single real
// bump from being triple-counted against three identical target slots, not
// just double-counted against two. Targets equal the bump's own observed
// values, so only the first matched slot contributes a nonzero distance term;
// the second and third find an empty candidate pool (matchClosestBump returns
// std::nullopt) and contribute nothing.
TEST_CASE("Solution threeBumpsAtPositionWithAmplitudeAndWidth does not credit the same bump for all three target positions", "[Solution]")
{
    const auto topology = makeTopology(1, 1);
    SingleBumpThreeBumpsSolution solution(topology);
    solution.initialize();

    REQUIRE_NOTHROW(solution.evaluate());
    REQUIRE(solution.observedBumps.size() == 1);

    // Same weights as Solution::threeBumpsAtPositionWithAmplitudeAndWidth.
    static constexpr int targetNumberOfBumps = 3;
    static constexpr double weightBumps = 0.40;
    static constexpr double weightPos   = 0.20 / targetNumberOfBumps;
    static constexpr double weightAmp   = 0.20 / targetNumberOfBumps;
    static constexpr double weightWidth = 0.20 / targetNumberOfBumps;

    const double bumpsTerm = weightBumps / (1.0 + std::abs(targetNumberOfBumps - 1));
    // Target equals the observed bump exactly, so every distance term is zero.
    const double matchedBumpTerm = weightPos + weightAmp + weightWidth;

    const double tripleCountedFitness = bumpsTerm + 3.0 * matchedBumpTerm;
    const double expectedFitness = bumpsTerm + matchedBumpTerm;

    REQUIRE(solution.getFitness() == Catch::Approx(expectedFitness).margin(1e-9));
    REQUIRE(solution.getFitness() < tripleCountedFitness - 1e-9);
}

// Issue #68: the missing-field-name guard, checked directly against
// preShapednessAtPosition.
TEST_CASE("Solution preShapednessAtPosition throws on a field name that doesn't exist", "[Solution]")
{
    const auto topology = makeTopology(1, 1);
    MissingFieldPreShapednessSolution solution(topology);
    solution.initialize();

    REQUIRE_THROWS_AS(solution.evaluate(), std::invalid_argument);
    REQUIRE(solution.getPhenotype().getNumberOfElements() == 0);
}

// Issue #68: preShapednessAtPosition's zero-bump analogue -- a field that
// never receives a stimulus sits exactly at its startingRestingLevel, which
// fails the "must be higher than the resting level" guard, so the score must
// be exactly 0.0, not merely finite/bounded.
TEST_CASE("Solution preShapednessAtPosition scores zero for a field at its resting level", "[Solution]")
{
    const auto topology = makeTopology(1, 1);
    RestingLevelPreShapednessSolution solution(topology);
    solution.initialize();

    REQUIRE_NOTHROW(solution.evaluate());
    REQUIRE(solution.getFitness() == 0.0);
}

// Issue #68: the missing-field-name guard, checked directly against
// negativePreShapednessAtPosition -- the last of the five fitness primitives
// issue #68 names that previously had no dedicated test at all.
TEST_CASE("Solution negativePreShapednessAtPosition throws on a field name that doesn't exist", "[Solution]")
{
    const auto topology = makeTopology(1, 1);
    MissingFieldNegativePreShapednessSolution solution(topology);
    solution.initialize();

    REQUIRE_THROWS_AS(solution.evaluate(), std::invalid_argument);
    REQUIRE(solution.getPhenotype().getNumberOfElements() == 0);
}

// Issue #68: negativePreShapednessAtPosition's zero-bump analogue -- at rest,
// activation equals startingRestingLevel exactly, which fails the "must be
// lower than resting level minus epsilon" guard, so the score must be exactly
// 0.0.
TEST_CASE("Solution negativePreShapednessAtPosition scores zero for a field at its resting level", "[Solution]")
{
    const auto topology = makeTopology(1, 1);
    RestingLevelNegativePreShapednessSolution solution(topology);
    solution.initialize();

    REQUIRE_NOTHROW(solution.evaluate());
    REQUIRE(solution.getFitness() == 0.0);
}

// Issue #68 / #56: same boundary-index guard as preShapednessAtPosition
// (both route through the shared clampedIndexForPosition helper), checked
// directly against negativePreShapednessAtPosition at the field's own upper
// spatial bound.
TEST_CASE("Solution negativePreShapednessAtPosition does not read past the end of the field at the upper boundary", "[Solution]")
{
    const auto topology = makeTopology(1, 1);
    BoundaryPositionNegativePreShapednessSolution solution(topology);
    solution.initialize();

    REQUIRE_NOTHROW(solution.evaluate());
    REQUIRE(std::isfinite(solution.getFitness()));
    REQUIRE(solution.getFitness() >= 0.0);
    REQUIRE(solution.getFitness() <= 1.0);
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
