#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cmath>
#include <limits>
#include <type_traits>

#include <dnf_composer/tools/math.h>

#include "neat/solution.h"
#include "solutions/detection_instability.h"
#include "test_helpers.h"
#include "test_stub_solution.h"

using namespace neat_dnfs;
using namespace neat_dnfs::test;
using namespace dnf_composer::element;

TEST_CASE("Solution::getGenome and getPhenotype return references, not copies", "[Solution]")
{
    // Locks in that these hot-path accessors bind to the existing genome/phenotype
    // members instead of deep-copying a Genome (and its gene vectors) or a whole
    // dnf_composer::Simulation on every call in the O(n^2) evolutionary loop.
    static_assert(std::is_reference_v<decltype(std::declval<const Solution&>().getGenome())>);
    static_assert(std::is_reference_v<decltype(std::declval<const Solution&>().getPhenotype())>);
    SUCCEED();
}

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
        const size_t initialNumConnectionGenes = solution.getNumConnectionGenes();

        REQUIRE_NOTHROW(solution.mutate());
        REQUIRE(solution.getNumConnectionGenes() >= initialNumConnectionGenes);
    }
}

TEST_CASE("Solution Getters", "[Solution]")
{
    auto topology = makeTopology(1, 1);
    DetectionInstability solution(topology);

    REQUIRE(solution.getGenome().getFieldGenes().empty());
    REQUIRE(solution.getGenome().getConnectionGenes().empty());
    REQUIRE(solution.getFitness() == 0.0);
    REQUIRE(solution.getNumConnectionGenes() == 0);
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
// and indexes the "activation" component directly. A position at the field's own upper bound (xSize) --
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

TEST_CASE("Solution moveGaussianStimulusContinuously simulates a leftward move as long as a rightward one", "[Solution]")
{
    const auto topology = makeTopology(1, 1);
    MovingStimulusSolution solution(topology);
    solution.initialize();

    REQUIRE_NOTHROW(solution.evaluate());
    const auto& durations = solution.getParameters().partialFitness;
    REQUIRE(durations.size() == 2);
    const double leftwardDuration = durations[0];
    const double rightwardDuration = durations[1];

    REQUIRE(rightwardDuration > 0.0);
    REQUIRE(leftwardDuration == rightwardDuration);
}

TEST_CASE("Solution addBoostStimulus lifts the whole field and removeBoostStimuli lets it return to rest", "[Solution]")
{
    const auto topology = makeTopology(1, 1);
    BoostStimulusSolution solution(topology);
    solution.initialize();

    REQUIRE_NOTHROW(solution.evaluate());
    const auto& lift = solution.getParameters().partialFitness;
    REQUIRE(lift.size() == 2);
    // Noise moves single cells by a few hundredths, so a uniform lift shows within a tenth of the amplitude.
    CHECK(lift[0] > 0.9 * BoostStimulusSolution::boostAmplitude);
    CHECK(lift[1] < 0.1 * BoostStimulusSolution::boostAmplitude);
}

TEST_CASE("Solution negativePreShapingDepthAtPosition scores trough depth below rest, not against the field's peak", "[Solution]")
{
    const auto topology = makeTopology(1, 1);
    NegativePreShapingDepthSolution solution(topology);
    solution.initialize();

    REQUIRE_NOTHROW(solution.evaluate());
    const auto& scores = solution.getParameters().partialFitness;
    REQUIRE(scores.size() == 2);

    REQUIRE(scores[0] == Catch::Approx(1.0).margin(1e-9));
    REQUIRE(scores[1] == Catch::Approx(std::exp(-0.5)).margin(1e-9));
}

TEST_CASE("Solution moveGaussianStimulusContinuously throws for a stimulus that does not exist", "[Solution][MoveStimulus]")
{
    MoveScenarioSolution solution(makeTopology(1, 1), MoveScenario{ "gs nf 1 30.000000", 30.0, 10.0, -5.0 });
    solution.initialize();

    REQUIRE_THROWS_AS(solution.evaluate(), std::invalid_argument);
}

TEST_CASE("Solution moveGaussianStimulusContinuously throws for a step pointing away from the target", "[Solution][MoveStimulus]")
{
    MoveScenarioSolution solution(makeTopology(1, 1), MoveScenario{ "gs nf 1 30", 30.0, 10.0, 5.0 });
    solution.initialize();

    REQUIRE_THROWS_AS(solution.evaluate(), std::invalid_argument);
}

TEST_CASE("Solution moveGaussianStimulusContinuously throws for a zero step", "[Solution][MoveStimulus]")
{
    MoveScenarioSolution solution(makeTopology(1, 1), MoveScenario{ "gs nf 1 30", 30.0, 10.0, 0.0 });
    solution.initialize();

    REQUIRE_THROWS_AS(solution.evaluate(), std::invalid_argument);
}

TEST_CASE("Solution moveGaussianStimulusContinuously lands on a target that is not a whole number of steps away", "[Solution][MoveStimulus]")
{
    MoveScenarioSolution solution(makeTopology(1, 1), MoveScenario{ "gs nf 1 30", 30.0, 12.0, -5.0 });
    solution.initialize();

    REQUIRE_NOTHROW(solution.evaluate());
    const auto& recorded = solution.getParameters().partialFitness;
    const double finalPosition = recorded[0];
    const double moveDuration = recorded[1];
    REQUIRE(finalPosition == Catch::Approx(12.0).margin(1e-9));
    REQUIRE(moveDuration > 0.0);
}

TEST_CASE("Solution moveGaussianStimulusContinuously does nothing when the stimulus is already at the target", "[Solution][MoveStimulus]")
{
    MoveScenarioSolution solution(makeTopology(1, 1), MoveScenario{ "gs nf 1 30", 30.0, 30.0, 5.0 });
    solution.initialize();

    REQUIRE_NOTHROW(solution.evaluate());
    const auto& recorded = solution.getParameters().partialFitness;
    const double finalPosition = recorded[0];
    const double moveDuration = recorded[1];
    REQUIRE(finalPosition == Catch::Approx(30.0).margin(1e-9));
    REQUIRE(moveDuration == 0.0);
}

TEST_CASE("Solution moveGaussianStimulusContinuously throws for a step that is not finite", "[Solution][MoveStimulus]")
{
    const double step = GENERATE(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity());
    MoveScenarioSolution solution(makeTopology(1, 1), MoveScenario{ "gs nf 1 30", 30.0, 50.0, step });
    solution.initialize();

    REQUIRE_THROWS_AS(solution.evaluate(), std::invalid_argument);
}

// The simulation steps per increment are the step budget divided by the
// number of increments; a step small enough to need more increments than
// the budget truncated that to zero, so the stimulus moved without the
// field ever being simulated.
TEST_CASE("Solution moveGaussianStimulusContinuously simulates a move with more increments than simulation steps", "[Solution][MoveStimulus]")
{
    const double step = 20.0 / (2.0 * SimulationConstants::maxSimulationSteps);
    MoveScenarioSolution solution(makeTopology(1, 1), MoveScenario{ "gs nf 1 30", 30.0, 50.0, step });
    solution.initialize();

    REQUIRE_NOTHROW(solution.evaluate());
    const auto& recorded = solution.getParameters().partialFitness;
    const double finalPosition = recorded[0];
    const double moveDuration = recorded[1];
    REQUIRE(finalPosition == Catch::Approx(50.0).margin(1e-6));
    REQUIRE(moveDuration > 0.0);
}

// Every bump-matching function documents a score in [0,1]. Shares that sum
// to 1 but include a negative one can push the score above 1, e.g. a matched
// position weighted 2.0 with a mismatched amplitude weighted -1.0.
TEST_CASE("Solution bump-matching functions reject a negative weight share", "[Solution]")
{
    const auto function = GENERATE(BumpFunction::OneBump, BumpFunction::TwoBumps, BumpFunction::JustOneBump);
    BumpWeightsSolution solution(makeTopology(1, 1), function, BumpFitnessWeights{ 0.0, 2.0, -1.0, 0.0 });
    solution.initialize();

    REQUIRE_THROWS_AS(solution.evaluate(), std::invalid_argument);
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
    // SingleBumpTwoBumpsSolution pins its field and kernel parameters
    // explicitly (see makeFixedFieldGene), so a single stimulus forms exactly
    // one bump on every construction rather than ~98% of them. If this ever
    // fails again it means the fixed regime stopped producing a stable bump --
    // fix that there, rather than weakening this assertion.
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
// never been directly asserted -- only twoBumps had. A field that
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

// Adds a Gaussian of the given amplitude and width, centred on a field position, to an activation profile. It uses
// dnf-composer's own Gaussian, so the profile sits where dnf-composer would put a stimulus at that position.
static void addGaussianProfile(std::vector<double>& activation, const double amplitude, const double position, const double width)
{
    const auto gauss = dnf_composer::tools::math::circularGauss<double>(static_cast<uint32_t>(activation.size()), width,
        position / DimensionConstants::dx);
    for (size_t i = 0; i < activation.size(); ++i)
    {
        activation[i] += amplitude * gauss[i];
    }
}

static void runFieldProbe(const FieldProbeSolution::Scenario& scenario)
{
    FieldProbeSolution probe(makeTopology(1, 1), scenario);
    probe.initialize();
    probe.evaluate();
}

// A profile this narrow raises its own sample only: its neighbours move by under 0.4% of the amplitude.
static constexpr double oneSampleWidth = 0.3;

TEST_CASE("Solution preShapednessAtPosition reads the activation at the position it is given", "[Solution][FieldHelpers]")
{
    resetGlobalState();
    double score = -1.0;
    runFieldProbe([&](FieldProbeSolution& probe) {
        probe.initSimulation();
        auto& activation = probe.activationOf("nf 1");
        const double restingLevel = probe.restingLevelOf("nf 1");
        std::ranges::fill(activation, restingLevel);
        addGaussianProfile(activation, -restingLevel / 2.0, 50.0, oneSampleWidth);
        score = probe.preShapednessAtPosition("nf 1", 50.0, 0.5, BumpFitnessDefaults::preShapednessEpsilon);
    });

    CHECK(score == Catch::Approx(1.0));
}

TEST_CASE("Solution preShapednessAtPosition scores 1 at half the resting level, falling off as a Gaussian of width sigma", "[Solution][FieldHelpers]")
{
    resetGlobalState();
    static constexpr double sigma = 2.0;
    std::vector<double> scores;
    runFieldProbe([&](FieldProbeSolution& probe) {
        probe.initSimulation();
        auto& activation = probe.activationOf("nf 1");
        const double target = probe.restingLevelOf("nf 1") / 2.0;
        for (const double level : { target, target + sigma, target - sigma })
        {
            std::ranges::fill(activation, level);
            scores.push_back(probe.preShapednessAtPosition("nf 1", 50.0, sigma, BumpFitnessDefaults::preShapednessEpsilon));
        }
    });

    REQUIRE(scores.size() == 3);
    CHECK(scores[0] == Catch::Approx(1.0));
    CHECK(scores[1] == Catch::Approx(std::exp(-0.5)));
    CHECK(scores[2] == Catch::Approx(std::exp(-0.5)));
}

TEST_CASE("Solution preShapednessAtPosition scores zero at or above the threshold", "[Solution][FieldHelpers]")
{
    resetGlobalState();
    std::vector<double> scores;
    runFieldProbe([&](FieldProbeSolution& probe) {
        probe.initSimulation();
        auto& activation = probe.activationOf("nf 1");
        for (const double level : { 0.0, 1.0 })
        {
            std::ranges::fill(activation, level);
            scores.push_back(probe.preShapednessAtPosition("nf 1", 50.0));
        }
    });

    CHECK(scores == std::vector<double>{ 0.0, 0.0 });
}

TEST_CASE("Solution preShapednessAtPosition scores zero within epsilon of the resting level and below it", "[Solution][FieldHelpers]")
{
    resetGlobalState();
    static constexpr double epsilon = 0.1;
    std::vector<double> scores;
    runFieldProbe([&](FieldProbeSolution& probe) {
        probe.initSimulation();
        auto& activation = probe.activationOf("nf 1");
        const double restingLevel = probe.restingLevelOf("nf 1");
        for (const double level : { restingLevel + epsilon / 2.0, restingLevel - 1.0 })
        {
            std::ranges::fill(activation, level);
            scores.push_back(probe.preShapednessAtPosition("nf 1", 50.0, BumpFitnessDefaults::preShapednessSigma, epsilon));
        }
    });

    CHECK(scores == std::vector<double>{ 0.0, 0.0 });
}

TEST_CASE("Solution negativePreShapingDepthAtPosition reads the depth at the position it is given", "[Solution][FieldHelpers]")
{
    resetGlobalState();
    static constexpr double depth = 6.0;
    double score = -1.0;
    runFieldProbe([&](FieldProbeSolution& probe) {
        probe.initSimulation();
        auto& activation = probe.activationOf("nf 1");
        std::ranges::fill(activation, probe.restingLevelOf("nf 1"));
        addGaussianProfile(activation, -depth, 50.0, oneSampleWidth);
        score = probe.negativePreShapingDepthAtPosition("nf 1", 50.0, depth, 0.5);
    });

    CHECK(score == Catch::Approx(1.0));
}

TEST_CASE("Solution negativePreShapingDepthAtPosition scores 1 at the target depth, falling off as a Gaussian, and little at rest", "[Solution][FieldHelpers]")
{
    resetGlobalState();
    static constexpr double depth = 6.0;
    static constexpr double width = 2.0;
    std::vector<double> scores;
    runFieldProbe([&](FieldProbeSolution& probe) {
        probe.initSimulation();
        auto& activation = probe.activationOf("nf 1");
        const double restingLevel = probe.restingLevelOf("nf 1");
        for (const double level : { restingLevel - depth, restingLevel - depth - width, restingLevel })
        {
            std::ranges::fill(activation, level);
            scores.push_back(probe.negativePreShapingDepthAtPosition("nf 1", 50.0, depth, width));
        }
    });

    REQUIRE(scores.size() == 3);
    CHECK(scores[0] == Catch::Approx(1.0));
    CHECK(scores[1] == Catch::Approx(std::exp(-0.5)));
    CHECK(scores[2] == Catch::Approx(std::exp(-0.5 * (depth / width) * (depth / width))));
}

TEST_CASE("Solution noBumps scores 1 while a field stays below threshold and exp(-rate x height) once it peaks", "[Solution][FieldHelpers]")
{
    resetGlobalState();
    static constexpr double decayRate = BumpFitnessDefaults::noBumpsDecayRate;
    double belowThreshold = -1.0;
    double peaked = -1.0;
    double peakHeight = 0.0;
    runFieldProbe([&](FieldProbeSolution& probe) {
        probe.initSimulation();
        probe.runSimulation(static_cast<int>(SimulationConstants::maxSimulationSteps));
        belowThreshold = probe.noBumps("nf 1", decayRate);

        probe.stimulate("nf 1", GaussStimulusConstants::amplitude, 50.0);
        probe.runSimulation(static_cast<int>(SimulationConstants::maxSimulationSteps));
        peakHeight = probe.highestActivationOf("nf 1");
        peaked = probe.noBumps("nf 1", decayRate);
    });

    CHECK(belowThreshold == 1.0);
    REQUIRE(peakHeight > 0.0);
    CHECK(peaked == Catch::Approx(std::exp(-decayRate * peakHeight)));
}

TEST_CASE("Solution oneBumpAtPositionWithAmplitudeAndWidth places a stimulus's bump at the stimulus's own position", "[Solution][FieldHelpers]")
{
    resetGlobalState();
    std::vector<NeuralFieldBump> bumps;
    double score = -1.0;
    runFieldProbe([&](FieldProbeSolution& probe) {
        probe.initSimulation();
        probe.stimulate("nf 1", GaussStimulusConstants::amplitude, 50.0);
        probe.runSimulation(static_cast<int>(SimulationConstants::maxSimulationSteps));
        bumps = probe.bumpsOf("nf 1");
        if (bumps.size() == 1)
        {
            score = probe.oneBumpAtPositionWithAmplitudeAndWidth("nf 1", 50.0, bumps.front().amplitude, bumps.front().width);
        }
    });

    REQUIRE(bumps.size() == 1);
    // A bump's centroid is the middle of its first and last samples, so an asymmetric flank can move it by half a sample.
    CHECK(std::abs(bumps.front().centroid - 50.0) <= 0.5);
    CHECK(score == Catch::Approx(0.45 + 0.45 / (1.0 + std::abs(bumps.front().centroid - 50.0)) + 0.05 + 0.05));
}

TEST_CASE("Solution oneBumpAtPositionWithAmplitudeAndWidth gives two bumps half the count credit and scores the closest", "[Solution][FieldHelpers]")
{
    resetGlobalState();
    std::vector<NeuralFieldBump> bumps;
    double score = -1.0;
    runFieldProbe([&](FieldProbeSolution& probe) {
        probe.initSimulation();
        probe.stimulate("nf 1", GaussStimulusConstants::amplitude, 30.0);
        probe.stimulate("nf 1", GaussStimulusConstants::amplitude, 70.0);
        probe.runSimulation(static_cast<int>(SimulationConstants::maxSimulationSteps));
        bumps = probe.bumpsOf("nf 1");
        if (bumps.size() == 2)
        {
            score = probe.oneBumpAtPositionWithAmplitudeAndWidth("nf 1", 30.0, bumps.front().amplitude, bumps.front().width);
        }
    });

    REQUIRE(bumps.size() == 2);
    const auto& closest = bumps.front();
    REQUIRE(std::abs(closest.centroid - 30.0) <= 0.5);
    CHECK(score == Catch::Approx(0.45 / 2.0 + 0.45 / (1.0 + std::abs(closest.centroid - 30.0)) + 0.05 + 0.05));
}

TEST_CASE("Solution closenessOfNearestBumpToPosition is 1 within the tolerance of a bump and falls off as a Gaussian beyond it", "[Solution][FieldHelpers]")
{
    resetGlobalState();
    static constexpr double tolerance = 2.0;
    std::vector<NeuralFieldBump> bumps;
    std::vector<double> scores;
    runFieldProbe([&](FieldProbeSolution& probe) {
        probe.initSimulation();
        probe.stimulate("nf 1", GaussStimulusConstants::amplitude, 50.0);
        probe.runSimulation(static_cast<int>(SimulationConstants::maxSimulationSteps));
        bumps = probe.bumpsOf("nf 1");
        if (bumps.size() == 1)
        {
            for (const double offset : { 0.0, tolerance, tolerance + 4.0, 20.0 })
            {
                scores.push_back(probe.closenessOfNearestBumpToPosition("nf 1", bumps.front().centroid + offset, tolerance));
            }
        }
    });

    REQUIRE(bumps.size() == 1);
    REQUIRE(scores.size() == 4);
    CHECK(scores[0] == Catch::Approx(1.0));
    CHECK(scores[1] == Catch::Approx(1.0));
    CHECK(scores[2] == Catch::Approx(std::exp(-16.0 / (2.0 * tolerance * tolerance))));
    CHECK(scores[3] < 1e-6);
}

TEST_CASE("Solution closenessOfNearestBumpToPosition scores the nearest of several bumps, and 0 with none", "[Solution][FieldHelpers]")
{
    resetGlobalState();
    std::vector<NeuralFieldBump> bumps;
    double withoutBumps = -1.0;
    double nearSecondBump = -1.0;
    runFieldProbe([&](FieldProbeSolution& probe) {
        probe.initSimulation();
        probe.runSimulation(static_cast<int>(SimulationConstants::maxSimulationSteps));
        withoutBumps = probe.closenessOfNearestBumpToPosition("nf 1", 70.0);
        probe.stimulate("nf 1", GaussStimulusConstants::amplitude, 30.0);
        probe.stimulate("nf 1", GaussStimulusConstants::amplitude, 70.0);
        probe.runSimulation(static_cast<int>(SimulationConstants::maxSimulationSteps));
        bumps = probe.bumpsOf("nf 1");
        nearSecondBump = probe.closenessOfNearestBumpToPosition("nf 1", 70.0);
    });

    CHECK(withoutBumps == 0.0);
    REQUIRE(bumps.size() == 2);
    CHECK(nearSecondBump == Catch::Approx(1.0));
}

TEST_CASE("Solution preShapeAboveFieldAtPosition scores a sub-threshold activation by its height above the field", "[Solution][FieldHelpers]")
{
    resetGlobalState();
    static constexpr double fullElevation = 2.0;
    std::vector<double> scores;
    runFieldProbe([&](FieldProbeSolution& probe) {
        probe.initSimulation();
        auto& activation = probe.activationOf("nf 1");
        std::ranges::fill(activation, probe.restingLevelOf("nf 1"));
        addGaussianProfile(activation, 2.0 * fullElevation, 50.0, oneSampleWidth);
        addGaussianProfile(activation, 0.5 * fullElevation, 70.0, oneSampleWidth);
        for (const double position : { 50.0, 70.0, 30.0 })
        {
            scores.push_back(probe.preShapeAboveFieldAtPosition("nf 1", position, fullElevation));
        }
    });

    REQUIRE(scores.size() == 3);
    CHECK(scores[0] == Catch::Approx(1.0));
    CHECK(scores[1] == Catch::Approx(0.5));
    CHECK(scores[2] == 0.0);
}

TEST_CASE("Solution preShapeAboveFieldAtPosition measures the height from the field's median, not its resting level", "[Solution][FieldHelpers]")
{
    resetGlobalState();
    static constexpr double fullElevation = 2.0;
    std::vector<double> scores;
    runFieldProbe([&](FieldProbeSolution& probe) {
        probe.initSimulation();
        auto& activation = probe.activationOf("nf 1");
        const double restingLevel = probe.restingLevelOf("nf 1");
        std::ranges::fill(activation, restingLevel / 2.0);
        scores.push_back(probe.preShapeAboveFieldAtPosition("nf 1", 50.0, fullElevation));
        addGaussianProfile(activation, 0.5 * fullElevation, 50.0, oneSampleWidth);
        scores.push_back(probe.preShapeAboveFieldAtPosition("nf 1", 50.0, fullElevation));
    });

    REQUIRE(scores.size() == 2);
    CHECK(scores[0] == 0.0);
    CHECK(scores[1] == Catch::Approx(0.5));
}

TEST_CASE("Solution preShapeAboveFieldAtPosition scores zero at or above the threshold", "[Solution][FieldHelpers]")
{
    resetGlobalState();
    std::vector<double> scores;
    runFieldProbe([&](FieldProbeSolution& probe) {
        probe.initSimulation();
        auto& activation = probe.activationOf("nf 1");
        const double restingLevel = probe.restingLevelOf("nf 1");
        std::ranges::fill(activation, restingLevel);
        addGaussianProfile(activation, -restingLevel, 50.0, oneSampleWidth);
        addGaussianProfile(activation, 1.0 - restingLevel, 70.0, oneSampleWidth);
        scores.push_back(probe.preShapeAboveFieldAtPosition("nf 1", 50.0));
        scores.push_back(probe.preShapeAboveFieldAtPosition("nf 1", 70.0));
    });

    CHECK(scores == std::vector<double>{ 0.0, 0.0 });
}

TEST_CASE("Solution troughBelowFieldAtPosition scores an activation by its depth below the field", "[Solution][FieldHelpers]")
{
    resetGlobalState();
    static constexpr double fullDepth = 2.0;
    static constexpr double liftedField = -4.0;
    std::vector<double> scores;
    runFieldProbe([&](FieldProbeSolution& probe) {
        probe.initSimulation();
        auto& activation = probe.activationOf("nf 1");
        std::ranges::fill(activation, liftedField);
        addGaussianProfile(activation, -2.0 * fullDepth, 50.0, oneSampleWidth);
        addGaussianProfile(activation, -0.5 * fullDepth, 70.0, oneSampleWidth);
        addGaussianProfile(activation, 3.0, 90.0, oneSampleWidth);
        for (const double position : { 50.0, 70.0, 30.0, 90.0 })
        {
            scores.push_back(probe.troughBelowFieldAtPosition("nf 1", position, fullDepth));
        }
    });

    REQUIRE(scores.size() == 4);
    CHECK(scores[0] == Catch::Approx(1.0));
    CHECK(scores[1] == Catch::Approx(0.5));
    CHECK(scores[2] == 0.0);
    CHECK(scores[3] == 0.0);
}

TEST_CASE("Solution troughBelowFieldAtPosition scores zero for a field lowered as a whole", "[Solution][FieldHelpers]")
{
    resetGlobalState();
    double score = -1.0;
    runFieldProbe([&](FieldProbeSolution& probe) {
        probe.initSimulation();
        std::ranges::fill(probe.activationOf("nf 1"), probe.restingLevelOf("nf 1") - 5.0);
        score = probe.troughBelowFieldAtPosition("nf 1", 50.0);
    });

    CHECK(score == 0.0);
}

TEST_CASE("Solution activationAtPosition reads the sample dnf-composer places at the position", "[Solution][FieldHelpers]")
{
    resetGlobalState();
    std::vector<double> activations;
    runFieldProbe([&](FieldProbeSolution& probe) {
        probe.initSimulation();
        auto& activation = probe.activationOf("nf 1");
        std::ranges::fill(activation, -4.0);
        activation[49] = 3.0; // position 50
        for (const double position : { 50.0, 51.0 })
        {
            activations.push_back(probe.activationAtPosition("nf 1", position));
        }
    });

    CHECK(activations == std::vector<double>{ 3.0, -4.0 });
}

TEST_CASE("Solution medianActivation is the field's median activation, which a peak does not move", "[Solution][FieldHelpers]")
{
    resetGlobalState();
    std::vector<double> medians;
    runFieldProbe([&](FieldProbeSolution& probe) {
        probe.initSimulation();
        auto& activation = probe.activationOf("nf 1");
        std::ranges::fill(activation, -4.0);
        medians.push_back(probe.medianActivation("nf 1"));
        addGaussianProfile(activation, 10.0, 50.0, oneSampleWidth);
        medians.push_back(probe.medianActivation("nf 1"));
        std::fill(activation.begin(), activation.begin() + static_cast<long>(activation.size() / 2) + 1, -8.0);
        medians.push_back(probe.medianActivation("nf 1"));
    });

    CHECK(medians == std::vector<double>{ -4.0, -4.0, -8.0 });
}

TEST_CASE("Solution addTimedGaussianStimulus drives its field inside its time window only", "[Solution][FieldHelpers]")
{
    resetGlobalState();
    static constexpr double windowStart = 20.0;
    // Five time constants: long enough for the field to come close to what the stimulus drives it to.
    const double windowEnd = windowStart + 5.0 * NeuralFieldConstants::tau;
    double liftBeforeWindow = -1.0;
    double liftInWindow = -1.0;
    runFieldProbe([&](FieldProbeSolution& probe) {
        probe.addTimedGaussianStimulus("tgs probe", "nf 1",
            TimedGaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0,
                { { windowStart, windowEnd } }, true, false },
            ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
        probe.initSimulation();
        const double restingLevel = probe.restingLevelOf("nf 1");
        // dnf-composer samples position 50 at index 49.
        probe.runUntil(windowStart - SimulationConstants::deltaT);
        liftBeforeWindow = probe.activationOf("nf 1")[49] - restingLevel;
        probe.runUntil(windowEnd);
        liftInWindow = probe.activationOf("nf 1")[49] - restingLevel;
    });

    CHECK(std::abs(liftBeforeWindow) < 0.5);
    CHECK(liftInWindow > 5.0);
}

TEST_CASE("Solution activationsDescendAcrossPositions scores the share of consecutive positions in descending order", "[Solution][FieldHelpers]")
{
    resetGlobalState();
    std::vector<double> scores;
    runFieldProbe([&](FieldProbeSolution& probe) {
        probe.initSimulation();
        auto& activation = probe.activationOf("nf 1");
        std::ranges::fill(activation, probe.restingLevelOf("nf 1"));
        addGaussianProfile(activation, 8.0, 70.0, 2.0);
        addGaussianProfile(activation, 6.0, 50.0, 2.0);
        addGaussianProfile(activation, 4.0, 20.0, 2.0);
        addGaussianProfile(activation, 2.0, 40.0, 2.0);
        scores.push_back(probe.activationsDescendAcrossPositions("nf 1", { 70.0, 50.0, 20.0, 40.0 }));
        scores.push_back(probe.activationsDescendAcrossPositions("nf 1", { 70.0, 20.0, 50.0, 40.0 }));
        scores.push_back(probe.activationsDescendAcrossPositions("nf 1", { 40.0, 20.0, 50.0, 70.0 }));
    });

    REQUIRE(scores.size() == 3);
    CHECK(scores[0] == Catch::Approx(1.0));
    CHECK(scores[1] == Catch::Approx(2.0 / 3.0));
    CHECK(scores[2] == Catch::Approx(0.0));
}

TEST_CASE("Solution activationsDescendAcrossPositions counts two equal activations as out of order", "[Solution][FieldHelpers]")
{
    resetGlobalState();
    double score = -1.0;
    runFieldProbe([&](FieldProbeSolution& probe) {
        probe.initSimulation();
        std::ranges::fill(probe.activationOf("nf 1"), probe.restingLevelOf("nf 1"));
        score = probe.activationsDescendAcrossPositions("nf 1", { 20.0, 40.0 });
    });

    CHECK(score == 0.0);
}

TEST_CASE("Solution activationsDescendAcrossPositions reads the activations at the positions it is given", "[Solution][FieldHelpers]")
{
    resetGlobalState();
    double score = -1.0;
    runFieldProbe([&](FieldProbeSolution& probe) {
        probe.initSimulation();
        auto& activation = probe.activationOf("nf 1");
        std::ranges::fill(activation, probe.restingLevelOf("nf 1"));
        addGaussianProfile(activation, 5.0, 50.0, oneSampleWidth);
        score = probe.activationsDescendAcrossPositions("nf 1", { 50.0, 49.0 });
    });

    CHECK(score == 1.0);
}

TEST_CASE("Solution activationsDescendAcrossPositions throws for fewer than two positions", "[Solution][FieldHelpers]")
{
    resetGlobalState();
    REQUIRE_THROWS_AS(runFieldProbe([](FieldProbeSolution& probe) {
        probe.initSimulation();
        static_cast<void>(probe.activationsDescendAcrossPositions("nf 1", { 50.0 }));
    }), std::invalid_argument);
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
    // genome from the phenotype via translatePhenotypeToGenome() and would
    // therefore carry a different innovation numbering; see
    // "Solution::translatePhenotypeToGenome recovers the exact original field
    // genes" above for what the round-trip does guarantee).
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
    // getNumConnectionGenes() counts connection genes only; a freshly initialized
    // solution has field genes but no connections yet, so check field genes.
    REQUIRE(!solution.getGenome().getFieldGenes().empty());

    solution.clearGenome();
    REQUIRE(solution.getNumConnectionGenes() == 0);
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

TEST_CASE("Solution::translatePhenotypeToGenome recovers the exact original field genes", "[Solution]")
{
    // Field gene role (INPUT/OUTPUT/HIDDEN) is recovered from the gene's id via
    // initialTopology's declared ordering (see solution.cpp
    // translatePhenotypeToGenome), not guessed from connection degree, so the
    // decode reproduces the original field count and types exactly -- even
    // with zero connection genes, as here.
    resetGlobalState();
    const auto topology = makeTopology(1, 1);
    DetectionInstability solution(topology);
    solution.initialize();

    const auto originalFieldGenes = solution.getGenome().getFieldGenes();
    solution.buildPhenotype();

    REQUIRE_NOTHROW(solution.translatePhenotypeToGenome());
    const auto& decodedFieldGenes = solution.getGenome().getFieldGenes();

    REQUIRE(decodedFieldGenes.size() == originalFieldGenes.size());
    for (size_t i = 0; i < originalFieldGenes.size(); ++i)
    {
        CHECK(decodedFieldGenes[i].getParameters().id == originalFieldGenes[i].getParameters().id);
        CHECK(decodedFieldGenes[i].getParameters().type == originalFieldGenes[i].getParameters().type);
    }
}

TEST_CASE("Solution::translatePhenotypeToGenome keeps new innovation numbers clear of decoded ones", "[Solution]")
{
    // A template decoded at the start of incremental evolution carries its own
    // innovation numbers in its connection names ("gk cg 1-2 7"). Connections
    // added by mutation afterwards must not be given those numbers again, or
    // two different connections would share one and corrupt compatibility and
    // crossover alignment.
    resetGlobalState();
    const auto topology = makeTopology(1, 1);
    DetectionInstability solution(topology);
    solution.initialize();
    solution.addConnectionGene(ConnectionGene({ 1, 2 }, 7));
    solution.buildPhenotype();

    resetGlobalState();
    REQUIRE_NOTHROW(solution.translatePhenotypeToGenome());
    REQUIRE(solution.getGenome().getConnectionGenes().front().getInnovationNumber() == 7);

    CHECK(Genome::allocateInnovationNumber({ 2, 1 }) > 7);
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

TEST_CASE("SolutionParameters::toString separates partial fitness from species with a comma", "[Solution]")
{
    // The analysis dashboard parses this line with a regex anchored on "), spec.:".
    // A std::format refactor once dropped that comma, which silently made every new
    // per_generation_overview.txt unreadable while the C++ suite stayed green.
    SolutionParameters parameters{ 0.5, 0.25, 3 };
    parameters.speciesId = 7;
    parameters.partialFitness = { 0.25, 0.75 };

    const std::string text = parameters.toString();

    REQUIRE(text.find("), spec.: 7") != std::string::npos);
}

// closenessToRestingLevel scores the single highest cell of a field. The
// maximum of ~100 zero-mean noisy cells sits ~2.5 sd above rest, so under
// selection-instability's regime (noise amplitude 0.2, deltaT 10) a field
// sitting exactly at its resting level scored ~0.92 instead of ~1.0, capping
// that task's p3/p4 no matter how good the solution was. Scoring the mean
// activation must put a field at rest near 1.0 under the same noise.
TEST_CASE("Solution closenessOfMeanActivationToRestingLevel scores a noisy field at rest near 1.0", "[Solution]")
{
    const ScopedNoiseAndTimestep noisyRegime{ 0.2, 10.0 };
    RestingLevelClosenessSolution solution(makeTopology(1, 1), false);
    solution.initialize();

    static constexpr int evaluations = 20;
    for (int i = 0; i < evaluations; ++i)
    {
        solution.evaluate();
        REQUIRE(solution.getFitness() >= 0.98);
    }
}

// Tolerating noise must not mean tolerating a field that is genuinely off
// rest: with a stimulus holding the whole field below rest, the mean-activation
// score must fall below what the old max-based helper gave a field that was
// only noisy.
TEST_CASE("Solution closenessOfMeanActivationToRestingLevel still penalises a field held off rest", "[Solution]")
{
    RestingLevelClosenessSolution solution(makeTopology(1, 1), true);
    solution.initialize();
    solution.evaluate();

    REQUIRE(solution.getFitness() < 0.9);
}
