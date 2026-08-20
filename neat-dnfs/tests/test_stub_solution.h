#pragma once
#include <atomic>
#include <thread>
#include <stdexcept>

#include "neat/solution.h"
#include "test_helpers.h"

namespace neat_dnfs::test {

// Fast stand-in for a real Solution used only to observe how many
// evaluations run concurrently. testPhenotype() holds its "slot" for a short
// sleep so that concurrent evaluations overlap long enough to be counted.
class CountingSolution final : public Solution
{
public:
    explicit CountingSolution(const SolutionTopology& topology)
        : Solution(topology)
    {
        name = "Counting";
    }

    CountingSolution(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype)
        : Solution(initialTopology, phenotype)
    {
        name = "Counting";
    }

    SolutionPtr clone() const override
    {
        CountingSolution solution(initialTopology);
        return std::make_shared<CountingSolution>(solution);
    }

    SolutionPtr copy() const override
    {
        CountingSolution solution(initialTopology, phenotype);
        return std::make_shared<CountingSolution>(solution);
    }

    static std::atomic<int> live;
    static std::atomic<int> peak;

private:
    void testPhenotype() override
    {
        const int now = ++live;
        int previousPeak = peak.load();
        while (now > previousPeak && !peak.compare_exchange_weak(previousPeak, now)) {}

        std::this_thread::sleep_for(std::chrono::milliseconds(2));

        parameters.fitness = static_cast<double>(getNumFieldGenes() + getNumConnectionGenes()) / 100.0;
        --live;
    }

    void createPhenotypeEnvironment() override {}
};

inline std::atomic<int> CountingSolution::live{0};
inline std::atomic<int> CountingSolution::peak{0};

// Fast stand-in that throws from a deterministic subset of evaluations, used
// to verify Population::evaluate() propagates exceptions from parallel work.
class ThrowingSolution final : public Solution
{
public:
    explicit ThrowingSolution(const SolutionTopology& topology)
        : Solution(topology)
    {
        name = "Throwing";
    }

    ThrowingSolution(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype)
        : Solution(initialTopology, phenotype)
    {
        name = "Throwing";
    }

    SolutionPtr clone() const override
    {
        ThrowingSolution solution(initialTopology);
        return std::make_shared<ThrowingSolution>(solution);
    }

    SolutionPtr copy() const override
    {
        ThrowingSolution solution(initialTopology, phenotype);
        return std::make_shared<ThrowingSolution>(solution);
    }

private:
    void testPhenotype() override
    {
        if (getId() % 7 == 3)
            throw std::runtime_error("boom");

        parameters.fitness = static_cast<double>(getNumFieldGenes() + getNumConnectionGenes()) / 100.0;
    }

    void createPhenotypeEnvironment() override {}
};

// Stand-in whose fitness is set directly by the test rather than computed by
// a real DNF simulation, for tests that need a specific, deterministic
// fitness value (e.g. proving fitter-parent inheritance in crossover(), or
// fitness-improvement tracking) without depending on task-specific,
// seed-dependent evaluate() outcomes.
class FixedFitnessSolution final : public Solution
{
public:
    FixedFitnessSolution(const SolutionTopology& topology, double fitness)
        : Solution(topology)
    {
        name = "FixedFitness";
        fitnessToReport = fitness;
    }

    FixedFitnessSolution(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype)
        : Solution(initialTopology, phenotype)
    {
        name = "FixedFitness";
    }

    SolutionPtr clone() const override
    {
        FixedFitnessSolution solution(initialTopology, fitnessToReport);
        return std::make_shared<FixedFitnessSolution>(solution);
    }

    SolutionPtr copy() const override
    {
        FixedFitnessSolution solution(initialTopology, phenotype);
        solution.fitnessToReport = fitnessToReport;
        return std::make_shared<FixedFitnessSolution>(solution);
    }

private:
    double fitnessToReport = 0.0;

    void testPhenotype() override
    {
        parameters.fitness = fitnessToReport;
    }

    void createPhenotypeEnvironment() override {}
};

// Stand-in whose testPhenotype() calls a fitness helper with a field name
// that doesn't exist in its own topology, used to verify that the shared
// null-field guard (Solution::getNeuralFieldOrThrow) raises an indicative
// error instead of silently treating a configuration bug as a fitness score.
class MissingFieldSolution final : public Solution
{
public:
    explicit MissingFieldSolution(const SolutionTopology& topology)
        : Solution(topology)
    {
        name = "MissingField";
    }

    MissingFieldSolution(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype)
        : Solution(initialTopology, phenotype)
    {
        name = "MissingField";
    }

    SolutionPtr clone() const override
    {
        MissingFieldSolution solution(initialTopology);
        return std::make_shared<MissingFieldSolution>(solution);
    }

    SolutionPtr copy() const override
    {
        MissingFieldSolution solution(initialTopology, phenotype);
        return std::make_shared<MissingFieldSolution>(solution);
    }

private:
    void testPhenotype() override
    {
        initSimulation();
        parameters.fitness = closenessToRestingLevel("this field does not exist");
    }

    void createPhenotypeEnvironment() override {}
};

// Stand-in whose testPhenotype() calls a fitness helper with the name of a
// real phenotype element that exists but isn't a NeuralField (a GaussStimulus),
// used to verify Solution::getNeuralFieldOrThrow rejects a wrong-type element
// the same way it rejects a missing one, rather than only guarding against
// phenotype.getElement() returning null.
class WrongElementTypeSolution final : public Solution
{
public:
    explicit WrongElementTypeSolution(const SolutionTopology& topology)
        : Solution(topology)
    {
        name = "WrongElementType";
    }

    WrongElementTypeSolution(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype)
        : Solution(initialTopology, phenotype)
    {
        name = "WrongElementType";
    }

    SolutionPtr clone() const override
    {
        WrongElementTypeSolution solution(initialTopology);
        return std::make_shared<WrongElementTypeSolution>(solution);
    }

    SolutionPtr copy() const override
    {
        WrongElementTypeSolution solution(initialTopology, phenotype);
        return std::make_shared<WrongElementTypeSolution>(solution);
    }

private:
    void testPhenotype() override
    {
        initSimulation();
        addGaussianStimulus("nf 1",
            dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0,
                GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
            dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });

        parameters.fitness = closenessToRestingLevel("gs nf 1 50");
    }

    void createPhenotypeEnvironment() override {}
};

// Stand-in whose testPhenotype() calls twoBumpsAtPositionWithAmplitudeAndWidth
// with a field name that doesn't exist, used to verify the shared null-field
// guard (Solution::getNeuralFieldOrThrow) also covers the multi-bump fitness
// primitives named in issue #56, not just closenessToRestingLevel.
class MissingFieldTwoBumpsSolution final : public Solution
{
public:
    explicit MissingFieldTwoBumpsSolution(const SolutionTopology& topology)
        : Solution(topology)
    {
        name = "MissingFieldTwoBumps";
    }

    MissingFieldTwoBumpsSolution(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype)
        : Solution(initialTopology, phenotype)
    {
        name = "MissingFieldTwoBumps";
    }

    SolutionPtr clone() const override
    {
        MissingFieldTwoBumpsSolution solution(initialTopology);
        return std::make_shared<MissingFieldTwoBumpsSolution>(solution);
    }

    SolutionPtr copy() const override
    {
        MissingFieldTwoBumpsSolution solution(initialTopology, phenotype);
        return std::make_shared<MissingFieldTwoBumpsSolution>(solution);
    }

private:
    void testPhenotype() override
    {
        initSimulation();
        parameters.fitness = twoBumpsAtPositionWithAmplitudeAndWidth(
            "this field does not exist", 30.0, 10.0, 10.0, 70.0, 10.0, 10.0);
    }

    void createPhenotypeEnvironment() override {}
};

// Stand-in whose testPhenotype() calls preShapednessAtPosition with a position
// exactly at the field's upper spatial bound (DimensionConstants::xSize), one
// past the last valid activation-component index. Used to verify the
// position-to-index conversion is bounded rather than reading past the end of
// the component vector (issue #56).
class BoundaryPositionPreShapednessSolution final : public Solution
{
public:
    explicit BoundaryPositionPreShapednessSolution(const SolutionTopology& topology)
        : Solution(topology)
    {
        name = "BoundaryPositionPreShapedness";
    }

    BoundaryPositionPreShapednessSolution(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype)
        : Solution(initialTopology, phenotype)
    {
        name = "BoundaryPositionPreShapedness";
    }

    SolutionPtr clone() const override
    {
        BoundaryPositionPreShapednessSolution solution(initialTopology);
        return std::make_shared<BoundaryPositionPreShapednessSolution>(solution);
    }

    SolutionPtr copy() const override
    {
        BoundaryPositionPreShapednessSolution solution(initialTopology, phenotype);
        return std::make_shared<BoundaryPositionPreShapednessSolution>(solution);
    }

private:
    void testPhenotype() override
    {
        initSimulation();
        parameters.fitness = preShapednessAtPosition("nf 1", static_cast<double>(DimensionConstants::xSize));
    }

    void createPhenotypeEnvironment() override {}
};

// Stand-in whose testPhenotype() calls twoBumpsAtPositionWithAmplitudeAndWidth
// against a field that never receives a stimulus, so it forms zero bumps.
// Used to verify the empty-field guard (issue #53): an inactive field must
// not be credited as though it contained a bump.
class EmptyFieldTwoBumpsSolution final : public Solution
{
public:
    explicit EmptyFieldTwoBumpsSolution(const SolutionTopology& topology)
        : Solution(topology)
    {
        name = "EmptyFieldTwoBumps";
    }

    EmptyFieldTwoBumpsSolution(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype)
        : Solution(initialTopology, phenotype)
    {
        name = "EmptyFieldTwoBumps";
    }

    SolutionPtr clone() const override
    {
        EmptyFieldTwoBumpsSolution solution(initialTopology);
        return std::make_shared<EmptyFieldTwoBumpsSolution>(solution);
    }

    SolutionPtr copy() const override
    {
        EmptyFieldTwoBumpsSolution solution(initialTopology, phenotype);
        return std::make_shared<EmptyFieldTwoBumpsSolution>(solution);
    }

private:
    void testPhenotype() override
    {
        initSimulation();
        parameters.fitness = twoBumpsAtPositionWithAmplitudeAndWidth("nf 1", 30.0, 10.0, 10.0, 70.0, 10.0, 10.0);
    }

    void createPhenotypeEnvironment() override {}
};

// Same as EmptyFieldTwoBumpsSolution, but for the three-bump helper.
class EmptyFieldThreeBumpsSolution final : public Solution
{
public:
    explicit EmptyFieldThreeBumpsSolution(const SolutionTopology& topology)
        : Solution(topology)
    {
        name = "EmptyFieldThreeBumps";
    }

    EmptyFieldThreeBumpsSolution(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype)
        : Solution(initialTopology, phenotype)
    {
        name = "EmptyFieldThreeBumps";
    }

    SolutionPtr clone() const override
    {
        EmptyFieldThreeBumpsSolution solution(initialTopology);
        return std::make_shared<EmptyFieldThreeBumpsSolution>(solution);
    }

    SolutionPtr copy() const override
    {
        EmptyFieldThreeBumpsSolution solution(initialTopology, phenotype);
        return std::make_shared<EmptyFieldThreeBumpsSolution>(solution);
    }

private:
    void testPhenotype() override
    {
        initSimulation();
        parameters.fitness = threeBumpsAtPositionWithAmplitudeAndWidth(
            "nf 1", 20.0, 10.0, 10.0, 50.0, 10.0, 10.0, 80.0, 10.0, 10.0);
    }

    void createPhenotypeEnvironment() override {}
};

// Stand-in that drives a single field with one Gaussian stimulus so exactly
// one real bump forms, then queries twoBumpsAtPositionWithAmplitudeAndWidth
// with position1 == position2 (both targeting that same bump). Used to prove
// the injective bump-matching fix (issue #53): the one real bump must not be
// credited against both target slots at once. The bumps observed just before
// the fitness call are captured in the public `observedBumps` member so the
// test can derive the expected fitness independently -- evaluate() clears the
// phenotype on return, so the live NeuralField can't be queried afterwards.
//
// Its genes are seeded explicitly with makeFixedFieldGene() rather than left to
// Solution::initialize(), which would randomize tau, restingLevel, the kernel
// type and the kernel's parameters -- and roughly 2% of those draws produce a
// field that never forms a bump, which is what made this fixture's test flaky
// in CI. Solution::initialize() no-ops on a non-empty genome, so seeding here
// bypasses createInputGenes()/createOutputGenes() entirely and callers can
// still call initialize() as usual.
class SingleBumpTwoBumpsSolution final : public Solution
{
public:
    explicit SingleBumpTwoBumpsSolution(const SolutionTopology& topology)
        : Solution(topology)
    {
        name = "SingleBumpTwoBumps";
        seedFixedGenes();
    }

    SingleBumpTwoBumpsSolution(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype)
        : Solution(initialTopology, phenotype)
    {
        name = "SingleBumpTwoBumps";
        seedFixedGenes();
    }

    SolutionPtr clone() const override
    {
        SingleBumpTwoBumpsSolution solution(initialTopology);
        return std::make_shared<SingleBumpTwoBumpsSolution>(solution);
    }

    SolutionPtr copy() const override
    {
        SingleBumpTwoBumpsSolution solution(initialTopology, phenotype);
        return std::make_shared<SingleBumpTwoBumpsSolution>(solution);
    }

    std::vector<dnf_composer::element::NeuralFieldBump> observedBumps;

    static constexpr double targetPosition = 50.0;
    static constexpr double targetAmplitude = 3.0;
    static constexpr double targetWidth = 3.0;

private:
    // Seeds one INPUT ("nf 1") and one OUTPUT ("nf 2") gene with fixed field and
    // kernel parameters, so the field this fixture drives is identical on every
    // construction. Guarded on isEmpty() because the phenotype-taking
    // constructor is used by copy(), where the genome may already be populated.
    void seedFixedGenes()
    {
        if (!genome.isEmpty())
        {
            return;
        }
        addFieldGene(makeFixedFieldGene(FieldGeneType::INPUT, 1));
        addFieldGene(makeFixedFieldGene(FieldGeneType::OUTPUT, 2));
    }

    void testPhenotype() override
    {
        using namespace dnf_composer::element;

        initSimulation();
        // The field's own parameters are pinned (see seedFixedGenes), so this
        // stimulus drives a bump that forms well clear of the detection
        // threshold. That margin is what makes the fixture reliable: the
        // simulation stays stochastic -- every field carries a NormalNoise
        // whose RNG lives in dnf_composer with no seed hook -- and bump
        // detection is a threshold, so a marginal bump would still flip on
        // noise alone even with the field pinned.
        //
        // These values were briefly raised to GaussStimulusConstants (20.0) to
        // paper over the random-field flake; with the field pinned they are
        // back to their original 5.0/15.0 and measure 2000/2000, so the margin
        // now comes from the fixed kernel rather than from an inflated stimulus.
        addGaussianStimulus("nf 1",
            GaussStimulusParameters{ 5.0, 15.0, targetPosition, true, false },
            ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
        runSimulation(SimulationConstants::maxSimulationSteps);

        const auto neuralField = std::dynamic_pointer_cast<NeuralField>(phenotype.getElement("nf 1"));
        observedBumps = neuralField->getBumps();

        parameters.fitness = twoBumpsAtPositionWithAmplitudeAndWidth("nf 1",
            targetPosition, targetAmplitude, targetWidth,
            targetPosition, targetAmplitude, targetWidth);
    }

    void createPhenotypeEnvironment() override {}
};

// Same idea as EmptyFieldTwoBumpsSolution, but for oneBumpAtPositionWithAmplitudeAndWidth:
// a field that never receives a stimulus forms zero bumps, so the fitness must be
// the same zero baseline the multi-bump helpers already guard for (issue #68).
class EmptyFieldOneBumpSolution final : public Solution
{
public:
    explicit EmptyFieldOneBumpSolution(const SolutionTopology& topology)
        : Solution(topology)
    {
        name = "EmptyFieldOneBump";
    }

    EmptyFieldOneBumpSolution(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype)
        : Solution(initialTopology, phenotype)
    {
        name = "EmptyFieldOneBump";
    }

    SolutionPtr clone() const override
    {
        EmptyFieldOneBumpSolution solution(initialTopology);
        return std::make_shared<EmptyFieldOneBumpSolution>(solution);
    }

    SolutionPtr copy() const override
    {
        EmptyFieldOneBumpSolution solution(initialTopology, phenotype);
        return std::make_shared<EmptyFieldOneBumpSolution>(solution);
    }

private:
    void testPhenotype() override
    {
        initSimulation();
        parameters.fitness = oneBumpAtPositionWithAmplitudeAndWidth("nf 1", 50.0, 10.0, 10.0);
    }

    void createPhenotypeEnvironment() override {}
};

// Same guard as MissingFieldTwoBumpsSolution, checked directly against
// oneBumpAtPositionWithAmplitudeAndWidth -- the single-bump fitness primitive
// named first in issue #68's location list.
class MissingFieldOneBumpSolution final : public Solution
{
public:
    explicit MissingFieldOneBumpSolution(const SolutionTopology& topology)
        : Solution(topology)
    {
        name = "MissingFieldOneBump";
    }

    MissingFieldOneBumpSolution(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype)
        : Solution(initialTopology, phenotype)
    {
        name = "MissingFieldOneBump";
    }

    SolutionPtr clone() const override
    {
        MissingFieldOneBumpSolution solution(initialTopology);
        return std::make_shared<MissingFieldOneBumpSolution>(solution);
    }

    SolutionPtr copy() const override
    {
        MissingFieldOneBumpSolution solution(initialTopology, phenotype);
        return std::make_shared<MissingFieldOneBumpSolution>(solution);
    }

private:
    void testPhenotype() override
    {
        initSimulation();
        parameters.fitness = oneBumpAtPositionWithAmplitudeAndWidth("this field does not exist", 50.0, 10.0, 10.0);
    }

    void createPhenotypeEnvironment() override {}
};

// Stand-in whose testPhenotype() drives a single field with one Gaussian
// stimulus so exactly one real bump forms, then queries
// oneBumpAtPositionWithAmplitudeAndWidth with position/amplitude/width equal
// to that bump's own observed values. With the bump count exactly matching
// the target (1) and all three distance terms exactly zero, every weighted
// term is credited in full -- this is the theoretical maximum fitness (1.0),
// so it directly exercises the "exact match scores near the theoretical max"
// case from issue #68 without depending on how closely a Gaussian stimulus's
// parameters translate into the field's actual bump shape. The bumps observed
// just before the fitness call are captured in `observedBumps` so the test
// can derive the target independently -- evaluate() clears the phenotype on
// return, so the live NeuralField can't be queried afterwards.
class SingleBumpOneBumpSolution final : public Solution
{
public:
    explicit SingleBumpOneBumpSolution(const SolutionTopology& topology)
        : Solution(topology)
    {
        name = "SingleBumpOneBump";
    }

    SingleBumpOneBumpSolution(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype)
        : Solution(initialTopology, phenotype)
    {
        name = "SingleBumpOneBump";
    }

    SolutionPtr clone() const override
    {
        SingleBumpOneBumpSolution solution(initialTopology);
        return std::make_shared<SingleBumpOneBumpSolution>(solution);
    }

    SolutionPtr copy() const override
    {
        SingleBumpOneBumpSolution solution(initialTopology, phenotype);
        return std::make_shared<SingleBumpOneBumpSolution>(solution);
    }

    std::vector<dnf_composer::element::NeuralFieldBump> observedBumps;

private:
    void testPhenotype() override
    {
        using namespace dnf_composer::element;

        initSimulation();
        addGaussianStimulus("nf 1",
            GaussStimulusParameters{ 5.0, 15.0, 50.0, true, false },
            ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
        runSimulation(SimulationConstants::maxSimulationSteps);

        const auto neuralField = std::dynamic_pointer_cast<NeuralField>(phenotype.getElement("nf 1"));
        observedBumps = neuralField->getBumps();

        const auto& bump = observedBumps.front();
        parameters.fitness = oneBumpAtPositionWithAmplitudeAndWidth("nf 1",
            bump.centroid, bump.amplitude, bump.width);
    }

    void createPhenotypeEnvironment() override {}
};

// Same guard as MissingFieldTwoBumpsSolution, checked directly against
// threeBumpsAtPositionWithAmplitudeAndWidth.
class MissingFieldThreeBumpsSolution final : public Solution
{
public:
    explicit MissingFieldThreeBumpsSolution(const SolutionTopology& topology)
        : Solution(topology)
    {
        name = "MissingFieldThreeBumps";
    }

    MissingFieldThreeBumpsSolution(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype)
        : Solution(initialTopology, phenotype)
    {
        name = "MissingFieldThreeBumps";
    }

    SolutionPtr clone() const override
    {
        MissingFieldThreeBumpsSolution solution(initialTopology);
        return std::make_shared<MissingFieldThreeBumpsSolution>(solution);
    }

    SolutionPtr copy() const override
    {
        MissingFieldThreeBumpsSolution solution(initialTopology, phenotype);
        return std::make_shared<MissingFieldThreeBumpsSolution>(solution);
    }

private:
    void testPhenotype() override
    {
        initSimulation();
        parameters.fitness = threeBumpsAtPositionWithAmplitudeAndWidth(
            "this field does not exist",
            20.0, 10.0, 10.0, 50.0, 10.0, 10.0, 80.0, 10.0, 10.0);
    }

    void createPhenotypeEnvironment() override {}
};

// Same idea as SingleBumpTwoBumpsSolution, but for the three-bump helper: one
// real bump is queried against three target slots that all point at that same
// bump, proving matchClosestBump's injective consumption (issue #53) also
// prevents a single bump from being triple-counted, not just double-counted.
// Targets equal the bump's own observed values so every matched distance term
// is exactly zero, making the expected fitness computable independently of
// simulation jitter.
class SingleBumpThreeBumpsSolution final : public Solution
{
public:
    explicit SingleBumpThreeBumpsSolution(const SolutionTopology& topology)
        : Solution(topology)
    {
        name = "SingleBumpThreeBumps";
    }

    SingleBumpThreeBumpsSolution(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype)
        : Solution(initialTopology, phenotype)
    {
        name = "SingleBumpThreeBumps";
    }

    SolutionPtr clone() const override
    {
        SingleBumpThreeBumpsSolution solution(initialTopology);
        return std::make_shared<SingleBumpThreeBumpsSolution>(solution);
    }

    SolutionPtr copy() const override
    {
        SingleBumpThreeBumpsSolution solution(initialTopology, phenotype);
        return std::make_shared<SingleBumpThreeBumpsSolution>(solution);
    }

    std::vector<dnf_composer::element::NeuralFieldBump> observedBumps;

private:
    void testPhenotype() override
    {
        using namespace dnf_composer::element;

        initSimulation();
        addGaussianStimulus("nf 1",
            GaussStimulusParameters{ 5.0, 15.0, 50.0, true, false },
            ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
        runSimulation(SimulationConstants::maxSimulationSteps);

        const auto neuralField = std::dynamic_pointer_cast<NeuralField>(phenotype.getElement("nf 1"));
        observedBumps = neuralField->getBumps();

        const auto& bump = observedBumps.front();
        parameters.fitness = threeBumpsAtPositionWithAmplitudeAndWidth("nf 1",
            bump.centroid, bump.amplitude, bump.width,
            bump.centroid, bump.amplitude, bump.width,
            bump.centroid, bump.amplitude, bump.width);
    }

    void createPhenotypeEnvironment() override {}
};

// Same guard as MissingFieldSolution, checked directly against
// preShapednessAtPosition rather than closenessToRestingLevel.
class MissingFieldPreShapednessSolution final : public Solution
{
public:
    explicit MissingFieldPreShapednessSolution(const SolutionTopology& topology)
        : Solution(topology)
    {
        name = "MissingFieldPreShapedness";
    }

    MissingFieldPreShapednessSolution(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype)
        : Solution(initialTopology, phenotype)
    {
        name = "MissingFieldPreShapedness";
    }

    SolutionPtr clone() const override
    {
        MissingFieldPreShapednessSolution solution(initialTopology);
        return std::make_shared<MissingFieldPreShapednessSolution>(solution);
    }

    SolutionPtr copy() const override
    {
        MissingFieldPreShapednessSolution solution(initialTopology, phenotype);
        return std::make_shared<MissingFieldPreShapednessSolution>(solution);
    }

private:
    void testPhenotype() override
    {
        initSimulation();
        parameters.fitness = preShapednessAtPosition("this field does not exist", 50.0);
    }

    void createPhenotypeEnvironment() override {}
};

// Stand-in that queries preShapednessAtPosition against a field that never
// receives a stimulus, so activation everywhere sits exactly at the field's
// startingRestingLevel. preShapednessAtPosition requires activation strictly
// above resting level (u <= h + epsilon returns 0.0), so a field at rest must
// score exactly 0.0 -- the preShapedness analogue of the zero-bump case the
// multi-bump helpers are already guarded for.
class RestingLevelPreShapednessSolution final : public Solution
{
public:
    explicit RestingLevelPreShapednessSolution(const SolutionTopology& topology)
        : Solution(topology)
    {
        name = "RestingLevelPreShapedness";
    }

    RestingLevelPreShapednessSolution(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype)
        : Solution(initialTopology, phenotype)
    {
        name = "RestingLevelPreShapedness";
    }

    SolutionPtr clone() const override
    {
        RestingLevelPreShapednessSolution solution(initialTopology);
        return std::make_shared<RestingLevelPreShapednessSolution>(solution);
    }

    SolutionPtr copy() const override
    {
        RestingLevelPreShapednessSolution solution(initialTopology, phenotype);
        return std::make_shared<RestingLevelPreShapednessSolution>(solution);
    }

private:
    void testPhenotype() override
    {
        initSimulation();
        parameters.fitness = preShapednessAtPosition("nf 1", 50.0);
    }

    void createPhenotypeEnvironment() override {}
};

// Same guard as MissingFieldSolution, checked directly against
// negativePreShapednessAtPosition.
class MissingFieldNegativePreShapednessSolution final : public Solution
{
public:
    explicit MissingFieldNegativePreShapednessSolution(const SolutionTopology& topology)
        : Solution(topology)
    {
        name = "MissingFieldNegativePreShapedness";
    }

    MissingFieldNegativePreShapednessSolution(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype)
        : Solution(initialTopology, phenotype)
    {
        name = "MissingFieldNegativePreShapedness";
    }

    SolutionPtr clone() const override
    {
        MissingFieldNegativePreShapednessSolution solution(initialTopology);
        return std::make_shared<MissingFieldNegativePreShapednessSolution>(solution);
    }

    SolutionPtr copy() const override
    {
        MissingFieldNegativePreShapednessSolution solution(initialTopology, phenotype);
        return std::make_shared<MissingFieldNegativePreShapednessSolution>(solution);
    }

private:
    void testPhenotype() override
    {
        initSimulation();
        parameters.fitness = negativePreShapednessAtPosition("this field does not exist", 50.0);
    }

    void createPhenotypeEnvironment() override {}
};

// Analogue of RestingLevelPreShapednessSolution for negativePreShapednessAtPosition:
// with no stimulus, activation at the queried position sits exactly at
// startingRestingLevel, which fails the "activation must be lower than resting
// level minus epsilon" guard (u_pos >= restingLevel - epsilon returns 0.0), so
// the score must be exactly 0.0.
class RestingLevelNegativePreShapednessSolution final : public Solution
{
public:
    explicit RestingLevelNegativePreShapednessSolution(const SolutionTopology& topology)
        : Solution(topology)
    {
        name = "RestingLevelNegativePreShapedness";
    }

    RestingLevelNegativePreShapednessSolution(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype)
        : Solution(initialTopology, phenotype)
    {
        name = "RestingLevelNegativePreShapedness";
    }

    SolutionPtr clone() const override
    {
        RestingLevelNegativePreShapednessSolution solution(initialTopology);
        return std::make_shared<RestingLevelNegativePreShapednessSolution>(solution);
    }

    SolutionPtr copy() const override
    {
        RestingLevelNegativePreShapednessSolution solution(initialTopology, phenotype);
        return std::make_shared<RestingLevelNegativePreShapednessSolution>(solution);
    }

private:
    void testPhenotype() override
    {
        initSimulation();
        parameters.fitness = negativePreShapednessAtPosition("nf 1", 50.0);
    }

    void createPhenotypeEnvironment() override {}
};

// Same boundary case as BoundaryPositionPreShapednessSolution, but for
// negativePreShapednessAtPosition -- it too resolves position to an index via
// the shared clampedIndexForPosition helper (issue #56), so the upper spatial
// bound (DimensionConstants::xSize) must not read past the end of the
// activation component either.
class BoundaryPositionNegativePreShapednessSolution final : public Solution
{
public:
    explicit BoundaryPositionNegativePreShapednessSolution(const SolutionTopology& topology)
        : Solution(topology)
    {
        name = "BoundaryPositionNegativePreShapedness";
    }

    BoundaryPositionNegativePreShapednessSolution(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype)
        : Solution(initialTopology, phenotype)
    {
        name = "BoundaryPositionNegativePreShapedness";
    }

    SolutionPtr clone() const override
    {
        BoundaryPositionNegativePreShapednessSolution solution(initialTopology);
        return std::make_shared<BoundaryPositionNegativePreShapednessSolution>(solution);
    }

    SolutionPtr copy() const override
    {
        BoundaryPositionNegativePreShapednessSolution solution(initialTopology, phenotype);
        return std::make_shared<BoundaryPositionNegativePreShapednessSolution>(solution);
    }

private:
    void testPhenotype() override
    {
        initSimulation();
        parameters.fitness = negativePreShapednessAtPosition("nf 1", static_cast<double>(DimensionConstants::xSize));
    }

    void createPhenotypeEnvironment() override {}
};

} // namespace neat_dnfs::test
