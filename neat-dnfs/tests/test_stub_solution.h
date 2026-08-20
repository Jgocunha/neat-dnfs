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

} // namespace neat_dnfs::test
