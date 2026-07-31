#pragma once
#include <atomic>
#include <thread>
#include <stdexcept>

#include "neat/solution.h"

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

} // namespace neat_dnfs::test
