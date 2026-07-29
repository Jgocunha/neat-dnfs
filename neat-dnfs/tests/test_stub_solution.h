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

} // namespace neat_dnfs::test
