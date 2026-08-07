#pragma once
#include <memory>
#include <string>
#include <vector>

#include "neat/population.h"
#include "test_helpers.h"

namespace neat_dnfs::test {

struct EvolutionRunSettings
{
    int populationSize = 50;
    int numGenerations = 10;
    double targetFitness = 0.75; // a stop condition, not an expected outcome -- see runEvolution()
    int numRuns = 5;
    bool parallel = true;
};

struct EvolutionRunResult
{
    double initialBestFitness = 0.0;
    double finalBestFitness = 0.0;
    int generationsRun = 0;
    int finalPopulationSize = 0;
    int validationViolations = 0;
    std::vector<std::string> validationMessages;
    std::vector<double> bestFitnessHistory;
    int numGenerations = 0;
    double targetFitness = 0.0;
};

// Runs settings.numRuns independent Population::evolve() calls for SolutionType
// over topology, each starting from resetGlobalState() so global id/innovation
// counters don't drift across runs. Population is non-copyable and non-movable,
// so it is constructed in place inside the loop; only the plain-data result
// escapes.
//
// targetFitness is Population's early-stop condition (endConditionMet()), not
// something these tests expect to reach at populationSize=50/numGenerations=10 --
// do not turn it into a "reached target" assertion.
//
// These runs are deliberately not seeded, and cannot be made reproducible:
// every FieldGene carries a NormalNoise whose per-step draws come from a
// thread-local engine in an anonymous namespace inside dnf_composer
// (src/tools/math.cpp), with no seed hook, and evaluation is spread across
// std::async workers that each get their own engine. Near a bump-formation
// boundary that noise swings a partial-fitness term by ~0.1 -- the same effect
// PopulationConstants::elitismFitnessEpsilon exists to absorb.
//
// So assert only on properties that hold for *every* realization: invariants,
// bounds, and internal consistency. Do not assert that fitness improved, or
// that any particular fitness is reached -- that is a property of the search
// algorithm on a noisy landscape, not of the code, and it makes the suite
// flaky without detecting anything a deterministic assertion would miss.
template <typename SolutionType>
std::vector<EvolutionRunResult> runEvolution(const SolutionTopology& topology,
    const EvolutionRunSettings& settings = {})
{
    std::vector<EvolutionRunResult> results;
    results.reserve(settings.numRuns);

    for (int run = 0; run < settings.numRuns; ++run)
    {
        resetGlobalState();

        const PopulationParameters parameters(settings.populationSize, settings.numGenerations,
            settings.targetFitness, settings.parallel);
        Population population{ parameters, std::make_shared<SolutionType>(topology), /*enableFileIO=*/false };
        population.initialize();
        population.evolve();

        EvolutionRunResult result;
        const auto& history = population.getBestFitnessHistory();
        result.initialBestFitness = history.empty() ? 0.0 : history.front();
        result.finalBestFitness = population.getBestSolution()->getFitness();
        result.generationsRun = population.getCurrentGeneration();
        result.finalPopulationSize = static_cast<int>(population.getSolutions().size());

        const auto& report = population.getValidationReport();
        result.validationViolations = report.total();
        result.validationMessages = report.messages;
        result.bestFitnessHistory = history;
        result.numGenerations = settings.numGenerations;
        result.targetFitness = settings.targetFitness;

        results.push_back(std::move(result));
    }

    return results;
}

inline std::string join(const std::vector<std::string>& lines, const std::string& separator)
{
    std::string joined;
    for (size_t i = 0; i < lines.size(); ++i)
    {
        if (i != 0)
        {
            joined += separator;
        }
        joined += lines[i];
    }
    return joined;
}

} // namespace neat_dnfs::test
