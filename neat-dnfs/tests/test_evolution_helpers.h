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
