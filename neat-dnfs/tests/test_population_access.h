#pragma once

#include "neat/population.h"

namespace neat_dnfs::test {

// Grants tests direct, deterministic access to Population's private
// speciation/reproduction internals (issue #59: extinct species were never
// erased from Population::speciesList, and assignToSpecies() re-randomized
// an existing species' representative on every insertion during a single
// assignment pass). Reproducing either defect through the public evolve()
// API alone would mean waiting on stochastic genome compatibility and
// fitness dynamics to happen to trigger them -- this lets the regression
// tests drive the exact internal state instead.
class PopulationTestAccess
{
public:
    static void speciate(Population& population) { population.speciate(); }
    static void reproduceAndSelect(Population& population) { population.reproduceAndSelect(); }
    static std::vector<std::shared_ptr<Species>>& speciesList(Population& population) { return population.speciesList; }
    static void setBestSolution(Population& population, const SolutionPtr& solution) { population.bestSolution = solution; }
};

} // namespace neat_dnfs::test
