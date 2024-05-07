#pragma once

#include "solution.h"
#include "species.h"

namespace neat_dnfs
{
	class Population
	{
	private:
		uint16_t size;
		uint16_t currentGeneration;
		std::vector<SolutionPtr> solutions;
		std::vector<Species> speciesList;
		SolutionPtr bestSolution;
	public:
		Population(uint16_t size, const SolutionPtr& initialSolution);
		void initialize() const;
		void evolve(uint16_t maxGeneration);
		void evaluate() const;
		void speciate();
		void select();
		void reproduce() const;
		SolutionPtr getBestSolution() const;
	private:
		void createInitialEmptySolutions(const SolutionPtr& initialSolution);
		void buildInitialSolutionsGenome() const;
		void mutate() const;
		void upkeepBestSolution();
		void updateGenerationAndAges();
		void assignToSpecies(const SolutionPtr& solution);
		Species* findSpecies(const SolutionPtr& solution);
		void calculateAdjustedFitness();
		void calculateSpeciesOffspring();
		void killLeastFitSolutions();
	};
}
