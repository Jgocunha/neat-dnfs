#pragma once

#include "solution.h"
#include "species.h"

namespace neat_dnfs
{
	struct PopulationParameters
	{
		uint16_t size;
		uint16_t currentGeneration;
		uint16_t numGenerations;
		double targetFitness;

		PopulationParameters(uint16_t size = 100,
			uint16_t numGenerations = 1000,
			double targetFitness = 0.95);
	};
	
	class Population
	{
	private:
		PopulationParameters parameters;
		std::vector<Species> speciesList;
		SolutionPtr bestSolution;
	public:
		Population(const PopulationParameters& parameters, 
			const SolutionPtr& initialSolution);

		void initialize() const;
		void evolve();
		SolutionPtr getBestSolution() const { return bestSolution; }
	private:
		void createInitialEmptySolutions(const SolutionPtr& initialSolution);
		void evaluate() const;
		void speciate();
		void reproduce();
		void assignToSpecies(const SolutionPtr& solution);
		Species* findSpeciesOfSolution(const SolutionPtr& solution);
		void calculateAdjustedFitness() const;
		void calculateReproductionProbabilities() const;
		uint16_t getNumSolutionsToKill() const;
		SolutionPtr crossover() const;
		uint16_t size() const;
		void trackBestSolution();
		void trackGenerationalInfo();
		void logGenerationalInfo() const;
		bool endConditionMet() const;
	};
}
