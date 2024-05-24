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
			double targetFitness = 0.95)
			: size(size), currentGeneration(0),
			numGenerations(numGenerations),
			targetFitness(targetFitness)
		{}
	};

	class Population
	{
	private:
		PopulationParameters parameters;
		std::vector<SolutionPtr> solutions;
		std::vector<Species> speciesList;
		SolutionPtr bestSolution;
	public:
		Population(const PopulationParameters& parameters, const SolutionPtr& initialSolution);
		void initialize() const;
		void evolve();
		void evaluate() const;
		void speciate();
		void select();
		void reproduce();
		void upkeepBestSolution();
		void updateGenerationAndAges();
		bool endConditionMet() const;
		std::vector<Species>& getSpeciesList() { return speciesList; }
		std::vector<SolutionPtr> getSolutions() const { return solutions; }
		SolutionPtr getBestSolution() const;
		void setSize(uint16_t size) { parameters.size = size; }
		void setNumGenerations(uint16_t numGenerations) { parameters.numGenerations = numGenerations; }
		uint16_t getSize() const { return parameters.size; }
		uint16_t getCurrentGeneration() const { return parameters.currentGeneration; }
		uint16_t getNumGenerations() const { return parameters.numGenerations; }
	private:
		void createInitialEmptySolutions(const SolutionPtr& initialSolution);
		void buildInitialSolutionsGenome() const;
		void assignToSpecies(const SolutionPtr& solution);
		Species* findSpecies(const SolutionPtr& solution);
		void calculateAdjustedFitness();
		void crossover();
		void calculateSpeciesOffspring();
		void killLeastFitSolutions();
		void validateElitism() const;
	};
}