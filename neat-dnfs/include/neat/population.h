#pragma once

#include <future>

#include <dnf_composer/simulation/simulation_file_manager.h>

#include "solution.h"
#include "species.h"

namespace neat_dnfs
{
	struct PopulationParameters
	{
		int size;
		int currentGeneration;
		int numGenerations;
		double targetFitness;

		PopulationParameters(int size = 100, int numGenerations = 1000, double targetFitness = 0.95);
	};

	struct PopulationControl
	{
		bool pause;
		bool stop;

		PopulationControl(bool pause = false, bool stop = false);
	};

	class Population
	{
	private:
		PopulationParameters parameters;
		std::vector<SolutionPtr> solutions;
		std::vector<Species> speciesList;
		SolutionPtr bestSolution;
		std::vector<SolutionPtr> champions;
		PopulationControl control;
		bool hasFitnessImproved;
		int generationsWithoutImprovement = 0;
	public:
		Population(const PopulationParameters& parameters, 
			const SolutionPtr& initialSolution);
		void initialize() const;
		void evolve();

		SolutionPtr getBestSolution() const { return bestSolution; }
		std::vector<Species>& getSpeciesList() { return speciesList; }
		std::vector<SolutionPtr> getSolutions() const { return solutions; }
		int getSize() const { return parameters.size; }
		int getCurrentGeneration() const { return parameters.currentGeneration; }
		int getNumGenerations() const { return parameters.numGenerations; }
		bool isInitialized() const { return !solutions.empty(); }

		void setSize(int size) { parameters.size = size; }
		void setNumGenerations(int numGenerations) { parameters.numGenerations = numGenerations; }

		void pause() { control.pause = true; }
		void resume() { control.pause = false; }
		void stop() { control.stop = true; }
		void start() { control.stop = false; }
	private:
		void evaluate() const;
		void speciate();
		void reproduceAndSelect();

		bool endConditionMet() const;

		void upkeep();
		void createInitialEmptySolutions(const SolutionPtr& initialSolution);
		void buildInitialSolutionsGenome() const;

		void assignToSpecies(const SolutionPtr& solution);
		Species* findSpecies(const SolutionPtr& solution);
		Species* getBestActiveSpecies();

		void calculateAdjustedFitness();
		void assignOffspringToSpecies();
		void clearSpeciesOffspring();
		bool hasFitnessImprovedOverTheLastGenerations();
		void assignOffspringToTopTwoSpecies();
		void assignOffspringBasedOnAdjustedFitness();
		void reassignOffspringIfFitnessIsStagnant();

		void pruneWorsePreformingSolutions();
		void replaceEntirePopulationWithOffspring();
		void mutate();

		void upkeepBestSolution();
		void upkeepChampions();
		void updateGenerationAndAges();
		void validateElitism() const;
		void validateUniqueSolutions() const;
		void validatePopulationSize() const;
		void validateUniqueGenesInGenomes() const;
		void validateUniqueKernelAndNeuralFieldPtrs() const;
		void validateIfSpeciesHaveUniqueRepresentative() const;

		void print() const;
		void saveAllSolutionsWithFitnessAbove(double fitness) const;

		void resetGenerationalInnovations() const;
		void resetMutationStatisticsPerGeneration() const;

		void logSolutions() const;
		void logSpecies() const;
		void logOverview() const;
		void logMutationStatistics() const;

		void startKeyListenerForUserCommands();
	};
}