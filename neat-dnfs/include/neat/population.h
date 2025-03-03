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

	struct PopulationStatistics
	{
		std::chrono::time_point<std::chrono::steady_clock> start;
		std::chrono::time_point<std::chrono::steady_clock> end;
		long long duration;

		PopulationStatistics() = default;

	};


	class Population
	{
	private:
		PopulationParameters parameters;
		std::vector<SolutionPtr> solutions;
		std::vector<std::shared_ptr<Species>> speciesList;
		SolutionPtr bestSolution;
		std::vector<SolutionPtr> champions;
		PopulationControl control;
		PopulationStatistics statistics;
		bool hasFitnessImproved;
		int generationsWithoutImprovement = 0;
		std::string fileDirectory;
	public:
		Population(const PopulationParameters& parameters, 
			const SolutionPtr& initialSolution);
		void initialize() const;
		void evolve();

		SolutionPtr getBestSolution() const { return bestSolution; }
		std::vector<std::shared_ptr<Species>> getSpeciesList() { return speciesList; }
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
		std::shared_ptr<Species> findSpecies(const SolutionPtr& solution);
		std::shared_ptr<Species> getBestActiveSpecies();

		void calculateAdjustedFitness();
		void assignOffspringToSpecies();
		void clearSpeciesOffspring() const;
		bool hasFitnessImprovedOverTheLastGenerations();
		void assignOffspringToTopTwoSpecies();
		void sortSpeciesListByChampionFitness();
		void assignOffspringBasedOnAdjustedFitness() const;
		void reassignOffspringIfFitnessIsStagnant();

		void pruneWorsePreformingSolutions() const;
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

		void setFileDirectory();
		void print() const;
		void saveAllSolutionsWithFitnessAbove(double fitness) const;
		void saveChampions() const;
		void saveTimestampsAndDuration() const;
		void saveFinalStatistics() const;
		void savePerGenerationStatistics() const;
		void saveBestSolutionOfEachGeneration() const;
		void saveChampionsOfEachGeneration() const;

		void resetGenerationalInnovations() const;
		void resetMutationStatisticsPerGeneration() const;

		void logSolutions() const;
		void logSpecies() const;
		void logOverview() const;
		void logMutationStatistics() const;

		void startKeyListenerForUserCommands();
	};
}