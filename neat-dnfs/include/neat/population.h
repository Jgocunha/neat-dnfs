#pragma once

#include <future>

#include <dnf_composer/tools/logger.h>
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

		explicit PopulationParameters(int size = 100, int numGenerations = 1000, double targetFitness = 0.95);
	};

	struct PopulationControl
	{
		bool pause;
		bool stop;

		explicit PopulationControl(bool pause = false, bool stop = false);
	};

	struct PopulationStatistics
	{
		std::chrono::time_point<std::chrono::steady_clock> start;
		std::chrono::time_point<std::chrono::steady_clock> end;
		long long duration{};
		
		PopulationStatistics() = default;
	};

	struct PerGenerationStatistics
	{
		double averageFitness = 0.0f;
		// double stdDevFitness = 0.0f;
		double bestFitness = 0.0f;
		int numberOfSpecies = 0;
		int numberOfActiveSpecies = 0;
		// double averageCompatibilityDistance;
		int innovationNumber = 0;
		double averageGenomeSize = 0.0f;
		double averageConnectionGenes = 0.0f;
		double averageFieldGenes = 0.0f;

		PerGenerationStatistics() = default;
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
		PerGenerationStatistics perGenStatistics;
		bool hasFitnessImproved{};
		int generationsWithoutImprovement = 0;
		std::string fileDirectory;
	public:
		Population(const PopulationParameters& parameters, 
			const SolutionPtr& initialSolution);
		~Population();

		void initialize() const;
		void evolve();

		[[nodiscard]] SolutionPtr getBestSolution() const { return bestSolution; }
		std::vector<std::shared_ptr<Species>> getSpeciesList() { return speciesList; }
		[[nodiscard]] std::vector<SolutionPtr> getSolutions() const { return solutions; }
		[[nodiscard]] int getSize() const { return parameters.size; }
		[[nodiscard]] int getCurrentGeneration() const { return parameters.currentGeneration; }
		[[nodiscard]] int getNumGenerations() const { return parameters.numGenerations; }
		[[nodiscard]] bool isInitialized() const { return !solutions.empty(); }

		void setSize(const int size) { parameters.size = size; }
		void setNumGenerations(const int numGenerations) { parameters.numGenerations = numGenerations; }

		void pause() { control.pause = true; }
		void resume() { control.pause = false; }
		void stop() { control.stop = true; }
		void start() { control.stop = false; }
	private:
		void evaluate() const;
		void speciate();
		void reproduceAndSelect();

		bool endConditionMet() const;

		void startup();
		void upkeep();
		void cleanup();
		void createInitialSolutions(const SolutionPtr& initialSolution);
		void buildInitialSolutionsGenome() const;

		void assignToSpecies(const SolutionPtr& solution);
		std::shared_ptr<Species> findSpecies(const SolutionPtr& solution);
		[[nodiscard]] std::shared_ptr<Species> getBestActiveSpecies() const;

		void calculateAdjustedFitness();
		void setSpeciesAsExtinct() const;
		void assignOffspringToSpecies();
		void clearSpeciesOffspring() const;
		bool hasFitnessImprovedOverTheLastGenerations();
		void assignOffspringToTopTwoSpecies();
		void sortSpeciesListByChampionFitness();
		void assignOffspringBasedOnAdjustedFitness() const;
		void reassignOffspringIfFitnessIsStagnant() const;

		void pruneWorsePreformingSolutions() const;
		void replaceEntirePopulationWithOffspring();
		void mutate();

		void upkeepBestSolution();
		void upkeepChampions();
		void upkeepPerGenerationStatistics();
		void updateGenerationAndAges();
		void validateElitism() const;
		void validateUniqueSolutions() const;
		void validatePopulationSize() const;
		void validateUniqueGenesInGenomes() const;
		void validateUniqueKernelAndNeuralFieldPtrs() const;
		void validateIfSpeciesHaveUniqueRepresentative() const;
		void validateAssignmentIntoSpecies() const;

		void setFileDirectory();
		void print() const;
		void saveAllSolutionsWithFitnessAbove(double fitness) const;
		void saveChampions() const;
		void saveTimestampsAndDuration() const;
		void saveAllSolutionsPerGeneration() const;
		void savePerGenerationOverview() const;
		void saveBestSolutionOfEachGeneration() const;
		void saveChampionsOfEachGeneration() const;
		void savePerGenerationStatistics() const;
		void savePerGenerationSpecies() const;

		static void resetGenerationalInnovations();
		void clearLastMutations() const;

		void logSolutions() const;
		void logSpecies() const;
		void logOverview() const;

		void startKeyListenerForUserCommands();
	};
}