#pragma once

#include <array>
#include <atomic>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "genome.h"
#include "solution.h"
#include "species.h"

namespace neat_dnfs
{
	class PopulationFileManager;

	/// @brief Identifies which internal invariant check reported a violation.
	enum class ValidationCheck
	{
		PopulationSize,
		UniqueSolutions,
		Elitism,
		UniqueGenesInGenomes,
		UniqueKernelAndNeuralFieldPtrs,
		SpeciesHaveUniqueRepresentative,
		AssignmentIntoSpecies,
		Count
	};

	/// @brief How a Population reacts to an invariant violation.
	/// Log (the production default) only records/prints it; Throw raises
	/// ValidationError, which is what the test binary opts into via
	/// Population::setDefaultValidationPolicy so violations fail tests loudly
	/// without changing production behaviour.
	enum class ValidationPolicy
	{
		Log,
		Throw
	};

	/// @brief Accumulates invariant-check violations observed during a run.
	/// Messages are capped (not one per violation) since an O(n^2) check at a
	/// large population size can otherwise emit far more strings than anyone
	/// will read; counts remain exact regardless of the cap.
	struct ValidationReport
	{
		static constexpr size_t maxRetainedMessages = 32;

		std::array<int, static_cast<size_t>(ValidationCheck::Count)> counts{};
		std::vector<std::string> messages;

		[[nodiscard]] int total() const;
		[[nodiscard]] int count(ValidationCheck check) const;
		[[nodiscard]] bool clean() const { return total() == 0; }
		void clear();
	};

	/// @brief Thrown by Population::reportViolation when the active
	/// ValidationPolicy is Throw.
	class ValidationError : public std::runtime_error
	{
	public:
		ValidationError(ValidationCheck check, const std::string& message);
		[[nodiscard]] ValidationCheck getCheck() const { return check; }
	private:
		ValidationCheck check;
	};

	/// @brief Configuration for a NEAT population run.
	struct PopulationParameters
	{
		int size; ///< Number of solutions per generation; must be greater than 0.
		int currentGeneration{0};
		int numGenerations;
		double targetFitness; ///< Evolution stops early when the best solution reaches this fitness.
		bool parallelEvolution; ///< Evaluate solutions concurrently via std::async.

		explicit PopulationParameters(int size = 100, int numGenerations = 1000, double targetFitness = 0.95, bool parallelEvolution = true);
	};

	/// @brief Runtime flags for pausing or stopping evolution from an external thread.
	struct PopulationControl
	{
		std::atomic<bool> pause;
		std::atomic<bool> stop;

		explicit PopulationControl(bool pause = false, bool stop = false);
	};

	struct PopulationStatistics
	{
		std::chrono::time_point<std::chrono::steady_clock> start;
		std::chrono::time_point<std::chrono::steady_clock> end;
		long long duration{};

		PopulationStatistics() = default;
	};

	/// @brief Per-generation snapshot of population health metrics.
	struct PerGenerationStatistics
	{
		double averageFitness = 0.0F;
		double bestFitness = 0.0F;
		int numberOfSpecies = 0;
		int numberOfActiveSpecies = 0;
		int innovationNumber = 0;
		double averageGenomeSize = 0.0F;
		double averageConnectionGenes = 0.0F;
		double averageFieldGenes = 0.0F;

		PerGenerationStatistics() = default;
	};

	/// @brief Manages a NEAT population: speciation, evaluation, reproduction, and selection.
	///
	/// Call @c initialize() once, then @c evolve() to run the full evolutionary loop.
	/// Evolution stops when @c PopulationParameters::targetFitness is reached or
	/// @c numGenerations is exhausted. Use @c pause() / @c stop() for interactive control.
	class Population
	{
		friend class PopulationFileManager;
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
		double previousBestFitness = 0.0;
		SolutionPtr previousBestSolution;
		std::vector<double> bestFitnessHistory;
		std::vector<int> bestSolutionIdHistory;
		std::vector<Genome> bestSolutionGenomeHistory;
		std::unique_ptr<PopulationFileManager> fileManager;
		ValidationReport validationReport;
		ValidationPolicy validationPolicy = defaultValidationPolicy;

		// Not thread-safe; only ever called from upkeep()/speciate(), both
		// main-thread. Must not be called from the parallel evaluate() path.
		void reportViolation(ValidationCheck check, const std::string& message);
	public:
		Population(const PopulationParameters& parameters,
			const SolutionPtr& initialSolution,
			bool enableFileIO = true);
		~Population();
		Population(const Population& other) = delete;
		Population(Population&& other) = delete;
		Population& operator=(const Population& other) = delete;
		Population& operator=(Population&& other) = delete;

		void initialize() const;
		void evolve();

		[[nodiscard]] SolutionPtr getBestSolution() const { return bestSolution; }
		std::vector<std::shared_ptr<Species>> getSpeciesList() { return speciesList; }
		[[nodiscard]] std::vector<SolutionPtr> getSolutions() const { return solutions; }
		[[nodiscard]] int getSize() const { return parameters.size; }
		[[nodiscard]] int getCurrentGeneration() const { return parameters.currentGeneration; }
		[[nodiscard]] int getNumGenerations() const { return parameters.numGenerations; }
		[[nodiscard]] bool isInitialized() const { return !solutions.empty(); }
		[[nodiscard]] const std::vector<double>& getBestFitnessHistory() const { return bestFitnessHistory; }
		[[nodiscard]] const std::vector<int>& getBestSolutionIdHistory() const { return bestSolutionIdHistory; }
		[[nodiscard]] const std::vector<Genome>& getBestSolutionGenomeHistory() const { return bestSolutionGenomeHistory; }

		[[nodiscard]] const ValidationReport& getValidationReport() const { return validationReport; }
		void setValidationPolicy(ValidationPolicy policy) { validationPolicy = policy; }

		/// Sets the ValidationPolicy every subsequently constructed Population
		/// starts with. Production code never calls this (default stays Log);
		/// the test binary calls it once at startup (see tests/entry.cpp) so
		/// invariant violations throw during tests without any production
		/// behaviour change.
		static void setDefaultValidationPolicy(ValidationPolicy policy) { defaultValidationPolicy = policy; }

		void setSize(const int size) { parameters.size = size; }
		void setNumGenerations(const int numGenerations) { parameters.numGenerations = numGenerations; }

		void pause() { control.pause = true; }
		void resume() { control.pause = false; }
		void stop() { control.stop = true; }
		void start() { control.stop = false; }
	private:
		static inline ValidationPolicy defaultValidationPolicy = ValidationPolicy::Log;
		void evaluate() const;
		void speciate();
		void reproduceAndSelect();

		[[nodiscard]] bool endConditionMet() const;

		void startup();
		void upkeep();
		void cleanup();
		void createInitialSolutions(const SolutionPtr& initialSolution);
		void buildInitialSolutionsGenome() const;

		void assignToSpecies(const SolutionPtr& solution);
		std::shared_ptr<Species> findSpecies(const SolutionPtr& solution);
		[[nodiscard]] std::shared_ptr<Species> getBestActiveSpecies() const;

		void calculateAdjustedFitness();
		void assignOffspringToSpecies();
		void clearSpeciesOffspring() const;
		bool hasFitnessImprovedOverTheLastGenerations();
		void assignOffspringToTopTwoSpecies();
		void sortSpeciesListByChampionFitness();
		void assignOffspringBasedOnAdjustedFitness() const;
		void reassignOffspringIfFitnessIsStagnant() const;

		void pruneWorsePreformingSolutions() const;
		void replaceEntirePopulationWithOffspring();
		void preserveGlobalBestSolution();
		void mutate();

		void upkeepBestSolution();
		void upkeepChampions();
		void upkeepPerGenerationStatistics();
		void updateGenerationAndAges();
		void validateElitism();
		void validateUniqueSolutions();
		void validatePopulationSize();
		void validateUniqueGenesInGenomes();
		void validateUniqueKernelAndNeuralFieldPtrs();
		void validateIfSpeciesHaveUniqueRepresentative();
		void validateAssignmentIntoSpecies();

		void print() const;

		static void resetGenerationalInnovations();
		void clearLastMutations() const;

		void logSolutions() const;
		void logSpecies() const;
		void logOverview() const;
	};
}