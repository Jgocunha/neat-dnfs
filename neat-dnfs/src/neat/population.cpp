#include "neat/population.h"

#include "neat/population_file_manager.h"
#include "neat_tools/profiler.h"
#include <algorithm>
#include <cassert>
#include <format>
#include <atomic>
#include <thread>
#include <cmath>

namespace neat_dnfs
{
	int ValidationReport::total() const
	{
		int sum = 0;
		for (const int c : counts)
		{
			sum += c;
		}
		return sum;
	}

	int ValidationReport::count(const ValidationCheck check) const
	{
		return counts.at(static_cast<size_t>(check));
	}

	void ValidationReport::clear()
	{
		counts.fill(0);
		messages.clear();
	}

	ValidationError::ValidationError(const ValidationCheck check, const std::string& message)
		: std::runtime_error(message), check(check)
	{}

	void Population::reportViolation(const ValidationCheck check, const std::string& message)
	{
		validationReport.counts.at(static_cast<size_t>(check))++;
		if (validationReport.messages.size() < ValidationReport::maxRetainedMessages)
		{
			validationReport.messages.push_back(message);
		}
		log(tools::logger::LogLevel::FATAL, message, tools::logger::LogOutputMode::CONSOLE);
		if (validationPolicy == ValidationPolicy::Throw)
		{
			throw ValidationError(check, message);
		}
	}

	PopulationParameters::PopulationParameters(const int size, const int numGenerations, const double targetFitness, const bool parallelEvolution)
		: size(size), numGenerations(numGenerations), targetFitness(targetFitness), parallelEvolution(parallelEvolution)
	{
		// A population with no individuals is a category error, not a degenerate
		// run: Population::createInitialSolutions() would leave `solutions` empty,
		// so upkeepBestSolution() would leave bestSolution null and evolve() would
		// dereference it in endConditionMet() and in
		// hasFitnessImprovedOverTheLastGenerations(). Rejecting here keeps
		// bestSolution non-null for the whole lifetime of any Population that
		// exists at all, instead of guarding every use site.
		if (size <= 0)
		{
			throw std::invalid_argument("Population size must be greater than 0");
		}
	}

	PopulationControl::PopulationControl(bool pause, bool stop)
		: pause(pause), stop(stop)
	{}

	Population::Population(const PopulationParameters& parameters, const SolutionPtr& initialSolution, const bool enableFileIO)
		: parameters(parameters)
	{
		createInitialSolutions(initialSolution);
		if (enableFileIO)
		{
			fileManager = std::make_unique<PopulationFileManager>(*this);
		}
	}

	Population::~Population()
	{
		bestSolution = nullptr;
		speciesList.clear();
		champions.clear();
		solutions.clear();
	}

	void Population::resetGlobalCounters()
	{
		Species::resetUniqueIdentifier();
		Genome::resetGlobalInnovationNumber();
		Solution::resetUniqueIdentifier();
	}

	void Population::initialize() const
	{
		buildInitialSolutionsGenome();
	}

	void Population::startup()
	{
		statistics.start = std::chrono::steady_clock::now();
		if (fileManager)
		{
			fileManager->setFileDirectory();
		}
	}

	void Population::evolve()
	{
		startup();

		do
		{
			tools::profiler::resetGeneration();

			{
				const tools::profiler::ScopedTimer timer("evaluate");
				evaluate();
			}
			{
				const tools::profiler::ScopedTimer timer("speciate");
				speciate();
			}
			{
				const tools::profiler::ScopedTimer timer("upkeep");
				upkeep();
			}
			{
				const tools::profiler::ScopedTimer timer("reproduceAndSelect");
				reproduceAndSelect();
			}

#ifdef NEAT_DNFS_PROFILE
			if (fileManager)
			{
				fileManager->saveProfileForGeneration();
			}
#endif

			while (control.pause)
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(300));
				tools::logger::log(tools::logger::LogLevel::INFO, "Evolution paused.");
			}

		} while (!endConditionMet());

		cleanup();
	}

	void Population::evaluate() const
	{
		const size_t solutionCount = solutions.size();
		const unsigned hardwareConcurrency = std::max(1U, std::thread::hardware_concurrency());
		const size_t numWorkers = std::min<size_t>(hardwareConcurrency, solutionCount);

		if (!parameters.parallelEvolution || numWorkers <= 1)
		{
			for (const auto& solution : solutions)
			{
				solution->evaluate();
			}
			return;
		}

		std::atomic<size_t> nextIndex{ 0 };
		std::vector<std::future<void>> futures;
		futures.reserve(numWorkers);
		for (size_t w = 0; w < numWorkers; ++w)
		{
			futures.emplace_back(std::async(std::launch::async, [this, &nextIndex, solutionCount]()
				{
					for (size_t i = nextIndex.fetch_add(1); i < solutionCount; i = nextIndex.fetch_add(1))
					{
						solutions[i]->evaluate();
					}
				}));
		}

		std::exception_ptr firstError;
		for (auto& future : futures)
		{
			try
			{
				future.get();
			}
			catch (...)
			{
				if (!firstError)
				{
					firstError = std::current_exception();
				}
			}
		}

		if (firstError)
		{
			std::rethrow_exception(firstError);
		}
	}

	void Population::speciate()
	{
		for (const auto& solution : solutions)
		{
			assignToSpecies(solution);
		}

		// A species that lost all its members to reassignment this generation is
		// not touched again until crossover() runs later in reproduceAndSelect(),
		// so without this it would still look alive (and keep a stale
		// representative) when upkeep()'s validation runs right after speciate().
		for (const auto& species : speciesList)
		{
			if (!species->isExtinct() && species->size() == 0)
			{
				species->extinguish();
			}
		}

		for (const auto& species : speciesList)
		{
			species->assignChampion();
		}

		if (validationPolicy == ValidationPolicy::Throw)
		{
			validateAssignmentIntoSpecies();
		}

		calculateAdjustedFitness();
	}

	void Population::reproduceAndSelect()
	{
		assignOffspringToSpecies();

		if (PopulationConstants::logSpecies)
		{
			logSpecies();
		}

		pruneWorsePreformingSolutions();
		replaceEntirePopulationWithOffspring();
		mutate();
	}

	void Population::upkeep()
	{
		upkeepBestSolution();
		upkeepChampions();
		upkeepPerGenerationStatistics();

		if (PopulationConstants::logSolutions)
		{
			logSolutions();
		}
		if (PopulationConstants::logOverview)
		{
			logOverview();
		}
		if (PopulationConstants::logSpecies)
		{
			logSpecies();
		}

		if (validationPolicy == ValidationPolicy::Throw)
		{
			validatePopulationSize();
			validateIfSpeciesHaveUniqueRepresentative();
			validateUniqueSolutions();
			validateElitism();
			validateUniqueGenesInGenomes();
			validateUniqueKernelAndNeuralFieldPtrs();
		}

		if (fileManager)
		{
			fileManager->saveOverviewForGeneration();
		}

		resetGenerationalInnovations();
		updateGenerationAndAges();

		if (fileManager)
		{
			// Nested inside the "upkeep" scope above (upkeep() is what calls
			// this), so "upkeep"'s total is inclusive of "save" rather than
			// disjoint from it -- see PopulationFileManager::saveProfileForGeneration.
			const tools::profiler::ScopedTimer timer("save");
			fileManager->savePerGenerationData();
		}

		clearLastMutations();
	}

	void Population::cleanup()
	{
		statistics.end = std::chrono::steady_clock::now();
		statistics.duration = std::chrono::duration_cast<std::chrono::seconds>(statistics.end - statistics.start).count();

		if (fileManager)
		{
			fileManager->saveEndOfRunData();
		}
	}


	void Population::createInitialSolutions(const SolutionPtr& initialSolution)
	{
		initialSolution->buildPhenotype();
		const size_t numElements = initialSolution->getPhenotype().getNumberOfElements();

		if (numElements > 0) // incremental evolution, non-empty initial solution
		{
			for (int i = 0; i < parameters.size; i++)
			{
				solutions.emplace_back(initialSolution->copy());
			}
		}
		else // empty initial/base solution
		{
			for (int i = 0; i < parameters.size; i++)
			{
				solutions.emplace_back(initialSolution->clone());
			}
		}
	}

	void Population::buildInitialSolutionsGenome() const
	{
		for (const auto& solution : solutions)
		{
			solution->initialize();
		}
	}

	void Population::upkeepBestSolution()
	{
		bestSolution = nullptr;
		for (const auto& solution : solutions)
		{
			if (bestSolution == nullptr || solution->getFitness() > bestSolution->getFitness())
			{
				bestSolution = solution;
			}
		}
	}

	void Population::upkeepChampions()
	{
		champions.clear();
		for (const auto& species : speciesList)
		{
			champions.emplace_back(species->getChampion());
		}
	}

	void Population::upkeepPerGenerationStatistics()
	{
		// PopulationParameters rejects size <= 0, so `solutions` is never empty,
		// and upkeepBestSolution() -- which always runs immediately before this in
		// upkeep() -- only ever assigns bestSolution from `solutions`, so it is
		// non-null here. A violation would mean that invariant broke elsewhere,
		// not that this is a normal, recoverable state.
		assert(!solutions.empty() && "population must have solutions when statistics are computed");

		// average fitness
		perGenStatistics.averageFitness = 0.0;
		for (const auto& solution : solutions)
		{
			perGenStatistics.averageFitness += solution->getFitness();
		}
		perGenStatistics.averageFitness /= static_cast<double>(solutions.size());

		// best fitness
		perGenStatistics.bestFitness = bestSolution->getFitness();
		bestFitnessHistory.push_back(perGenStatistics.bestFitness);
		bestSolutionIdHistory.push_back(bestSolution->getId());
		bestSolutionGenomeHistory.push_back(bestSolution->getGenome());

		// number of species
		perGenStatistics.numberOfSpecies = static_cast<int>(speciesList.size());

		// number of active species
		perGenStatistics.numberOfActiveSpecies = 0;
		for (const auto& species : speciesList)
		{
			if (species->isExtinct())
			{
				continue;
			}
			perGenStatistics.numberOfActiveSpecies++;
		}

		// average compatibility distance
		// to do

		// innovation number
		for (const auto& solution : solutions)
		{
			auto solutionInnovs = solution->getGenome().getInnovationNumbers();
			for (const auto& solutionInnov : solutionInnovs)
			{
				if (solutionInnov > perGenStatistics.innovationNumber)
				{
					perGenStatistics.innovationNumber = solutionInnov;
				}
			}
		}

		// average genome size
		perGenStatistics.averageGenomeSize = 0.0;
		for (const auto& solution : solutions)
		{
			perGenStatistics.averageGenomeSize += static_cast<double>(solution->getNumConnectionGenes() + solution->getNumFieldGenes());
		}
		perGenStatistics.averageGenomeSize /= static_cast<double>(solutions.size());

		// average connection genes
		perGenStatistics.averageConnectionGenes = 0.0;
		for (const auto& solution : solutions)
		{
			perGenStatistics.averageConnectionGenes += static_cast<double>(solution->getNumConnectionGenes());
		}
		perGenStatistics.averageConnectionGenes /= static_cast<double>(solutions.size());

		// average field genes
		perGenStatistics.averageFieldGenes = 0.0;
		for (const auto& solution : solutions)
		{
			perGenStatistics.averageFieldGenes += static_cast<double>(solution->getNumFieldGenes());
		}
		perGenStatistics.averageFieldGenes /= static_cast<double>(solutions.size());
	}


	void Population::updateGenerationAndAges()
	{
		parameters.currentGeneration++;
		for (const auto& solution : solutions)
		{
			solution->incrementAge();
		}
		for (const auto& species : speciesList)
		{
			if (!species->isExtinct())
			{
				species->incrementAge();
			}
		}
	}	

	void Population::assignToSpecies(const SolutionPtr& solution)
	{
		bool assigned = false;
		const std::shared_ptr<Species> currentSpecies = findSpecies(solution);
		for (const auto& species : speciesList)
		{
			if (!species->isExtinct())
			{
				if (species->isCompatible(solution))
				{
					if (currentSpecies != species) 
					{
						if (currentSpecies != nullptr)
						{
							currentSpecies->removeSolution(solution);
						}
						species->addSolution(solution);
					}
					solution->setSpeciesId(species->getId());
					// The representative must stay fixed for the whole
					// assignment pass -- standard NEAT measures compatibility
					// against the previous generation's representative, not
					// one that drifts as more solutions join this species
					// during the same pass. A brand-new species still picks
					// its initial representative below.
					assigned = true;
					break;
				}
			}
		}
		if (!assigned)
		{
			if (currentSpecies != nullptr)
			{
				currentSpecies->removeSolution(solution);
			}

			auto newSpecies = std::make_shared<Species>();
			newSpecies->addSolution(solution);
			solution->setSpeciesId(newSpecies->getId());
			newSpecies->randomlyAssignRepresentative();
			speciesList.emplace_back(newSpecies);
		}
	}

	std::shared_ptr<Species> Population::findSpecies(const SolutionPtr& solution)
	{
		for (auto& species : speciesList)
		{
			if (species->contains(solution))
			{
				return species;
			}
		}
		return nullptr;
	}

	std::shared_ptr<Species> Population::getBestActiveSpecies() const
	{
		std::shared_ptr<Species> bestSpecies = nullptr;
		double bestFitness = 0.0;
		for (const auto& species : speciesList)
		{
			if (species->isExtinct())
			{
				continue;
			}
			const SolutionPtr champ = species->getChampion();
			if (champ == nullptr)
			{
				continue;
			}
			if (champ->getFitness() > bestFitness)
			{
				bestFitness = champ->getFitness();
				bestSpecies = species;
			}
		}
		return bestSpecies;
	}

	void Population::calculateAdjustedFitness()
	{
		for (const auto& solution : solutions)
		{
			const std::shared_ptr<Species> species = findSpecies(solution);
			// Invariant: calculateAdjustedFitness() only runs from speciate(),
			// right after its loop has called assignToSpecies(solution) for
			// every solution in `solutions` -- placing each one into either an
			// existing compatible species or a brand-new one. The empty-species
			// extinguish pass that follows only removes species with zero
			// members, never the one that was just given this solution, so
			// findSpecies() cannot return nullptr here. A null result would
			// mean that invariant was broken elsewhere, not that this is a
			// normal, recoverable state.
			assert(species != nullptr && "solution must belong to a species when adjusted fitness is calculated");
			const size_t speciesSize = species->size();
			const double adjustedFitness = solution->getFitness() / static_cast<double>(speciesSize);
			if (std::isnan (adjustedFitness))
			{
				log(tools::logger::LogLevel::FATAL, "Adjusted fitness is NaN.");
				log(tools::logger::LogLevel::FATAL, std::format("Fitness: {} Species size: {}", solution->getFitness(), speciesSize));
				throw std::runtime_error("Adjusted fitness is NaN.");
			}
			solution->setAdjustedFitness(adjustedFitness);
		}
	}

	void Population::assignOffspringToSpecies()
	{
		clearSpeciesOffspring();
		// if fitness of population does not improve for Y generations
		// only the top two species are allowed to reproduce
		// (a species is "better than the other" based on its champion)
		const int numActiveSpecies =
			static_cast<int>(std::ranges::count_if(speciesList.begin(), speciesList.end(), [](const auto& species)
			{ return !species->isExtinct(); }));
		if (!hasFitnessImprovedOverTheLastGenerations())
		{
			if (numActiveSpecies > 2)
			{
				assignOffspringToTopTwoSpecies();
				return;
			}
		}
		// every species is assigned a potentially different number of offspring
		// in proportion to the sum of adjusted fitness of its members fitness
		assignOffspringBasedOnAdjustedFitness();
		// after X generations if fitness did not improve, the species is not allowed to reproduce
		reassignOffspringIfFitnessIsStagnant();
	}

	void Population::clearSpeciesOffspring() const
	{
		for (const auto& species : speciesList)
		{
			species->setOffspringCount(0);
		}
	}

	bool Population::hasFitnessImprovedOverTheLastGenerations()
	{
		if (bestSolution->getFitness() > previousBestFitness)
		{
			previousBestFitness = bestSolution->getFitness();
			previousBestSolution = bestSolution;
			generationsWithoutImprovement = 0;
			hasFitnessImproved = true;
			return true;
		}
		hasFitnessImproved = false;
		generationsWithoutImprovement++;
		if (generationsWithoutImprovement >= PopulationConstants::generationsWithoutImprovementThresholdInPopulation)
		{
			generationsWithoutImprovement = 0;
			return false;
		}

		return true;
	}

	void Population::assignOffspringToTopTwoSpecies()
	{
		// sort the two best species to the beginning of the list
		sortSpeciesListByChampionFitness();

		// assign offspring count only to the top two **non-extinct** species
		int assigned = 0;
		for (const auto& species : speciesList) 
		{
			if (!species->isExtinct()) 
			{
				species->setOffspringCount(parameters.size / 2);
				if (++assigned == 2)
				{
					break; // Stop after assigning two species
				}
			}
		}
		log(tools::logger::LogLevel::WARNING, std::format("Fitness of entire population has not improved for the last {} generations. Assigned offspring to top two species", 
                                                      PopulationConstants::generationsWithoutImprovementThresholdInPopulation));
	}

	void Population::sortSpeciesListByChampionFitness()
	{
		std::ranges::sort(speciesList, [](const auto& a, const auto& b) {
			if (a->isExtinct() != b->isExtinct()) {
				return !a->isExtinct(); // Non-extinct species come first
			}
			// Handle cases where getChampion() might return nullptr
			const SolutionPtr championA = a->getChampion();
			const SolutionPtr championB = b->getChampion();
			if (!championA && !championB) {
				return false; // If both are null, maintain relative order
			}
			if (!championA) {
				return false; // Null champions should be treated as less fit
			}
			if (!championB) {
				return true; // Non-null champions come before null ones
			}
			return championA->getFitness() > championB->getFitness(); // Sort by fitness
			});
	}

	void Population::assignOffspringBasedOnAdjustedFitness() const
	{
		double total_adjusted_fitness = 0.0;

		// Step 1: Calculate total adjusted fitness
		for (const auto& species_ptr : speciesList)  // Use auto& to iterate over shared_ptr
		{
			total_adjusted_fitness += species_ptr->totalAdjustedFitness();
		}

		// Step 2: Assign offspring count based on fitness proportion
		const int total_offspring = parameters.size; // Define how many new organisms we want

		double accumulated_offspring = 0.0;
		int assigned_offspring = 0;

		for (const auto& species_ptr : speciesList)
		{
			if (total_adjusted_fitness > 0)
			{
				species_ptr->setOffspringCount(
					static_cast<int>((species_ptr->totalAdjustedFitness() / total_adjusted_fitness) * total_offspring));
			}
			else
			{
				species_ptr->setOffspringCount(0); // Edge case: If total fitness is 0, prevent division error
			}

			// Step 3: Stochastic Rounding
			accumulated_offspring += species_ptr->getOffspringCount();
			const int rounded_offspring = static_cast<int>(std::lround(accumulated_offspring));
			species_ptr->setOffspringCount(rounded_offspring - assigned_offspring);
			assigned_offspring += species_ptr->getOffspringCount();
		}

		// Ensure total assigned offspring matches population_size
		while (assigned_offspring < total_offspring)
		{
			// Assign an extra offspring to the best-performing species
			std::shared_ptr<Species> best_species = nullptr;
			double max_fitness = -1.0;

			for (const auto& species_ptr : speciesList)
			{
				if (species_ptr->totalAdjustedFitness() > max_fitness)
				{
					max_fitness = species_ptr->totalAdjustedFitness();
					best_species = species_ptr;
				}
			}

			if (best_species)
			{
				best_species->setOffspringCount(best_species->getOffspringCount() + 1);
				assigned_offspring++;
			}
		}
	}

	void Population::reassignOffspringIfFitnessIsStagnant() const
	{
		int totalOffspringToReassign = 0;
		for (const auto& species : speciesList)
		{
			if (species->getOffspringCount() == 0)
			{
				continue;
			}

			if (!species->hasFitnessImprovedOverTheLastGenerations())
			{
				totalOffspringToReassign += species->getOffspringCount();
				species->setOffspringCount(0);
				log(tools::logger::LogLevel::WARNING, std::format("Fitness of species {} has not improved for the last {} generations.", 
                                                          species->getId(), 
                                                          PopulationConstants::generationsWithoutImprovementThresholdInSpecies));
			}
		}
		if (totalOffspringToReassign == 0)
		{
			return;
		}
		// give the offspring to the top species
		const std::shared_ptr<Species> topSpecies = getBestActiveSpecies();
		if (topSpecies == nullptr)
		{
			return;
		}
		topSpecies->setOffspringCount(topSpecies->getOffspringCount() + totalOffspringToReassign);
		log(tools::logger::LogLevel::WARNING, std::format("Reassigned {} offspring to species {}", totalOffspringToReassign, topSpecies->getId()));
	}

	void Population::pruneWorsePreformingSolutions() const
	{
		// species then reproduce by eliminating the lowest performing members of the population
		for (const auto& species : speciesList)
		{
			species->pruneWorsePerformingMembers(PopulationConstants::pruneRatio);
		}
	}

	void Population::replaceEntirePopulationWithOffspring()
	{
		// the entire population is then replaced by the offspring
		// of the remaining organisms in each species

		// if elitism is enabled
		// the champion of each species with more than five networks
		// is copied into the next generation unchanged

		for (const auto& species : speciesList)
		{
			species->crossover(); // creation of offspring
			species->replaceMembersWithOffspring(); // replacement of population with offspring
			if (PopulationConstants::elitism)
			{
				if (species->size() > 5)
				{
					species->copyChampionToNextGeneration(); // elitism
				}
			}
		}
		// A species that lost all its members -- whether speciate() extinguished
		// it earlier this generation or crossover() just found it empty -- is
		// done for good: it produces no offspring and assignToSpecies() never
		// reassigns into an extinct species. Erasing it here, right after every
		// species has had its one crossover() pass, keeps speciesList (and the
		// per-generation species count derived from it) reflecting only species
		// that are still alive, instead of accumulating dead ones for the rest
		// of the run (issue #59).
		std::erase_if(speciesList, [](const std::shared_ptr<Species>& species)
			{ return species->isExtinct(); });

		solutions.clear();
		for (const auto& species : speciesList)
		{
			const auto speciesSolutions = species->getMembers();
			solutions.insert(solutions.end(), speciesSolutions.begin(), speciesSolutions.end());
		}

		preserveGlobalBestSolution();
	}

	void Population::preserveGlobalBestSolution()
	{
		if (!PopulationConstants::elitism || previousBestSolution == nullptr)
		{
			return;
		}
		if (std::ranges::find(solutions, previousBestSolution) != solutions.end())
		{
			return; // already survived reproduction on its own
		}

		// Find and evict the current worst solution to make room, keeping
		// solutions.size() == parameters.size (validatePopulationSize enforces this).
		auto worstIt = std::ranges::min_element(solutions,
			[](const SolutionPtr& a, const SolutionPtr& b) { return a->getFitness() < b->getFitness(); });
		if (worstIt == solutions.end())
		{
			return;
		}
		const SolutionPtr worst = *worstIt;

		const auto worstSpecies = findSpecies(worst);
		if (worstSpecies != nullptr)
		{
			worstSpecies->removeSolution(worst);
		}
		solutions.erase(worstIt);

		// Route through the same compatibility-based placement every other
		// solution uses -- never insert into a species by positional
		// convenience, since that would corrupt that species' genetic
		// coherence and its adjusted-fitness-based offspring allocation.
		solutions.push_back(previousBestSolution);
		assignToSpecies(previousBestSolution);
	}

	void Population::mutate()
	{
		upkeepBestSolution();
		upkeepChampions();
		for (const auto& solution : solutions)
		{
			// if champion, do not mutate
			if (solution != bestSolution && !std::ranges::any_of(champions,
																 [&solution](const auto& champion)
																 { return champion == solution; }))
			{
				solution->mutate();
			}
		}
	}

	bool Population::endConditionMet() const
	{
		const bool fitnessCondition = bestSolution->getFitness() > parameters.targetFitness;
		const bool generationCondition = parameters.currentGeneration >= parameters.numGenerations;
		return fitnessCondition || generationCondition || control.stop;
	}

	void Population::validateElitism()
	{
		// Nothing to compare against yet -- previousBestSolution/previousBestFitness
		// only hold meaningful state once hasFitnessImprovedOverTheLastGenerations()
		// has recorded a first improvement. Guarding on currentGeneration instead
		// would be off-by-one: it increments in updateGenerationAndAges(), which runs
		// after this check within the same upkeep() call, so currentGeneration == 1
		// is already the *second* generation's check, not the first.
		if (previousBestSolution == nullptr)
		{
			return;
		}

		const double bestFitness = bestSolution->getFitness();
		if (bestFitness >= previousBestFitness - PopulationConstants::elitismFitnessEpsilon)
		{
			return;
		}

		// Fitness dropped beyond jitter tolerance, which the DNF simulation's
		// noise can cause on its own (e.g. a bump forming or not near a
		// solution's decision boundary swings a partial-fitness term far more
		// than typical re-evaluation jitter). What elitism actually guarantees
		// is that the previous best genome itself survives reproduction, not
		// that its re-measured fitness stays within a fixed tolerance -- so
		// this is only a genuine violation if that genome is gone.
		const bool previousBestStillPresent = std::ranges::any_of(solutions,
			[this](const SolutionPtr& solution) { return solution == previousBestSolution; });
		if (previousBestStillPresent)
		{
			return;
		}

		reportViolation(ValidationCheck::Elitism, std::format(
			"Best fitness decreased and the previous best solution was lost. previous={} current={}",
			previousBestFitness, bestFitness));
	}

	void Population::validateUniqueSolutions()
	{
		int counter = 0;
		for (size_t i = 0; i < solutions.size(); ++i)
		{
			for (size_t j = i + 1; j < solutions.size(); ++j)
			{
				if (solutions[i] == solutions[j])
				{
					counter++;
				}
			}
		}
		if(counter > 0)
		{
			reportViolation(ValidationCheck::UniqueSolutions,
				std::format("Duplicate solutions found. count={}", counter));
		}
	}

	void Population::validatePopulationSize()
	{
		if (solutions.size() != parameters.size)
		{
			reportViolation(ValidationCheck::PopulationSize, "Population size does not match parameters.");
		}
	}

	void Population::validateUniqueGenesInGenomes()
	{
		for (const auto& solution : solutions)
		{
			const auto& genome = solution->getGenome();
			const auto& connectionGenes = genome.getConnectionGenes();
			for (auto const& connectionGene1 : connectionGenes)
			{
				for (auto const& connectionGene2 : connectionGenes)
				{
					if (connectionGene1 != connectionGene2)
					{
						if (connectionGene1.getInFieldGeneId() == connectionGene2.getInFieldGeneId() &&
							connectionGene1.getOutFieldGeneId() == connectionGene2.getOutFieldGeneId() &&
							connectionGene1.getInnovationNumber() == connectionGene2.getInnovationNumber())
						{
							const auto inFieldGeneId = connectionGene1.getInFieldGeneId();
							const auto outFieldGeneId = connectionGene1.getOutFieldGeneId();
							const auto innovationNumber = connectionGene1.getInnovationNumber();
							reportViolation(ValidationCheck::UniqueGenesInGenomes, std::format(
								"Connection genes are the same. InFieldGeneId: {} OutFieldGeneId: {} InnovationNumber: {}",
								inFieldGeneId, outFieldGeneId, innovationNumber));
						}
					}
				}
			}
		}
	}

	void Population::validateUniqueKernelAndNeuralFieldPtrs()
	{
		for (const auto& solution_a : solutions)
		{
			const auto& genome_a = solution_a->getGenome();
			const auto& connectionGenes_a = genome_a.getConnectionGenes();
			const auto& fieldGenes_a = genome_a.getFieldGenes();

			for (const auto& solution_b : solutions)
			{
				if (solution_a == solution_b)
				{
					continue;
				}

				const auto& genome_b = solution_b->getGenome();
				const auto& connectionGenes_b = genome_b.getConnectionGenes();
				const auto& fieldGenes_b = genome_b.getFieldGenes();

				for (const auto& connectionGene_a : connectionGenes_a)
				{
					for (const auto& connectionGene_b : connectionGenes_b)
					{
						const auto kernel_a = connectionGene_a.getKernel();
						const auto kernel_b = connectionGene_b.getKernel();
						if (kernel_a == kernel_b)
						{
							reportViolation(ValidationCheck::UniqueKernelAndNeuralFieldPtrs, "Kernels are the same.");
						}
					}
				}

				for (const auto& fieldGene_a : fieldGenes_a)
				{
					for (const auto& fieldGene_b : fieldGenes_b)
					{
						const auto neuralField_a = fieldGene_a.getNeuralField();
						const auto neuralField_b = fieldGene_b.getNeuralField();
						if (neuralField_a == neuralField_b)
						{
							reportViolation(ValidationCheck::UniqueKernelAndNeuralFieldPtrs, "Neural fields are the same.");
						}
					}
				}

			}
		}
	}

	void Population::validateIfSpeciesHaveUniqueRepresentative()
	{
		for (const auto& species_a : speciesList)
		{
			for (const auto& species_b : speciesList)
			{
				if (species_a->getId() == species_b->getId())
				{
					continue;
				}
				if (species_a->isExtinct() || species_b->isExtinct())
				{
					continue;
				}

				const auto representative_a = species_a->getRepresentative()->getAddress();
				const auto representative_b = species_b->getRepresentative()->getAddress();

				if (representative_a == representative_b)
				{
					reportViolation(ValidationCheck::SpeciesHaveUniqueRepresentative, std::format(
						"Species have the same representative. Species a id: {} Representative a id: {} Species b id: {} Representative b id: {}",
						species_a->getId(), representative_a, species_b->getId(), representative_b));
				}
			}
		}
	}

	void Population::validateAssignmentIntoSpecies()
	{
		std::vector<SolutionPtr> speciesSolutions;
		speciesSolutions.reserve(parameters.size);
		for (const auto& species : speciesList)
		{
			for (const auto& member : species->getMembers())
			{
				speciesSolutions.emplace_back(member);
			}
		}

		if (speciesSolutions.size() != static_cast<size_t>(parameters.size))
		{
			reportViolation(ValidationCheck::AssignmentIntoSpecies, std::format(
				"Species membership count does not match population size. expected={} actual={}",
				parameters.size, speciesSolutions.size()));
		}

		int counter = 0;
		for (size_t i = 0; i < speciesSolutions.size(); ++i)
		{
			for (size_t j = i + 1; j < speciesSolutions.size(); ++j)
			{
				if (speciesSolutions[i] == speciesSolutions[j])
				{
					counter++;
				}
			}
		}
		if (counter > 0)
		{
			reportViolation(ValidationCheck::AssignmentIntoSpecies,
				std::format("Duplicate solutions found after speciation. count={}", counter));
		}
	}

	void Population::print() const
	{
		std::string result = "Population: \n";
		for (const auto& solution : solutions)
		{
			std::stringstream addr;
			addr << solution.get();
			result += "Solution address: " + addr.str() + "\n";
			result += std::format("Fitness is: {}\n", solution->getFitness());
			const auto& genome = solution->getGenome();
			for (const auto& nodeGene : genome.getFieldGenes())
			{
				result += nodeGene.toString();
			}
			for (const auto& connectionGene : genome.getConnectionGenes())
			{
				result += connectionGene.toString();
			}
			result += "\n";
		}
		log(tools::logger::LogLevel::INFO, result);
	}

	void Population::resetGenerationalInnovations()
	{
		// clearGenerationalInnovations() clears static, per-generation
		// innovation bookkeeping shared by every Genome -- it is not instance
		// data, so it belongs on the class, not routed through bestSolution
		// (which can be null, e.g. for an empty population). Calling a static
		// method through an instance pointer is misleading regardless of
		// nullness, so this always goes through the class directly now.
		Genome::clearGenerationalInnovations();
	}

	void Population::clearLastMutations() const
	{
		for (const auto& solution : solutions)
		{
			solution->clearLastMutations();
		}
	}

	void Population::logSolutions() const
	{
		for (const auto& solution : solutions)
		{
			solution->print();
		}
	}

	void Population::logSpecies() const
	{
		for (const auto& species : speciesList)
		{
			species->print();
		}
	}

	void Population::logOverview() const
{
    log(tools::logger::LogLevel::INFO, std::format(
        "Current generation: {}\n"
        " Number of solutions: {}\n"
        " Number of species: {}\n"
        " Number of active species: {}\n"
        " Has fitness improved: {}\n"
        " Number of generations without improvement: {}\n"
        " Average fitness: {}\n"
        " Best fitness: {}\n"
        " Innovation number: {}\n"
        " Average genome size: {}\n"
        " Average connection genes: {}\n"
        " Average field genes: {}\n"
        " Best solution: [{}]",
        parameters.currentGeneration,
        solutions.size(),
        perGenStatistics.numberOfSpecies,
        perGenStatistics.numberOfActiveSpecies,
        hasFitnessImproved ? "yes" : "no",
        generationsWithoutImprovement,
        perGenStatistics.averageFitness,
        perGenStatistics.bestFitness,
        perGenStatistics.innovationNumber,
        perGenStatistics.averageGenomeSize,
        perGenStatistics.averageConnectionGenes,
        perGenStatistics.averageFieldGenes,
        bestSolution->toString()
    ));
}
}