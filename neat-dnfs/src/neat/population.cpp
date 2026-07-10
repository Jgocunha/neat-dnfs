#include "neat/population.h"

#include "neat/population_file_manager.h"
#include <format> 

namespace neat_dnfs
{
	PopulationParameters::PopulationParameters(const int size, const int numGenerations, const double targetFitness, const bool parallelEvolution)
		: size(size), numGenerations(numGenerations), targetFitness(targetFitness), parallelEvolution(parallelEvolution)
	{}

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
			evaluate();
			speciate();
			upkeep();

			reproduceAndSelect();

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
		if (parameters.parallelEvolution)
		{
			std::vector<std::future<void>> futures;
			for (const auto& solution : solutions)
			{
				futures.emplace_back(std::async(std::launch::async, [&solution]()
					{
						solution->evaluate();
					}));
			}

			for (auto& future : futures)
			{
				future.get();
			}
		}
		else
		{
			for (const auto& solution : solutions)
			{
				solution->evaluate();
			}
		}
	}

	void Population::speciate()
	{
		for (const auto& solution : solutions)
		{
			assignToSpecies(solution);
		}
		for (const auto& species : speciesList)
		{
			species->assignChampion();
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

		if (PopulationConstants::validatePopulationSize)
		{
			validatePopulationSize();
		}
		if (PopulationConstants::validateUniqueSolutions)
		{
			validateUniqueSolutions();
		}
		if (PopulationConstants::validateElitism)
		{
			validateElitism();
		}
		if (PopulationConstants::validateUniqueGenesInGenomes)
		{
			validateUniqueGenesInGenomes();
		}
		if (PopulationConstants::validateUniqueKernelAndNeuralFieldPtrs)
		{
			validateUniqueKernelAndNeuralFieldPtrs();
		}
		if (PopulationConstants::validateIfSpeciesHaveUniqueRepresentative)
		{
			validateIfSpeciesHaveUniqueRepresentative();
		}

		if (fileManager)
		{
			fileManager->saveOverviewForGeneration();
		}

		resetGenerationalInnovations();
		updateGenerationAndAges();

		if (fileManager)
		{
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
		// average fitness
		perGenStatistics.averageFitness = 0.0F;
		for (const auto& solution : solutions)
		{
			perGenStatistics.averageFitness += solution->getFitness();
		}
		perGenStatistics.averageFitness /= static_cast<double>(solutions.size());

		// best fitness
		perGenStatistics.bestFitness = bestSolution->getFitness();

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
		perGenStatistics.averageGenomeSize = 0.0F;
		for (const auto& solution : solutions)
		{
			perGenStatistics.averageGenomeSize += solution->getNumConnectionGenes() + solution->getNumFieldGenes();
		}
		perGenStatistics.averageGenomeSize /= static_cast<double>(solutions.size());

		// average connection genes
		perGenStatistics.averageConnectionGenes = 0.0F;
		for (const auto& solution : solutions)
		{
			perGenStatistics.averageConnectionGenes += solution->getNumConnectionGenes();
		}
		perGenStatistics.averageConnectionGenes /= static_cast<double>(solutions.size());

		// average field genes
		perGenStatistics.averageFieldGenes = 0.0F;
		for (const auto& solution : solutions)
		{
			perGenStatistics.averageFieldGenes += solution->getNumFieldGenes();
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
					species->randomlyAssignRepresentative();
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

		if (PopulationConstants::validateAssignmentIntoSpecies)
		{
			validateAssignmentIntoSpecies();
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
		solutions.clear();
		for (const auto& species : speciesList)
		{
			const auto speciesSolutions = species->getMembers();
			solutions.insert(solutions.end(), speciesSolutions.begin(), speciesSolutions.end());
		}
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

	void Population::validateElitism() const
	{
		static SolutionPtr pbs = nullptr; // previous best solution
		static SolutionPtr bs = nullptr; // best solution

		static double pbsf = 0.0;
		static double bsf = 0.0;

		if (parameters.currentGeneration == 1)
		{
			pbs = nullptr;
			bs = bestSolution;
			pbsf = 0.0;
			bsf = bestSolution->getFitness();
			return;
		}

		bs = bestSolution;
		bsf = bs->getFitness();

		static constexpr double epsilon = 0.000;
		const bool bsDecreased = bsf < pbsf - epsilon;
		bool pbsInPopulation = false;

		if (bsDecreased)
		{
			for (const auto& solution : solutions)
			{
				if (solution == pbs)
				{

					std::stringstream addr_bs;
					addr_bs << bs.get();
					std::stringstream addr_npbs;
					addr_npbs << solution.get();
					std::stringstream addr_opbs;
					addr_opbs << pbs.get();

					const double opbsf = pbsf;
					const double npbsf = solution->getFitness();

					log(tools::logger::LogLevel::WARNING, "Fitness decreased but previous best solution is in the population.");

					if (bs == pbs)
					{
						log(tools::logger::LogLevel::WARNING, "Best solution is the same as previous best solution.");
					}
					else
					{
						log(tools::logger::LogLevel::WARNING, "Best solution is not the same as previous best solution.");
					}

					//log(tools::logger::LogLevel::FATAL, "Best solution address: " + addr_bs.str() + " Fitness: " + std::to_string(bsf));
					//log(tools::logger::LogLevel::FATAL, "New previous best solution address: " + addr_npbs.str() + " Fitness: " + std::to_string(npbsf));
					//log(tools::logger::LogLevel::FATAL, "Old previous best solution address: " + addr_opbs.str() + " Fitness: " + std::to_string(opbsf));

					pbsInPopulation = true;
					break;
				}
			}
		}

		if (bsDecreased && !pbsInPopulation)
		{
			std::stringstream addr_bs;
			addr_bs << bs.get();
			std::stringstream addr_opbs;
			addr_opbs << pbs.get();
			log(tools::logger::LogLevel::WARNING, "Fitness decreased and previous best solution is not in the population.");
			log(tools::logger::LogLevel::WARNING, std::format("Best solution address: {} Fitness: {}", addr_bs.str(), bsf));
			log(tools::logger::LogLevel::WARNING, std::format("Previous best solution address: {} Fitness: {}", addr_opbs.str(), pbsf));
			//throw std::runtime_error("Best solution decreased and previous best solution not in population.");
		}

		pbs = bs;
		pbsf = bsf;
	}

	void Population::validateUniqueSolutions() const
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
			log(tools::logger::LogLevel::FATAL, "Duplicate solutions found.");
		}
	}

	void Population::validatePopulationSize() const
	{
		if (solutions.size() != parameters.size)
		{
			log(tools::logger::LogLevel::FATAL, "Population size does not match parameters.");
		}
	}

	void Population::validateUniqueGenesInGenomes() const
	{
		for (const auto& solution : solutions)
		{
			const auto genome = solution->getGenome();
			for (auto const& connectionGene1 : genome.getConnectionGenes())
			{
				for (auto const& connectionGene2 : genome.getConnectionGenes())
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
							log(tools::logger::LogLevel::FATAL, "Connection genes are the same.");
							log(tools::logger::LogLevel::FATAL, std::format("InFieldGeneId: {} OutFieldGeneId: {} InnovationNumber: {}", 
                                                        inFieldGeneId, 
                                                        outFieldGeneId, 
                                                        innovationNumber));
						}
					}
				}
			}
		}
	}

	void Population::validateUniqueKernelAndNeuralFieldPtrs() const
	{
		for (const auto& solution_a : solutions)
		{
			for (const auto& solution_b : solutions)
			{
				if (solution_a == solution_b)
				{
					continue;
				}

				const auto genome_a = solution_a->getGenome();
				const auto genome_b = solution_b->getGenome();
				const auto connectionGenes_a = genome_a.getConnectionGenes();
				const auto connectionGenes_b = genome_b.getConnectionGenes();
				const auto fieldGenes_a = genome_a.getFieldGenes();
				const auto fieldGenes_b = genome_b.getFieldGenes();

				for (const auto& connectionGene_a : connectionGenes_a)
				{
					for (const auto& connectionGene_b : connectionGenes_b)
					{
						const auto kernel_a = connectionGene_a.getKernel();
						const auto kernel_b = connectionGene_b.getKernel();
						if (kernel_a == kernel_b)
						{
							log(tools::logger::LogLevel::FATAL, "Kernels are the same.");
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
							log(tools::logger::LogLevel::FATAL, "Neural fields are the same.");
						}
					}
				}

			}
		}
	}

	void Population::validateIfSpeciesHaveUniqueRepresentative() const
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
					log(tools::logger::LogLevel::FATAL, "Species have the same representative.");
					log(tools::logger::LogLevel::FATAL, std::format("Species a id: {} Representative a id: {}", species_a->getId(), representative_a));
					log(tools::logger::LogLevel::FATAL, std::format("Species b id: {} Representative b id: {}", species_b->getId(), representative_b));
				}
			}
		}
	}

	void Population::validateAssignmentIntoSpecies() const
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
			log(tools::logger::LogLevel::FATAL, "Duplicate solutions found after speciation.");
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
			const auto genome = solution->getGenome();
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

	void Population::resetGenerationalInnovations() const
	{
		bestSolution->clearGenerationalInnovations();
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