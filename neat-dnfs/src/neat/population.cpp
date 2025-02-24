#include "neat/population.h"

namespace neat_dnfs
{
	PopulationParameters::PopulationParameters(int size, int numGenerations, double targetFitness)
		: size(size), currentGeneration(0), numGenerations(numGenerations), targetFitness(targetFitness)
	{}

	PopulationControl::PopulationControl(bool pause, bool stop)
		: pause(pause), stop(stop)
	{}

	Population::Population(const PopulationParameters& parameters, const SolutionPtr& initialSolution)
		: parameters(parameters), bestSolution(initialSolution->clone())
	{
		createInitialEmptySolutions(initialSolution);
	}

	void Population::initialize() const
	{
		buildInitialSolutionsGenome();
	}

	void Population::evolve()
	{
		startKeyListenerForUserCommands();

		do
		{
			evaluate();
			speciate();
			upkeep();
			reproduceAndSelect();

			while (control.pause)
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
				tools::logger::log(tools::logger::LogLevel::INFO, "Evolution paused.");
			}

		} while (!endConditionMet());

		saveAllSolutionsWithFitnessAbove(bestSolution->getFitness() - 0.1);
	}

	void Population::evaluate() const
	{
		if (PopulationConstants::parallelEvolution)
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
				solution->evaluate();
		}
	}

	void Population::speciate()
	{
		for (const auto& solution : solutions)
			assignToSpecies(solution);
	}

	void Population::reproduceAndSelect()
	{
		calculateAdjustedFitness();
		assignOffspringToSpecies();
		pruneWorsePreformingSolutions();
		replaceEntirePopulationWithOffspring();
		mutate();
	}

	void Population::upkeep()
	{
		upkeepBestSolution();

		if (PopulationConstants::logSolutions)
			logSolutions();
		if (PopulationConstants::logOverview)
			logOverview();
		if (PopulationConstants::logSpecies)
			logSpecies();
		if (PopulationConstants::logMutationStatistics)
			logMutationStatistics();

		if (PopulationConstants::validatePopulationSize)
			validatePopulationSize();
		if (PopulationConstants::validateUniqueSolutions)
			validateUniqueSolutions();
		if (PopulationConstants::validateElitism)
			validateElitism();
		if (PopulationConstants::validateUniqueGenesInGenomes)
			validateUniqueGenesInGenomes();
		if (PopulationConstants::validateUniqueKernelAndNeuralFieldPtrs)
			validateUniqueKernelAndNeuralFieldPtrs();
		if (PopulationConstants::validateIfSpeciesHaveUniqueRepresentative)
			validateIfSpeciesHaveUniqueRepresentative();

		resetGenerationalInnovations();
		updateGenerationAndAges();
		resetMutationStatisticsPerGeneration();
	}

	void Population::createInitialEmptySolutions(const SolutionPtr& initialSolution)
	{
		for (int i = 0; i < parameters.size; i++)
			solutions.push_back(initialSolution->clone());
	}

	void Population::buildInitialSolutionsGenome() const
	{
		for (const auto& solution : solutions)
			solution->initialize();
	}

	void Population::upkeepBestSolution()
	{
		for (const auto& solution : solutions)
		{
			if (bestSolution == nullptr || solution->getFitness() > bestSolution->getFitness())
				bestSolution = solution;
		}
	}

	void Population::updateGenerationAndAges()
	{
		parameters.currentGeneration++;
		for (const auto& solution : solutions)
			solution->incrementAge();
		for (auto& species : speciesList)
			if (!species.isExtinct())
				species.incrementAge();
	}	

	void Population::assignToSpecies(const SolutionPtr& solution)
	{
		bool assigned = false;
		Species* currentSpecies = findSpecies(solution);

		for (auto& species : speciesList)
		{
			if (species.isCompatible(solution))
			{
				if (currentSpecies != &species)
				{
					if (currentSpecies != nullptr)
						currentSpecies->removeSolution(solution);
					species.addSolution(solution);
					solution->setSpeciesId(species.getId());
				}
				assigned = true;
				species.randomlyAssignRepresentative();
				species.assignChampion();
				break;
			}
		}
		if (!assigned)
		{
			if (currentSpecies != nullptr)
				currentSpecies->removeSolution(solution);
			Species newSpecies;
			newSpecies.addSolution(solution);
			solution->setSpeciesId(newSpecies.getId());
			newSpecies.randomlyAssignRepresentative();
			newSpecies.assignChampion();
			speciesList.push_back(newSpecies);
		}
	}

	Species* Population::findSpecies(const SolutionPtr& solution)
	{
		for (auto& species : speciesList)
			if (species.contains(solution))
				return &species;
		return nullptr;
	}

	Species* Population::getBestActiveSpecies()
	{
		Species* bestSpecies = nullptr;
		double bestFitness = 0.0;
		for (auto& species : speciesList)
		{
			if (species.isExtinct())
				continue;
			if (species.getChampion()->getFitness() > bestFitness)
			{
				bestFitness = species.getChampion()->getFitness();
				bestSpecies = &species;
			}
		}
		return bestSpecies;
	}

	void Population::calculateAdjustedFitness()
	{
		for (const auto& solution : solutions)
		{
			const Species* species = findSpecies(solution);
			const size_t speciesSize = species->size();
			const double adjustedFitness = solution->getFitness() / static_cast<double>(speciesSize);
			if (std::isnan (adjustedFitness))
			{
				log(tools::logger::LogLevel::FATAL, "Adjusted fitness is NaN.");
				log(tools::logger::LogLevel::FATAL, "Fitness: " + std::to_string(solution->getFitness()) + " Species size: " + std::to_string(speciesSize));
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
		const int numActiveSpecies = std::ranges::count_if(speciesList.begin(), speciesList.end(), [](const Species& species)
			{ return !species.isExtinct(); });

		if (numActiveSpecies > 2)
		{
			if (!hasFitnessImprovedOverTheLastGenerations())
			{
				assignOffspringToTopTwoSpecies();
				return;
			}
		}

		// every species is assigned a potentially different number of offspring
		// in proportion to the sum of adjusted fitness of its members fitness
		assignOffspringBasedOnAdjustedFitness();

		// after X generations if fitness did not improve, the species is not allowed to reproduce
		//reassignOffspringIfFitnessIsStagnant();
	}

	void Population::clearSpeciesOffspring()
	{
		for (auto& species : speciesList)
			species.setOffspringCount(0);
	}

	bool Population::hasFitnessImprovedOverTheLastGenerations()
	{
		static double previousBestFitness = 0.0;
		static int generationsWithoutImprovement = 0;

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
			return false;

		return true;
	}

	void Population::assignOffspringToTopTwoSpecies()
	{
		// sort the two best species to the beginning of the list
		// species cannot be extinct! 
		std::ranges::sort(speciesList, [](const auto& a, const auto& b) {
			if (a.isExtinct() != b.isExtinct()) {
				return !a.isExtinct(); // Non-extinct species come first
			}
			return a.getChampion()->getFitness() > b.getChampion()->getFitness(); // Sort by fitness
			});

		// Assign offspring count only to the top two **non-extinct** species
		int assigned = 0;
		for (auto& species : speciesList) 
		{
			if (!species.isExtinct()) 
			{
				species.setOffspringCount(parameters.size / 2);
				if (++assigned == 2) break; // Stop after assigning two species
			}
		}
		log(tools::logger::LogLevel::WARNING, "Assigned offspring to top two species.");
	}

	void Population::assignOffspringBasedOnAdjustedFitness()
	{
		double total_adjusted_fitness = 0.0;

		// Step 1: Calculate total adjusted fitness
		for (Species& species : speciesList)
		{
			total_adjusted_fitness += species.totalAdjustedFitness();
		}

		// Step 2: Assign offspring count based on fitness proportion
		const int total_offspring = parameters.size; // Define how many new organisms we want

		double accumulated_offspring = 0.0;
		int assigned_offspring = 0;

		for (Species& species : speciesList)
		{
			if (total_adjusted_fitness > 0)
			{
				species.setOffspringCount((species.totalAdjustedFitness() / total_adjusted_fitness) * total_offspring);
			}
			else
			{
				species.setOffspringCount(0); // Edge case: If total fitness is 0, prevent division error
			}

			// Step 3: Stochastic Rounding
			accumulated_offspring += species.getOffspringCount();
			const int rounded_offspring = static_cast<int>(std::lround(accumulated_offspring));
			species.setOffspringCount(rounded_offspring - assigned_offspring);
			assigned_offspring += species.getOffspringCount();
		}

		// Ensure total assigned offspring matches population_size
		while (assigned_offspring < total_offspring)
		{
			// Assign an extra offspring to the best-performing species
			Species* best_species = nullptr;
			double max_fitness = -1.0;

			for (Species& species : speciesList)
			{
				if (species.totalAdjustedFitness() > max_fitness)
				{
					max_fitness = species.totalAdjustedFitness();
					best_species = &species;
				}
			}

			if (best_species)
			{
				best_species->setOffspringCount(best_species->getOffspringCount() + 1);
				assigned_offspring++;
			}
		}
	}

	void Population::reassignOffspringIfFitnessIsStagnant()
	{
		int totalOffspringToReassign = 0;
		for (auto& species : speciesList)
		{
			if (species.getOffspringCount() == 0)
				continue;

			if (!species.hasFitnessImprovedOverTheLastGenerations())
			{
				totalOffspringToReassign += species.getOffspringCount();
				species.setOffspringCount(0);
			}
		}
		if (totalOffspringToReassign == 0)
			return;
		// give the offspring to the top species
		Species* topSpecies = getBestActiveSpecies();
		topSpecies->setOffspringCount(topSpecies->getOffspringCount() + totalOffspringToReassign);
		log(tools::logger::LogLevel::WARNING, "Reassigned " + std::to_string(totalOffspringToReassign) + " offspring to the top species.");
	}

	void Population::pruneWorsePreformingSolutions()
	{
		// species then reproduce by eliminating the lowest performing members of the population
		for (auto& species : speciesList)
			species.pruneWorsePerformingMembers(PopulationConstants::pruneRatio);
	}

	void Population::replaceEntirePopulationWithOffspring()
	{
		// the entire population is then replaced by the offspring
		// of the remaining organisms in each species

		// the champion of each species with more than five networks
		// is copied into the next generation unchanged

		for (auto& species : speciesList)
		{
			species.crossover(); // creation of offspring
			species.replaceMembersWithOffspring(); // replacement of population with offspring
			if (PopulationConstants::elitism)
				if (species.size() > 5)
					species.copyChampionToNextGeneration(); // elitism
		}
		solutions.clear();
		solutions.reserve(parameters.size);
		for (auto& species : speciesList)
		{
			const auto speciesSolutions = species.getMembers();
			solutions.insert(solutions.end(), speciesSolutions.begin(), speciesSolutions.end());
		}
	}

	void Population::mutate() const
	{
		for (const auto& solution : solutions)
			if (solution != bestSolution)
				solution->mutate();
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
			for (auto& solution : solutions)
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
						log(tools::logger::LogLevel::WARNING, "Best solution is the same as previous best solution.");
					else
						log(tools::logger::LogLevel::ERROR, "Best solution is not the same as previous best solution.");

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
			log(tools::logger::LogLevel::FATAL, "Fitness decreased and previous best solution is not in the population.");
			log(tools::logger::LogLevel::FATAL, "Best solution address: " + addr_bs.str() + " Fitness: " + std::to_string(bsf));
			log(tools::logger::LogLevel::FATAL, "Previous best solution address: " + addr_opbs.str() + " Fitness: " + std::to_string(pbsf));
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
			throw std::runtime_error("Duplicate solutions found.");
		}
	}

	void Population::validatePopulationSize() const
	{
		if (solutions.size() != parameters.size)
		{
			log(tools::logger::LogLevel::FATAL, "Population size does not match parameters.");
			throw std::runtime_error("Population size does not match parameters.");
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
							log(tools::logger::LogLevel::FATAL, "InFieldGeneId: " + std::to_string(inFieldGeneId) +
								" OutFieldGeneId: " + std::to_string(outFieldGeneId) + " InnovationNumber: " + std::to_string(innovationNumber));
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
					continue;

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
				if (species_a.getId() == species_b.getId())
					continue;

				const auto representative_a = species_a.getRepresentative()->getAddress();
				const auto representative_b = species_b.getRepresentative()->getAddress();
				if (representative_a == representative_b)
				{
					log(tools::logger::LogLevel::FATAL, "Species have the same representative.");
				}
			}
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
			result += "Fitness is: " + std::to_string(solution->getFitness()) + "\n";
			const auto genome = solution->getGenome();
			for (const auto& nodeGene : genome.getFieldGenes())
				result += nodeGene.toString();
			for (const auto& connectionGene : genome.getConnectionGenes())
				result += connectionGene.toString();
			result += "\n";
		}
		log(tools::logger::LogLevel::INFO, result);
	}

	void Population::saveAllSolutionsWithFitnessAbove(double fitness) const
	{
		using namespace dnf_composer;
		if (solutions.empty()) return;

		// Construct directory path
		const std::string solution_name = solutions[0]->getName(); // Assuming at least one solution exists
		const auto now = std::time(nullptr);
		const auto localTime = *std::localtime(&now);
		char timeBuffer[100];
		(void)std::strftime(timeBuffer, sizeof(timeBuffer), "%Y~%m~%d_%H'%M'%S", &localTime);

		const std::string directoryPath = std::string(PROJECT_DIR) + "/data/" + solution_name + "/" + timeBuffer + "/";
		std::filesystem::create_directories(directoryPath); // Ensure directories exist

		for (const auto& solution : solutions)
		{
			if (solution->getFitness() > fitness)
			{
				//auto simulation = std::make_shared<Simulation>(solution->getPhenotype());
				auto simulation = solution->getPhenotype();
				// save weights
				for (const auto& element : simulation->getElements())
				{
					if (element->getLabel() == element::ElementLabel::FIELD_COUPLING)
					{
						const auto fieldCoupling = std::dynamic_pointer_cast<element::FieldCoupling>(element);
						fieldCoupling->writeWeights();
					}
				}
				// save elements
				simulation->setUniqueIdentifier(simulation->getUniqueIdentifier() + " " 
					+ std::to_string(solution->getFitness()));
				SimulationFileManager sfm(simulation, directoryPath);
				sfm.saveElementsToJson();
			}
		}
	}

	void Population::resetGenerationalInnovations() const
	{
		bestSolution->clearGenerationalInnovations();
	}

	void Population::resetMutationStatisticsPerGeneration() const
	{
			for (const auto& solution : solutions)
			{
				solution->resetMutationStatisticsPerGeneration();
				break;
			}
	}

	void Population::logSolutions() const
	{
		for (const auto& solution : solutions)
			solution->print();
	}

	void Population::logSpecies() const
	{
		for (const auto& species : speciesList)
			species.print();
	}

	void Population::logOverview() const
	{
		// count active species
		int numActiveSpecies = 0;
		for (const auto& species : speciesList)
		{
			if (species.isExtinct())
				continue;
			numActiveSpecies++;
		}

		tools::logger::log(tools::logger::INFO,
			"Current generation: " + std::to_string(parameters.currentGeneration) +
			" Number of solutions: " + std::to_string(solutions.size()) +
			" Number of active species: " + std::to_string(numActiveSpecies) +
			" Has fitness improved: " + (hasFitnessImproved ? "yes" : "no") +
			" Best solution: [" + bestSolution->toString() + "]");
	}

	void Population::logMutationStatistics() const
	{
		for (const auto& solution : solutions)
		{
			const Genome genome = solution->getGenome();
			const GenomeStatistics gs = genome.getStatistics();
			gs.print();
			const auto fieldGenes = genome.getFieldGenes();
			for (const auto& fieldGene : fieldGenes)
			{
				const auto fgs = fieldGene.getStatistics();
				fgs.print();
				break;
			}
			const auto connectionGenes = genome.getConnectionGenes();
			for (const auto& connectionGene : connectionGenes)
			{
				const auto cgs = connectionGene.getStatistics();
				cgs.print();
				break;
			}
			break;
		}
	}

	void Population::startKeyListenerForUserCommands()
	{
		std::thread keyListener([this]() {
			while (!control.stop)
			{
				if (std::cin.get() == 's')
				{
					control.stop = true;
					tools::logger::log(tools::logger::LogLevel::INFO, "Stopping evolution after the current cycle...");
				}
			}
			});
		keyListener.detach();
	}
}