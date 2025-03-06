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

	void Population::evolve()
	{
		statistics.start = std::chrono::high_resolution_clock::now();
		setFileDirectory();
		startKeyListenerForUserCommands();

		do
		{
			evaluate();
			speciate();
			upkeep();
			//saveBestSolutionOfEachGeneration();
			//saveChampionsOfEachGeneration();
			reproduceAndSelect();

			while (control.pause)
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
				tools::logger::log(tools::logger::LogLevel::INFO, "Evolution paused.");
			}

		} while (!endConditionMet());

		statistics.end = std::chrono::high_resolution_clock::now();
		statistics.duration = std::chrono::duration_cast<std::chrono::seconds>(statistics.end - statistics.start).count();

		//saveAllSolutionsWithFitnessAbove(bestSolution->getFitness() - 0.1);
		saveTimestampsAndDuration();
		if (PopulationConstants::saveStatistics)
			saveFinalStatistics();
		//if (PopulationConstants::saveChampions)
			//saveChampions();
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
		for (auto& species : speciesList)
			species->assignChampion();

		calculateAdjustedFitness();
	}

	void Population::reproduceAndSelect()
	{
		assignOffspringToSpecies();

		if (PopulationConstants::logSpecies)
			logSpecies();

		pruneWorsePreformingSolutions();
		replaceEntirePopulationWithOffspring();
		mutate();
	}

	void Population::upkeep()
	{
		upkeepBestSolution();
		upkeepChampions();

		if (PopulationConstants::logSolutions)
			logSolutions();
		if (PopulationConstants::logOverview)
			logOverview();
		/*if (PopulationConstants::logSpecies)
			logSpecies();*/
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

		if (PopulationConstants::saveStatistics)
			savePerGenerationStatistics();

		resetGenerationalInnovations();
		updateGenerationAndAges();
		resetMutationStatisticsPerGeneration();
	}

	void Population::createInitialEmptySolutions(const SolutionPtr& initialSolution)
	{
		for (int i = 0; i < parameters.size; i++)
			solutions.emplace_back(initialSolution->clone());
	}

	void Population::buildInitialSolutionsGenome() const
	{
		for (const auto& solution : solutions)
			solution->initialize();
	}

	void Population::upkeepBestSolution()
	{
		bestSolution = nullptr;
		for (const auto& solution : solutions)
		{
			if (bestSolution == nullptr || solution->getFitness() > bestSolution->getFitness())
				bestSolution = solution;
		}
	}

	void Population::upkeepChampions()
	{
		champions.clear();
		for (const auto& species : speciesList)
			champions.emplace_back(species->getChampion());
	}

	void Population::updateGenerationAndAges()
	{
		parameters.currentGeneration++;
		for (const auto& solution : solutions)
			solution->incrementAge();
		for (const auto& species : speciesList)
			if (!species->isExtinct())
				species->incrementAge();
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
							currentSpecies->removeSolution(solution);
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
				currentSpecies->removeSolution(solution);

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
			if (species->contains(solution))
				return species;
		return nullptr;
	}

	std::shared_ptr<Species> Population::getBestActiveSpecies() const
	{
		std::shared_ptr<Species> bestSpecies = nullptr;
		double bestFitness = 0.0;
		for (const auto& species : speciesList)
		{
			if (species->isExtinct())
				continue;
			if (species->getChampion()->getFitness() > bestFitness)
			{
				bestFitness = species->getChampion()->getFitness();
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
		const int numActiveSpecies = std::ranges::count_if(speciesList.begin(), speciesList.end(), [](const auto& species)
			{ return !species->isExtinct(); });
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
			species->setOffspringCount(0);
	}

	bool Population::hasFitnessImprovedOverTheLastGenerations()
	{
		static double previousBestFitness = 0.0;

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
		sortSpeciesListByChampionFitness();

		// assign offspring count only to the top two **non-extinct** species
		int assigned = 0;
		for (auto& species : speciesList) 
		{
			if (!species->isExtinct()) 
			{
				species->setOffspringCount(parameters.size / 2);
				if (++assigned == 2) break; // Stop after assigning two species
			}
		}
		log(tools::logger::LogLevel::WARNING, "Fitness of entire population has not improved for the last " 
			+ std::to_string(PopulationConstants::generationsWithoutImprovementThresholdInPopulation) + " generations. Assigned offspring to top two species.");
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
				species_ptr->setOffspringCount((species_ptr->totalAdjustedFitness() / total_adjusted_fitness) * total_offspring);
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
				continue;

			if (!species->hasFitnessImprovedOverTheLastGenerations())
			{
				totalOffspringToReassign += species->getOffspringCount();
				species->setOffspringCount(0);
				log(tools::logger::LogLevel::WARNING, "Fitness of species " + std::to_string(species->getId()) + " has not improved for the last " + std::to_string(PopulationConstants::generationsWithoutImprovementThresholdInSpecies) + " generations.");
			}
		}
		if (totalOffspringToReassign == 0)
			return;
		// give the offspring to the top species
		const std::shared_ptr<Species> topSpecies = getBestActiveSpecies();
		topSpecies->setOffspringCount(topSpecies->getOffspringCount() + totalOffspringToReassign);
		log(tools::logger::LogLevel::WARNING, "Reassigned " + std::to_string(totalOffspringToReassign) + " offspring to species " + std::to_string(topSpecies->getId()) + ".");
	}

	void Population::pruneWorsePreformingSolutions() const
	{
		// species then reproduce by eliminating the lowest performing members of the population
		for (const auto& species : speciesList)
			species->pruneWorsePerformingMembers(PopulationConstants::pruneRatio);
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
				if (species->size() > 5)
					species->copyChampionToNextGeneration(); // elitism
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
			// if champion, do not mutate
			if (solution != bestSolution && !std::ranges::any_of(champions, [&solution](const auto& champion) { return champion == solution; }))
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
						log(tools::logger::LogLevel::WARNING, "Best solution is not the same as previous best solution.");

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
			log(tools::logger::LogLevel::WARNING, "Best solution address: " + addr_bs.str() + " Fitness: " + std::to_string(bsf));
			log(tools::logger::LogLevel::WARNING, "Previous best solution address: " + addr_opbs.str() + " Fitness: " + std::to_string(pbsf));
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
				if (species_a->getId() == species_b->getId())
					continue;
				if (species_a->isExtinct() || species_b->isExtinct())
					continue;

				const auto representative_a = species_a->getRepresentative()->getAddress();
				const auto representative_b = species_b->getRepresentative()->getAddress();

				if (representative_a == representative_b)
				{
					log(tools::logger::LogLevel::FATAL, "Species have the same representative.");
					log(tools::logger::LogLevel::FATAL, "Species a id: " + std::to_string(species_a->getId()) + " Representative a id: " + representative_a);
					log(tools::logger::LogLevel::FATAL, "Species b id: " + std::to_string(species_b->getId()) + " Representative b id: " + representative_b);
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

	void Population::setFileDirectory()
	{
		using namespace dnf_composer;
		if (solutions.empty()) throw std::runtime_error("No solutions in population.");

		const std::string solutionName = solutions[0]->getName();
		const auto now = std::time(nullptr);
		const auto localTime = *std::localtime(&now);
		char timeBuffer[100];
		(void)std::strftime(timeBuffer, sizeof(timeBuffer), "%Y~%m~%d_%H'%M'%S", &localTime);

		fileDirectory = std::string(PROJECT_DIR) + "/data/" + solutionName + "/" + timeBuffer + "/";
		std::filesystem::create_directories(fileDirectory); // Ensure directory exist
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

	//void Population::saveAllSolutionsWithFitnessAbove(double fitness) const
	//{
	//	using namespace dnf_composer;

	//	const std::string directoryPath = fileDirectory + "solutions/last_generation/";
	//	std::filesystem::create_directories(directoryPath); // Ensure directory exist

	//	for (const auto& solution : solutions)
	//	{
	//		if (solution->getFitness() > fitness)
	//		{
	//			auto simulation = solution->getPhenotype();
	//			// save weights
	//			for (const auto& element : simulation->getElements())
	//			{
	//				if (element->getLabel() == element::ElementLabel::FIELD_COUPLING)
	//				{
	//					const auto fieldCoupling = std::dynamic_pointer_cast<element::FieldCoupling>(element);
	//					fieldCoupling->writeWeights();
	//				}
	//			}
	//			// save elements
	//			const std::string uniqueIdentifier = "solution " + std::to_string(solution->getId())
	//				+ " generation " + std::to_string(parameters.currentGeneration)
	//				+ " species " + std::to_string(solution->getSpeciesId())
	//				+ " fitness " + std::to_string(solution->getFitness());
	//			simulation->setUniqueIdentifier(uniqueIdentifier);
	//			SimulationFileManager sfm(simulation, directoryPath);
	//			sfm.saveElementsToJson();
	//		}
	//	}
	//}

	//void Population::saveChampions() const
	//{
	//	using namespace dnf_composer;

	//	const std::string directoryPath = fileDirectory + "champions/last_generation/";
	//	std::filesystem::create_directories(directoryPath); // Ensure directory exist

	//	if (champions.empty()) log(tools::logger::LogLevel::ERROR, "No champions to save.");

	//	for (const auto& champion : champions)
	//	{
	//		if (champion == nullptr)
	//		{
	//			continue;
	//		}
	//		auto simulation = champion->getPhenotype();
	//		// save weights
	//		for (const auto& element : simulation->getElements())
	//		{
	//			if (element->getLabel() == element::ElementLabel::FIELD_COUPLING)
	//							{
	//				const auto fieldCoupling = std::dynamic_pointer_cast<element::FieldCoupling>(element);
	//				fieldCoupling->writeWeights();
	//			}
	//		}
	//		// save elements
	//		const std::string uniqueIdentifier = "solution " + std::to_string(champion->getId())
	//			+ " generation " + std::to_string(parameters.currentGeneration)
	//			+ " species " + std::to_string(champion->getSpeciesId())
	//			+ " fitness " + std::to_string(champion->getFitness());
	//		simulation->setUniqueIdentifier(uniqueIdentifier);
	//		SimulationFileManager sfm(simulation, directoryPath);
	//		sfm.saveElementsToJson();
	//	}
	//}

	void Population::saveTimestampsAndDuration() const
	{
		const std::string directoryPath = fileDirectory + "statistics/";
		std::filesystem::create_directories(directoryPath); // Ensure directory exists

		std::ofstream logFile(directoryPath + "evolution_timestamps.txt", std::ios::app);
		if (logFile.is_open())
		{
			// Convert steady_clock timestamps to system_clock timestamps
			const auto system_start = std::chrono::system_clock::now() +
				std::chrono::duration_cast<std::chrono::system_clock::duration>(
					statistics.start - std::chrono::steady_clock::now());

			const auto system_end = std::chrono::system_clock::now() +
				std::chrono::duration_cast<std::chrono::system_clock::duration>(
					statistics.end - std::chrono::steady_clock::now());

			// Convert to time_t for formatting
			const std::time_t start_time_t = std::chrono::system_clock::to_time_t(system_start);
			const std::time_t end_time_t = std::chrono::system_clock::to_time_t(system_end);

			// Log number of generations
			logFile << "Number of generations: " << parameters.currentGeneration << "\n";
			// Format and write timestamps
			logFile << "Evolution Start Time: " << std::put_time(std::localtime(&start_time_t), "%Y-%m-%d %H:%M:%S") << "\n";
			logFile << "Evolution End Time: " << std::put_time(std::localtime(&end_time_t), "%Y-%m-%d %H:%M:%S") << "\n";
			logFile << "Duration (seconds): " << statistics.duration << "\n";
			logFile << "Duration (minutes): " << statistics.duration / 60 << "\n";
			logFile << "Duration (hours): " << statistics.duration / 3600 << "\n";

			logFile.close();
		}
		else
		{
			tools::logger::log(tools::logger::LogLevel::ERROR, "Failed to open log file for timestamps.");
		}
	}

	void Population::saveFinalStatistics() const
	{
		const std::string directoryPath = fileDirectory + "statistics/";
		std::filesystem::create_directories(directoryPath); // Ensure directory exists

		for (const auto& solution : solutions)
		{
			const Genome genome = solution->getGenome();
			const GenomeStatistics gs = genome.getStatistics();
			gs.saveTotal(directoryPath);
			const auto fieldGenes = genome.getFieldGenes();
			for (const auto& fieldGene : fieldGenes)
			{
				const auto fgs = fieldGene.getStatistics();
				fgs.saveTotal(directoryPath);
				break;
			}
			const auto connectionGenes = genome.getConnectionGenes();
			for (const auto& connectionGene : connectionGenes)
			{
				const auto cgs = connectionGene.getStatistics();
				cgs.saveTotal(directoryPath);
				break;
			}
			break;
		}
	}

	void Population::savePerGenerationStatistics() const
	{
		const std::string directoryPath = fileDirectory + "statistics/";
		std::filesystem::create_directories(directoryPath); // Ensure directory exists

		// count active species
		int numActiveSpecies = 0;
		for (const auto& species : speciesList)
		{
			if (species->isExtinct())
				continue;
			numActiveSpecies++;
		}

		std::ofstream logFile(directoryPath + "per_generation_overview.txt", std::ios::app);
		if (logFile.is_open())
		{
			logFile << "Current generation: " + std::to_string(parameters.currentGeneration);
			logFile << " Number of solutions: " + std::to_string(solutions.size());
			logFile << " Number of active species: " + std::to_string(numActiveSpecies);
			logFile << " Has fitness improved: " << (hasFitnessImproved ? "yes" : "no");
			logFile << " Number of generations without improvement: " + std::to_string(generationsWithoutImprovement);
			logFile << " Best solution: [" + bestSolution->toString() + "]";
			logFile << "\n";
			logFile.close();
		}
		else
		{
			tools::logger::log(tools::logger::LogLevel::ERROR, "Failed to open log file for field gene per generation statistics.");
		}


		// save genome and connection gene per generation statistics
		for (const auto& solution : solutions)
		{
			const Genome genome = solution->getGenome();
			const GenomeStatistics gs = genome.getStatistics();
			gs.savePerGeneration(directoryPath);
			const auto fieldGenes = genome.getFieldGenes();
			for (const auto& fieldGene : fieldGenes)
			{
				const auto fgs = fieldGene.getStatistics();
				fgs.savePerGeneration(directoryPath);
				break;
			}
			const auto connectionGenes = genome.getConnectionGenes();
			for (const auto& connectionGene : connectionGenes)
			{
				const auto cgs = connectionGene.getStatistics();
				cgs.savePerGeneration(directoryPath);
				break;
			}
			break;
		}
	}

	//void Population::saveBestSolutionOfEachGeneration() const
	//{
	//	using namespace dnf_composer;

	//	const std::string directoryPath = fileDirectory + "solutions/prev_generations/";
	//	std::filesystem::create_directories(directoryPath); // Ensure directory exist

	//	const auto simulation = bestSolution->getPhenotype();
	//	// save weights
	//	for (const auto& element : simulation->getElements())
	//	{
	//		if (element->getLabel() == element::ElementLabel::FIELD_COUPLING)
	//		{
	//			const auto fieldCoupling = std::dynamic_pointer_cast<element::FieldCoupling>(element);
	//			fieldCoupling->writeWeights();
	//		}
	//	}
	//	// save elements
	//	const std::string uniqueIdentifier = "solution " + std::to_string(bestSolution->getId())
	//		+ " generation " + std::to_string(parameters.currentGeneration)
	//		+ " species " + std::to_string(bestSolution->getSpeciesId())
	//		+ " fitness " + std::to_string(bestSolution->getFitness());
	//	simulation->setUniqueIdentifier(uniqueIdentifier);
	//	const SimulationFileManager sfm(simulation, directoryPath);
	//	sfm.saveElementsToJson();
	//}

	//void Population::saveChampionsOfEachGeneration() const
	//{
	//	using namespace dnf_composer;

	//	const std::string directoryPath = fileDirectory + "champions/prev_generations/";
	//	std::filesystem::create_directories(directoryPath); // Ensure directory exist

	//	for (const auto& champion : champions)
	//	{
	//		if (champion == nullptr)
	//		{
	//			continue;
	//		}
	//		auto simulation = champion->getPhenotype();
	//		// save weights
	//		for (const auto& element : simulation->getElements())
	//		{
	//			if (element->getLabel() == element::ElementLabel::FIELD_COUPLING)
	//			{
	//				const auto fieldCoupling = std::dynamic_pointer_cast<element::FieldCoupling>(element);
	//				fieldCoupling->writeWeights();
	//			}
	//		}
	//		// save elements
	//		const std::string uniqueIdentifier = "solution " + std::to_string(champion->getId())
	//			+ " generation " + std::to_string(parameters.currentGeneration)
	//			+ " species " + std::to_string(champion->getSpeciesId())
	//			+ " fitness " + std::to_string(champion->getFitness());
	//		simulation->setUniqueIdentifier(uniqueIdentifier);
	//		const SimulationFileManager sfm(simulation, directoryPath);
	//		sfm.saveElementsToJson();
	//	}
	//}

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
			species->print();
	}

	void Population::logOverview() const
	{
		// count active species
		int numActiveSpecies = 0;
		for (const auto& species : speciesList)
		{
			if (species->isExtinct())
				continue;
			numActiveSpecies++;
		}

		tools::logger::log(tools::logger::INFO,
			"Current generation: " + std::to_string(parameters.currentGeneration) +
			" Number of solutions: " + std::to_string(solutions.size()) +
			" Number of active species: " + std::to_string(numActiveSpecies) +
			" Has fitness improved: " + (hasFitnessImproved ? "yes" : "no") +
			" Number of generations without improvement: " + std::to_string(generationsWithoutImprovement));// +
			//" Best solution: [" + bestSolution->toString() + "]");
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