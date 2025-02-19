#include "neat/population.h"

namespace neat_dnfs
{
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

		do
		{
			evaluate();
			speciate();

			//for (const auto& solution : solutions)
			//{
			//	std::vector<ConnectionGene> connectionGenes = solution->getGenome().getConnectionGenes();
			//	std::vector<FieldGene> fieldGenes = solution->getGenome().getFieldGenes();
			//	const int species = findSpecies(solution)->getId();

			//	std::string result = "sol." + std::to_string(solution->getId());
			//	result += " fit." + std::to_string(solution->getFitness());
			//	result += " adj." + std::to_string(solution->getParameters().adjustedFitness);
			//	result += " spec." + std::to_string(species) + " { ";
			//	for (const auto& fieldGene : fieldGenes)
			//	{
			//		result += "fg [" + std::to_string(fieldGene.getParameters().id) + "] ";
			//	}
			//	result += "} { ";
			//	for (const auto& connectionGene : connectionGenes)
			//	{
			//		result += "cg [innov: " + std::to_string(connectionGene.getInnovationNumber()) + " ";
			//		result += ", tuple (" + std::to_string(connectionGene.getInFieldGeneId()) + "-" + std::to_string(connectionGene.getOutFieldGeneId()) + ") ] ";
			//	}
			//	result += "}";
			//	std::cout << result << std::endl;
			//}

			upkeep();
			reproduceAndSelect();

			while (control.pause)
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
				tools::logger::log(tools::logger::LogLevel::INFO, "Evolution paused.");
			}

		} while (!endConditionMet());

		saveAllSolutionsWithFitnessAbove(parameters.targetFitness - 0.1);

		keyListener.detach();
	}

	void Population::evolutionaryStep()
	{
		evaluate();
		speciate();
		reproduceAndSelect();
		upkeep();

		if(endConditionMet())
			saveAllSolutionsWithFitnessAbove(parameters.targetFitness - 0.2);
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
		for (auto& species : speciesList)
			if (!species.isExtinct())
				std::cout << "Species: " << species.getId() << " size: " << species.size() << " offspring: " << species.getOffspringCount() << std::endl;
	}

	void Population::upkeep()
	{
		upkeepBestSolution();
		bestSolution->clearGenerationalInnovations();
		updateGenerationAndAges();
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

		std::stringstream addr_bs;
		addr_bs << bestSolution.get();

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
			" Best solution address: " + addr_bs.str() +
			" Best fitness: " + std::to_string(bestSolution->getFitness()) +
			" Adjusted fitness: " + std::to_string(bestSolution->getParameters().adjustedFitness) +
			" Number of active species: " + std::to_string(numActiveSpecies) +
			" Number of solutions: " + std::to_string(solutions.size())
		);
	}

	SolutionPtr Population::getBestSolution() const
	{
		return bestSolution;
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
				}
				assigned = true;
				break;
			}
		}
		if (!assigned)
		{
			if (currentSpecies != nullptr)
				currentSpecies->removeSolution(solution);
			Species newSpecies;
			newSpecies.addSolution(solution);
			newSpecies.setRepresentative(solution);
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
		// every species is assigned a potentially different number of offspring
		// in proportion to the sum of adjusted fitness of its members fitness

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
			// casting (double + 0.5) to integer leads to incorrect rounding
			// use lround instead
			const int rounded_offspring = static_cast<int>(accumulated_offspring + 0.5);
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
		for (auto& species : speciesList)
		{
			species.crossover(); // creation of offspring
			species.replaceMembersWithOffspring(); // replacement of population
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

		const bool bsDecreased = bsf < pbsf;
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

					log(tools::logger::LogLevel::FATAL, "Fitness decreased but previous best solution is in the population.");

					if (bs == pbs)
						log(tools::logger::LogLevel::FATAL, "Best solution is the same as previous best solution.");
					else
						log(tools::logger::LogLevel::FATAL, "Best solution is not the same as previous best solution.");

					log(tools::logger::LogLevel::FATAL, "Best solution address: " + addr_bs.str() + " Fitness: " + std::to_string(bsf));
					log(tools::logger::LogLevel::FATAL, "New previous best solution address: " + addr_npbs.str() + " Fitness: " + std::to_string(npbsf));
					log(tools::logger::LogLevel::FATAL, "Old previous best solution address: " + addr_opbs.str() + " Fitness: " + std::to_string(opbsf));

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
}