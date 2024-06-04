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
		do
		{
			evaluate();
			speciate();
			reproduceAndSelect();
			upkeep();

		} while (!endConditionMet());
	}

	void Population::evaluate() const
	{
		for (const auto& solution : solutions)
			solution->evaluate();
	}

	void Population::speciate()
	{
		for (const auto& solution : solutions)
			assignToSpecies(solution);
	}

	void Population::reproduceAndSelect()
	{
		calculateAdjustedFitness();
		for (auto& species : speciesList)
			species.selectElitesAndLeastFit();
		std::vector<SolutionPtr> elites = selectElites();
		const std::vector<SolutionPtr> lessFit = selectLessFit();
		calculateSpeciesOffspring(elites.size(), lessFit.size());
		for (auto& species : speciesList)
			species.crossover();
		std::vector<SolutionPtr> offspring = reproduce();
		for (const auto& solution : offspring)
			solution->mutate();
		for (const auto& solution : offspring)
			solution->clearGenerationalInnovations();

		solutions.clear();
		solutions.insert(solutions.end(), elites.begin(), elites.end());
		solutions.insert(solutions.end(), offspring.begin(), offspring.end());
	}

	std::vector<SolutionPtr> Population::selectElites() const
	{
		std::vector<SolutionPtr> elites;
		for (auto& species : speciesList)
		{
			const auto speciesElites = species.getElites();
			elites.insert(elites.end(), speciesElites.begin(), speciesElites.end());
		}
		return elites;
	}

	std::vector<SolutionPtr> Population::selectLessFit() const
	{
		std::vector<SolutionPtr> lessFit;
		for (auto& species : speciesList)
		{
			const auto speciesLessFit = species.getLeastFit();
			lessFit.insert(lessFit.end(), speciesLessFit.begin(), speciesLessFit.end());
		}
		return lessFit;
	}

	std::vector<SolutionPtr> Population::reproduce() const
	{

		std::vector<SolutionPtr> offspring;
		for (auto& species : speciesList)
		{
			const auto speciesOffspring = species.getOffspring();
			offspring.insert(offspring.end(), speciesOffspring.begin(), speciesOffspring.end());
		}
		return offspring;
	}

	void Population::upkeep()
	{
		upkeepBestSolution();
		updateGenerationAndAges();
		if (PopulationConstants::validatePopulationSize)
			validatePopulationSize();
		if (PopulationConstants::validateUniqueSolutions)
			validateUniqueSolutions();
		if (PopulationConstants::validateElitism)
			validateElitism();

		std::stringstream addr_bs;
		addr_bs << bestSolution.get();

		tools::logger::log(tools::logger::INFO,
			"Current generation: " + std::to_string(parameters.currentGeneration) +
			" Best solution address: " + addr_bs.str() +
			" Best fitness: " + std::to_string(bestSolution->getFitness()) +
			" Adjusted fitness: " + std::to_string(bestSolution->getParameters().adjustedFitness) +
			" Number of species: " + std::to_string(speciesList.size()) +
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
			solution->setAdjustedFitness(adjustedFitness);
		}
	}

	void Population::calculateSpeciesOffspring(const size_t eliteCount, const size_t killCount)
	{
		const auto offspringPoolSize = static_cast<uint16_t>(killCount);

		double totalAdjustedFitnessAcrossSpecies = 0.0;
		for (const auto& species : speciesList)
			totalAdjustedFitnessAcrossSpecies += species.totalAdjustedFitness();

		for (auto& species : speciesList)
		{
			if (totalAdjustedFitnessAcrossSpecies == 0.0)
				species.setOffspringCount(offspringPoolSize);
			else
			{
				const double speciesProportion =
					species.totalAdjustedFitness() / totalAdjustedFitnessAcrossSpecies;
				const auto exactOffspring = static_cast<uint16_t>(speciesProportion * static_cast<double>(offspringPoolSize));
				species.setOffspringCount(exactOffspring);
			}
		}

		const int totalOffspringAcrossSpecies =
			std::accumulate(speciesList.begin(), speciesList.end(), 0,
				[](int sum, const Species& species) {
					return sum + species.getOffspringCount();
				});

		const uint16_t newPopulationSize = eliteCount + totalOffspringAcrossSpecies;
		int error = parameters.size - newPopulationSize;
		while (error != 0)
		{
			const int randomIndex = tools::utils::generateRandomInt(0, static_cast<int>(speciesList.size() - 1));
			if (error > 0 && speciesList[randomIndex].getOffspringCount() > 0)
			{
				speciesList[randomIndex].setOffspringCount(speciesList[randomIndex].getOffspringCount() + 1);
				error--;
			}
			else if (error < 0)
			{
				speciesList[randomIndex].setOffspringCount(speciesList[randomIndex].getOffspringCount() - 1);
				error++;
			}
		}
	}

	bool Population::endConditionMet() const
	{
		const bool fitnessCondition = bestSolution->getFitness() > parameters.targetFitness;
		const bool generationCondition = parameters.currentGeneration >= parameters.numGenerations;
		return fitnessCondition || generationCondition;
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

					log(tools::logger::LogLevel::WARNING, "Fitness decreased but previous best solution is in the population.");
					log(tools::logger::LogLevel::WARNING, "Best solution address: " + addr_bs.str() + " Fitness: " + std::to_string(bsf));
					log(tools::logger::LogLevel::WARNING, "New previous best solution address: " + addr_npbs.str() + " Fitness: " + std::to_string(npbsf));
					log(tools::logger::LogLevel::WARNING, "Old previous best solution address: " + addr_opbs.str() + " Fitness: " + std::to_string(opbsf));

					pbsInPopulation = true;
					break;
				}
			}
		}

		if (bsDecreased && !pbsInPopulation)
		{
			log(tools::logger::LogLevel::FATAL, "Fitness decreased and previous best solution is not in the population.");
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
		if(counter>0)
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

}