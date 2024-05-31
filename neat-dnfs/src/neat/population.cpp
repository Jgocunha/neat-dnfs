#include "neat/population.h"

namespace neat_dnfs
{
	Population::Population(const PopulationParameters& parameters, const SolutionPtr& initialSolution)
		: parameters(parameters), bestSolution(nullptr)
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
			int counter = validateUniqueSolutions(solutions);
			if (counter > 0)
				log(tools::logger::LogLevel::WARNING, "Number of duplicate solutions: " + std::to_string(counter));
			speciate();
			counter = validateUniqueSolutions(solutions);
			if (counter > 0)
				log(tools::logger::LogLevel::WARNING, "Number of duplicate solutions: " + std::to_string(counter));
			reproduceAndSelect();
			counter = validateUniqueSolutions(solutions);
			if (counter > 0)
				log(tools::logger::LogLevel::WARNING, "Number of duplicate solutions: " + std::to_string(counter));
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
		std::vector<SolutionPtr> elites;
		std::vector<SolutionPtr> lessFit;
		std::vector<SolutionPtr> offspring;

		calculateAdjustedFitness();
		for (auto& species : speciesList)
			species.selectElitesAndLeastFit();
		elites = selectElites();
		int counter = validateUniqueSolutions(elites);
		if (counter > 0)
			log(tools::logger::LogLevel::WARNING, "Number of duplicate elites: " + std::to_string(counter));
		lessFit = selectLessFit();
		counter = validateUniqueSolutions(lessFit);
		if (counter > 0)
			log(tools::logger::LogLevel::WARNING, "Number of duplicate less fit: " + std::to_string(counter));
		calculateSpeciesOffspring(elites.size(), lessFit.size());
		for (auto& species : speciesList)
			species.crossover();
		offspring = reproduce();
		counter = validateUniqueSolutions(offspring);
		if (counter > 0)
			log(tools::logger::LogLevel::WARNING, "Number of duplicate offspring: " + std::to_string(counter));

		for (const auto& solution : offspring)
			solution->mutate();
		for (const auto& solution : offspring)
			solution->clearGenerationalInnovations();

		solutions.clear();
		solutions.insert(solutions.end(), elites.begin(), elites.end());
		solutions.insert(solutions.end(), offspring.begin(), offspring.end());
		assert(solutions.size() == parameters.size);
		counter = validateUniqueSolutions(solutions);
		if (counter > 0)
			log(tools::logger::LogLevel::WARNING, "Number of duplicate solutions: " + std::to_string(counter));
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
			int counter = validateUniqueSolutions(speciesOffspring);
			if (counter > 0)
			{
				int numSpecies = speciesList.size();
				log(tools::logger::LogLevel::INFO, "Number of species: " + std::to_string(numSpecies));
				log(tools::logger::LogLevel::WARNING, "Number of duplicate species offspring: " + std::to_string(counter));
			}
			offspring.insert(offspring.end(), speciesOffspring.begin(), speciesOffspring.end());
		}
		int counter = validateUniqueSolutions(offspring);
		if (counter > 0)
			log(tools::logger::LogLevel::WARNING, "Number of duplicate offspring: " + std::to_string(counter));
		return offspring;
	}

	void Population::upkeep()
	{
		//evaluate();
		upkeepBestSolution();
		updateGenerationAndAges();
		validateElitism();

		int counter = validateUniqueSolutions(solutions);
		if (counter > 0)
			log(tools::logger::LogLevel::WARNING, "Number of duplicate solutions: " + std::to_string(counter));
		tools::logger::log(tools::logger::INFO,
			"Current generation: " + std::to_string(parameters.currentGeneration) +
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
		//for (auto& species : speciesList)
			//offspringPoolSize += species.getLeastFit().size();

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

		//log(tools::logger::LogLevel::INFO, "Offspring pool size adjusted. Total offspring: " + std::to_string(totalOffspringAcrossSpecies) +
		//	" Offspring pool size: " + std::to_string(offspringPoolSize));
	}

	bool Population::endConditionMet() const
	{
		const bool fitnessCondition = bestSolution->getFitness() > parameters.targetFitness;
		const bool generationCondition = parameters.currentGeneration > parameters.numGenerations;
		return fitnessCondition || generationCondition;
	}

	void Population::validateElitism() const
	{
		static SolutionPtr prevBestSolution = nullptr;
		static SolutionPtr pbsclone = nullptr;
		static double prevBestSolutionFitness = 0.0;

		if (bestSolution == nullptr)
			return;

		if (bestSolution->getFitness() < prevBestSolutionFitness)
		{
			log(tools::logger::LogLevel::FATAL, "Elitism failed. Best solution fitness decreased.");
			// is previous best solution still in the population?
			for (auto& solution : solutions)
			{
				if (solution == prevBestSolution)
				{
					SolutionPtr pbs = prevBestSolution;
					SolutionPtr cpbs = solution;
					SolutionPtr bs = bestSolution;
					log(tools::logger::LogLevel::WARNING, "Previous best solution is still in the population.");
					double fitness = solution->getFitness();
					log(tools::logger::LogLevel::WARNING, "*old* Previous best solution fitness: " + std::to_string(prevBestSolutionFitness));
					log(tools::logger::LogLevel::WARNING, "*new* Previous best solution fitness: " + std::to_string(fitness));
					log(tools::logger::LogLevel::WARNING, "Current best solution fitness: " + std::to_string(bestSolution->getFitness()));
					break;
				}
			}
		}

		prevBestSolution = bestSolution;
		pbsclone = bestSolution->clone();
		prevBestSolutionFitness = bestSolution->getFitness();
	}

	/*void Population::validateUniqueSolutions() const
	{
		for (size_t i = 0; i < solutions.size(); ++i)
		{
			for (size_t j = i + 1; j < solutions.size(); ++j)
			{
				if (solutions[i] == solutions[j])
					log(tools::logger::LogLevel::FATAL, "Duplicate solutions found in the population.");
			}
		}
	}*/

	int Population::validateUniqueSolutions(const std::vector<SolutionPtr>& solutions)
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
		return counter;
	}

}