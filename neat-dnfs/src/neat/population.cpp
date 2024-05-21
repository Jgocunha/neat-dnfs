#include "neat/population.h"

namespace neat_dnfs
{
	Population::Population(const PopulationParameters& parameters, const SolutionPtr& initialSolution)
		: parameters(parameters)
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
			select();
			reproduce();
			upkeepBestSolution();
			updateGenerationAndAges();
			tools::logger::log(tools::logger::INFO,
				"Current generation: " + std::to_string(parameters.currentGeneration) +
				" Best fitness: " + std::to_string(bestSolution->getFitness()) +
				" Adjusted fitness: " + std::to_string(bestSolution->getParameters().adjustedFitness) +
				" Number of species: " + std::to_string(speciesList.size()) +
				" Number of solutions: " + std::to_string(solutions.size())
			);

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

	void Population::select()
	{
		calculateAdjustedFitness();
		calculateSpeciesOffspring();
		killLeastFitSolutions();
	}

	void Population::reproduce()
	{
		crossover();
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
		for (const auto& species : speciesList)
		{
			for (const auto& solution : species.getMembers())
			{
				if (bestSolution == nullptr || solution->getFitness() > bestSolution->getFitness())
					bestSolution = solution;
			}
		}

		/*for (const auto& solution : solutions)
		{
			if (bestSolution == nullptr || solution->getFitness() > bestSolution->getFitness())
				bestSolution = solution;
		}*/
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

	void Population::calculateSpeciesOffspring()
	{
		static const int offspringPoolSize =
			static_cast<int>(parameters.size * PopulationConstants::killRatio);

		double totalAdjustedFitnessAcrossSpecies = 0.0;
		for (const auto& species : speciesList)
			totalAdjustedFitnessAcrossSpecies += species.totalAdjustedFitness();

		for (auto& species : speciesList)
		{
			const double speciesProportion =
				species.totalAdjustedFitness() / totalAdjustedFitnessAcrossSpecies;
			const auto exactOffspring = static_cast<uint16_t>(speciesProportion * offspringPoolSize);
			species.setOffspringCount(exactOffspring);
		}

		// Adjust offspring count to match the offspring pool size
		const int totalOffspringAcrossSpecies =
			std::accumulate(speciesList.begin(), speciesList.end(), 0,
				[](int sum, const Species& species) {
					return sum + species.getOffspringCount();
				});

		int totalIndividualsKilled = 0;
		for (const auto& species : speciesList)
			totalIndividualsKilled += species.getKillCount();
		const int totalIndividualsRemaining = static_cast<uint16_t>(solutions.size()) - totalIndividualsKilled;

		const int totalOffspringAvailable = static_cast<uint16_t>(solutions.size()) - totalIndividualsRemaining;
		int error = totalOffspringAcrossSpecies - totalOffspringAvailable;
		while (error != 0)
		{
			const int randomIndex = tools::utils::generateRandomInt(0, static_cast<int>(speciesList.size() - 1));
			if (error > 0 && speciesList[randomIndex].getOffspringCount() > 0)
			{
				speciesList[randomIndex].setOffspringCount(speciesList[randomIndex].getOffspringCount() - 1);
				error--;
			}
			else if (error < 0)
			{
				speciesList[randomIndex].setOffspringCount(speciesList[randomIndex].getOffspringCount() + 1);
				error++;
			}
		}

		log(tools::logger::LogLevel::INFO, "Offspring pool size adjusted. Total offspring: " + std::to_string(totalOffspringAcrossSpecies) +
			" Offspring pool size: " + std::to_string(offspringPoolSize));
	}

	void Population::killLeastFitSolutions()
	{
		std::vector<SolutionPtr> toRemove;

		// Gather all solutions that need to be removed from each species
		for (auto& species : speciesList)
		{
			std::vector<SolutionPtr> removed = species.killLeastFitSolutions();
			toRemove.insert(toRemove.end(), removed.begin(), removed.end());
		}

		// Remove dead solutions from the main population vector
		// // use ranges remove if instead of remove_if
		const auto newEnd = std::remove_if(solutions.begin(), solutions.end(),
			[&toRemove](const SolutionPtr& solution)
			{
				// use ranges find instead of find
				return std::find(toRemove.begin(), toRemove.end(), solution) != toRemove.end();
			});
		solutions.erase(newEnd, solutions.end());
	}

	void Population::crossover()
	{
		for (auto& species : speciesList)
			species.crossover();
		// Add offspring to the population
		for (auto& species : speciesList)
		{
			const auto offspring = species.getOffspring();
			for (const auto& solution : offspring)
				solution->mutate();
			for (const auto& solution : offspring)
				solution->clearGenerationalInnovations();
			solutions.insert(solutions.end(), offspring.begin(), offspring.end());
		}
	}

	bool Population::endConditionMet() const
	{
		const bool fitnessCondition = bestSolution->getFitness() > parameters.targetFitness;
		const bool generationCondition = parameters.currentGeneration > parameters.numGenerations;
		return fitnessCondition || generationCondition;
	}
}