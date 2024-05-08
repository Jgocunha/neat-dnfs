#include "neat/population.h"

namespace neat_dnfs
{
	Population::Population(uint16_t size, const SolutionPtr& initialSolution)
		: size(size), currentGeneration(0)
	{
		createInitialEmptySolutions(initialSolution);
	}

	void Population::initialize() const
	{
		buildInitialSolutionsGenome();
	}

	void Population::evolve(uint16_t maxGeneration)
	{
		constexpr double targetFitness = 0.8;
		do
		{
			evaluate();
			speciate();
			select();
			reproduce();
			upkeepBestSolution();
			updateGenerationAndAges();
		} while (currentGeneration < maxGeneration || bestSolution->getFitness() > targetFitness);
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
		mutate();
	}

	SolutionPtr Population::getBestSolution() const
	{
		return bestSolution;
	}

	void Population::createInitialEmptySolutions(const SolutionPtr& initialSolution)
	{
		for (int i = 0; i < size; i++)
			solutions.push_back(initialSolution->clone());
	}

	void Population::buildInitialSolutionsGenome() const
	{
		for (const auto& solution : solutions)
			solution->initialize();
	}

	void Population::mutate() const
	{
		for (const auto& solution : solutions)
			solution->mutate();
		for (const auto& solution : solutions)
			solution->clearGenerationalInnovations();
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
		currentGeneration++;
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
		double totalAdjustedFitness = 0;
		for (const auto& species : speciesList)
			totalAdjustedFitness += species.totalAdjustedFitness();

		// Adjust size to be half for offspring distribution
		const int offspringPoolSize = size / 2;
		double accumulatedOffspring = 0.0;

		for (auto& species : speciesList)
		{
			const double speciesProportion = species.totalAdjustedFitness() / totalAdjustedFitness;
			const double exactOffspring = speciesProportion * offspringPoolSize;

			// Accumulate the exact decimal offspring count to handle rounding error in final distribution
			accumulatedOffspring += exactOffspring;
			const auto roundedOffspring = static_cast<uint16_t>(round(exactOffspring));

			species.setOffspringCount(roundedOffspring);
		}

		// To ensure that the total offspring is exactly half of size, adjust the rounding error
		const int totalRoundedOffspring = std::accumulate(speciesList.begin(), speciesList.end(), 0,
			[](int sum, const auto& species) { return sum + species.getOffspringCount(); });

		if (totalRoundedOffspring != offspringPoolSize)
		{
			int error = abs(static_cast<int>(offspringPoolSize) - totalRoundedOffspring);
			do
			{
				// select a random species to decrement offspring
				const int randomIndex = rand() % speciesList.size();
				if (speciesList[randomIndex].getOffspringCount() > 0)
				{
					speciesList[randomIndex].setOffspringCount(speciesList[randomIndex].getOffspringCount() - 1);
					--error;
				}
			} while (error > 0);
		}
	}

	void Population::killLeastFitSolutions()
	{
		std::vector<SolutionPtr> toRemove;

		// Gather all solutions that need to be removed from each species
		for (auto& species : speciesList) {
			std::vector<SolutionPtr> removed = species.killLeastFitSolutions();
			toRemove.insert(toRemove.end(), removed.begin(), removed.end());
		}

		// Remove dead solutions from the main population vector
		auto newEnd = std::remove_if(solutions.begin(), solutions.end(),
			[&toRemove](const SolutionPtr& solution) {
				return std::find(toRemove.begin(), toRemove.end(), solution) != toRemove.end();
			});
		solutions.erase(newEnd, solutions.end());
	}


	void Population::crossover()
	{
		for (auto& species : speciesList)
			species.crossover();
	}

}
