#include "neat/population.h"

namespace neat_dnfs
{
	PopulationParameters::PopulationParameters(uint16_t size, uint16_t numGenerations, double targetFitness)
		: size(size), currentGeneration(0), numGenerations(numGenerations), targetFitness(targetFitness)
	{
		if (size == 0)
			throw std::invalid_argument("Population size must be greater than 0.");
		if (numGenerations == 0)
			throw std::invalid_argument("Number of generations must be greater than 0.");
	}

	Population::Population(const PopulationParameters& parameters, const SolutionPtr& initialSolution)
		: parameters(parameters)
	{
		createInitialEmptySolutions(initialSolution);
	}

	void Population::createInitialEmptySolutions(const SolutionPtr& initialSolution)
	{
		Species firstSpecies;
		for (int i = 0; i < parameters.size; i++)
		{
			SolutionPtr initialSolutionClone = initialSolution->clone();
			firstSpecies.addSolution(initialSolutionClone);
		}
		firstSpecies.updateRepresentative();
		speciesList.push_back(firstSpecies);
	}

	void Population::initialize() const
	{
		for (const auto& species : speciesList)
			for (const auto& solution : species.getSolutions())
				solution->initialize();
	}

	void Population::evolve()
	{
		do
		{
			evaluate();
			speciate();
			reproduce();
			trackBestSolution();
			if( bestSolution->getFitness() > 7)
				std::cout << "Best fitness: " << bestSolution->getFitness() << std::endl;
			trackGenerationalInfo();
			logGenerationalInfo();
		} while (!endConditionMet());
	}

	void Population::evaluate() const
	{
		for (const auto& species : speciesList)
		{
			for (const auto& solution : species.getSolutions())
				solution->evaluate();
		}
	}

	void Population::speciate()
	{
		for (const auto& species : speciesList)
		{
			for (const auto& solution : species.getSolutions())
				assignToSpecies(solution);
		}
	}

	void Population::assignToSpecies(const SolutionPtr& solution)
	{
		bool assigned = false;
		Species* currentSpecies = findSpeciesOfSolution(solution);

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
			newSpecies.updateRepresentative();
			speciesList.push_back(newSpecies);
		}
	}

	Species* Population::findSpeciesOfSolution(const SolutionPtr& solution)
	{
		for (auto& species : speciesList)
			if (species.contains(solution))
				return &species;

		return nullptr;
	}	

	void Population::reproduce()
	{
		calculateAdjustedFitness();
		calculateReproductionProbabilities();

		const uint16_t numSolutionsToKill = getNumSolutionsToKill();
		const uint16_t numOffspring = numSolutionsToKill;

		// Crossover and mutation
		std::vector<SolutionPtr> offspring;
		for(int i = 0; i< numOffspring; i++)
			offspring.push_back(crossover());
		for (const auto& solution : offspring)
			solution->mutate();
		for (const auto& solution : offspring)
			solution->clearGenerationalInnovations();

		// Selection
		static constexpr double killRatio = 0.9;
		uint16_t numSolutionsKilled = 0;
		for (auto& species : speciesList)
		{
			const int numSolutionsToKillInSpecies = static_cast<int>(static_cast<double>(species.size()) * killRatio);
			for (int i = 0; i < numSolutionsToKillInSpecies; i++)
				species.killLeastFitSolution();
			numSolutionsKilled += numSolutionsToKillInSpecies;
		}
		assert(numSolutionsKilled == numSolutionsToKill);

		for (const auto& solution : offspring)
			assignToSpecies(solution);
		assert(size() == parameters.size);
	}

	void Population::calculateAdjustedFitness() const
	{
		for (const auto& species : speciesList)
			species.calculateAdjustedFitness();
	}

	void Population::calculateReproductionProbabilities() const
	{
		double totalAdjustedFitness = 0;
		for (const auto& species : speciesList)
			totalAdjustedFitness += species.totalAdjustedFitness();

		for (auto& species : speciesList)
		{
			for (const auto& solution : species.getSolutions())
			{
				const double reproductionProbability = solution->getParameters().adjustedFitness / totalAdjustedFitness;
				solution->setReproductionProbability(reproductionProbability);
			}
		}
	}

	uint16_t Population::getNumSolutionsToKill() const
	{
		static constexpr double killRatio = 0.9;
		uint16_t numSolutionsToKill = 0;

		for (auto& species : speciesList)
		{
			const size_t numSolutionsInSpecies = species.size();
			numSolutionsToKill += static_cast<int>(static_cast<double>(numSolutionsInSpecies) * killRatio);
		}

		return numSolutionsToKill;
	}

	SolutionPtr Population::crossover() const
	{
		std::vector<SolutionPtr> solutions;
		for (const auto& species : speciesList) 
		{
			const auto& speciesSolutions = species.getSolutions();
			solutions.insert(solutions.end(), speciesSolutions.begin(), speciesSolutions.end());
		}
		std::vector<double> cumulativeProbabilities;
		for (const auto& member : solutions)
		{
			const double reproductionProbability = member->getParameters().reproductionProbability;
			if (cumulativeProbabilities.empty())
				cumulativeProbabilities.push_back(reproductionProbability);
			else
				cumulativeProbabilities.push_back(reproductionProbability + cumulativeProbabilities.back());
		}

		double random1, random2;
		constexpr double tolerance = 1e-10;
		do {
			random1 = static_cast<double>(std::rand()) / RAND_MAX;
			random2 = static_cast<double>(std::rand()) / RAND_MAX;
		} while (std::fabs(random1 - random2) < tolerance);

		const auto it1 = std::lower_bound(cumulativeProbabilities.begin(), cumulativeProbabilities.end(), random1);
		const auto it2 = std::lower_bound(cumulativeProbabilities.begin(), cumulativeProbabilities.end(), random2);
		const auto parent1 = solutions[std::distance(cumulativeProbabilities.begin(), it1)];
		const auto parent2 = solutions[std::distance(cumulativeProbabilities.begin(), it2)];

		return Solution::crossover(parent1, parent2);
	}

	uint16_t Population::size() const
	{
		return std::accumulate(speciesList.begin(), speciesList.end(), 0,
						[](int sum, const Species& species) { return sum + species.size(); });
	}

	void Population::trackBestSolution()
	{
		for (const auto& species : speciesList)
		{
			for (const auto& solution : species.getSolutions())
			{
				if (bestSolution == nullptr || solution->getFitness() > bestSolution->getFitness())
					bestSolution = solution;
			}
		}
	}

	void Population::trackGenerationalInfo()
	{
		for (auto& species : speciesList)
		{
			species.updateRepresentative();
			for (const auto& solution : species.getSolutions())
				solution->incrementAge();
		}
		parameters.currentGeneration++;
	}

	void Population::logGenerationalInfo() const
	{
		tools::logger::log(tools::logger::INFO,
			"Current generation: " + std::to_string(parameters.currentGeneration) +
			" Best fitness: " + std::to_string(bestSolution->getFitness()) +
			" Adjusted fitness: " + std::to_string(bestSolution->getParameters().adjustedFitness) +
			" Number of species: " + std::to_string(speciesList.size()) +
			" Number of solutions: " + std::to_string(size())
		);
	}

	bool Population::endConditionMet() const
	{
		const bool fitnessCondition = bestSolution->getFitness() > parameters.targetFitness;
		const bool generationCondition = parameters.currentGeneration > parameters.numGenerations;
		return fitnessCondition || generationCondition;
	}
}
