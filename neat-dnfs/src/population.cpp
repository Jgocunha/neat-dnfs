#include "population.h"

namespace neat_dnfs
{
	Population::Population(int size, const std::shared_ptr<Solution>& initialSolution)
		: size(size), currentGeneration(0)
	{
		createInitialEmptySolutions(initialSolution);
	}

	void Population::initialize() const
	{
		buildInitialSolutionsGenome();
	}

	void Population::evaluate() const
	{
		for (const auto& solution : solutions)
			solution->evaluate();
	}

	void Population::evolve(int maxGeneration)
	{
		do
		{
			evaluate();
			//select();
			reproduce();
			//speciate();
			upkeepBestSolution();
			currentGeneration++;
		} while (currentGeneration < maxGeneration);
	}

	std::shared_ptr<Solution> Population::getBestSolution() const
	{
		return bestSolution;
	}

	void Population::createInitialEmptySolutions(const std::shared_ptr<Solution>& initialSolution)
	{
		for (int i = 0; i < size; i++)
			solutions.push_back(initialSolution->clone());
	}

	void Population::buildInitialSolutionsGenome() const
	{
		for (const auto& solution : solutions)
			solution->initialize();
	}

	void Population::reproduce() const
	{
		//crossover();
		mutate();
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

}
