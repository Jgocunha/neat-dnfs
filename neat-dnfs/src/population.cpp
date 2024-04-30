#include "population.h"

namespace neat_dnfs
{
	Population::Population(int size, const SolutionPtr& initialSolution)
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
			speciate();
			reproduce();
			upkeepBestSolution();
			currentGeneration++;
		} while (currentGeneration < maxGeneration);
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

	void Population::speciate()
	{
		// divide the population into species
		// such that similar topologies are in the same species



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

	double Population::calculateCompatibilityDistanceBetweenTwoSolutions(const SolutionPtr& solution1, const SolutionPtr& solution2) const
	{
		const int excessGenes = getNumberOfExcessGenesBetweenTwoSolutions(solution1, solution2);
		const int disjointGenes = getNumberOfDisjointGenesBetweenTwoSolutions(solution1, solution2);
	}

	int Population::getNumberOfDisjointGenesBetweenTwoSolutions(const SolutionPtr& solution1, const SolutionPtr& solution2) const
	{
		int disjointGenes = 0;
		int i = 0, j = 0;
		while (i < solution1->getGenomeSize() && j < solution2->getGenomeSize()) 
		{
			if (solution1-> < solution2[j].innovationNumber) {
				disjointGenes++;
				i++;
			}
			else if (solution2[j].innovationNumber < solution1[i].innovationNumber) 
			{
				disjointGenes++;
				j++;
			}
			else {
				i++;
				j++;
			}
		}
		return disjointGenes;
	}

	int Population::getNumberOfExcessGenesBetweenTwoSolutions(const SolutionPtr& solution1, const SolutionPtr& solution2)
	{
		return abs(solution1->getGenomeSize() - solution2->getGenomeSize());
	}



}
