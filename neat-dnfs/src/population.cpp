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

	void Population::evaluate()
	{
		//for (auto& solution : solutions)
		//	solution.evaluate();
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

	//void Population::addRandomConnectionGenes(int numConnectionGenes)
	//{
	//	for (int i = 0; i < size; i++)
	//		for (int j = 0; j < numConnectionGenes; j++)
	//			genomes[i].addConnectionGene();
	//}
}
