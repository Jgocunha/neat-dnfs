#pragma once

#include "solution.h"

namespace neat_dnfs
{
	class Population
	{
	private:
		int size;
		unsigned long int currentGeneration;
		std::vector<std::shared_ptr<Solution>> solutions;
	public:
		Population(int size, const std::shared_ptr<Solution>& initialSolution);
		void initialize() const;
		void evaluate();
	private:
		void createInitialEmptySolutions(const std::shared_ptr<Solution>& initialSolution);
		void buildInitialSolutionsGenome() const;
		//void createInitialEmptyGenomes();
		//void addInputGenes(int numInputGenes);
		//void addOutputGenes(int numOutputGenes);
		//void addRandomConnectionGenes(int numConnectionGenes);
	};
}
