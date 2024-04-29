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
		void evaluate() const;
	private:
		void createInitialEmptySolutions(const std::shared_ptr<Solution>& initialSolution);
		void buildInitialSolutionsGenome() const;
	};
}
