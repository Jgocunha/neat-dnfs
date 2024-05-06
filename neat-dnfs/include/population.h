#pragma once

#include "solution.h"
#include "species.h"

namespace neat_dnfs
{
	class Population
	{
	private:
		int size;
		uint16_t currentGeneration;
		std::vector<SolutionPtr> solutions;
		std::vector<Species> species;
		SolutionPtr bestSolution;
	public:
		Population(int size, const SolutionPtr& initialSolution);
		void initialize() const;
		void evaluate() const;
		void evolve(int maxGeneration);
		SolutionPtr getBestSolution() const;
	private:
		void createInitialEmptySolutions(const SolutionPtr& initialSolution);
		void buildInitialSolutionsGenome() const;
		void speciate();
		void reproduce() const;
		void mutate() const;
		void upkeepBestSolution();
		double calculateCompatibilityDistanceBetweenTwoSolutions(const SolutionPtr& solution1, 
			const SolutionPtr& solution2) const;
		int getNumberOfDisjointGenesBetweenTwoSolutions(const SolutionPtr& solution1, 
			const SolutionPtr& solution2) const;
		static int getNumberOfExcessGenesBetweenTwoSolutions(const SolutionPtr& solution1, 
			const SolutionPtr& solution2);
		void createSpeciesAndAddSolutionToIt(const SolutionPtr& solution);
		void updateGenerationAndAges();
		double getAverageConnectionDifferenceBetweenTwoSolutions(const SolutionPtr& solution1, 
						const SolutionPtr& solution2) const;
	};
}
