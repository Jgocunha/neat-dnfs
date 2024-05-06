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
			updateGenerationAndAges();
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
		constexpr double compatibilityThreshold = 3.0;

		for(const auto& solution : solutions)
		{
			bool foundSpecies = false;

			for(auto& specie : species)
			{
				const double distance = 
					calculateCompatibilityDistanceBetweenTwoSolutions(solution, specie.getRepresentative());
				if(distance < compatibilityThreshold)
				{
					solution->setSpecies(specie.getId());
					specie.setRepresentative(solution);
					foundSpecies = true;
					break;
				}
			}

			if(!foundSpecies)
				createSpeciesAndAddSolutionToIt(solution);
		}
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
		constexpr double c1 = 0.45;
		constexpr double c2 = 0.45;
		constexpr double c3 = 0.1;

		const int excessGenes = getNumberOfExcessGenesBetweenTwoSolutions(solution1, solution2);
		const int disjointGenes = getNumberOfDisjointGenesBetweenTwoSolutions(solution1, solution2);
		const double connectionDifference = getAverageConnectionDifferenceBetweenTwoSolutions(solution1, solution2);

		int N = std::max(solution1->getGenomeSize(), solution2->getGenomeSize());
		if (N < 20) N = 1; // Normalize for small genomes

		return c1 * excessGenes + c2 * disjointGenes + c3; // * connectionDifference / N;
	}

	int Population::getNumberOfDisjointGenesBetweenTwoSolutions(const SolutionPtr& solution1, const SolutionPtr& solution2) const
	{
		const auto innovations1 = solution1->getInnovationNumbers();
		const auto innovations2 = solution2->getInnovationNumbers();

		std::vector<uint16_t> sortedInnovations1 = innovations1; // Make a copy to sort
		std::vector<uint16_t> sortedInnovations2 = innovations2; // Make a copy to sort

		std::sort(sortedInnovations1.begin(), sortedInnovations1.end());
		std::sort(sortedInnovations2.begin(), sortedInnovations2.end());

		std::vector<int> disjointInnovations1, disjointInnovations2;

		// Get the elements in innovations1 that are not in innovations2
		std::set_difference(sortedInnovations1.begin(), sortedInnovations1.end(),
			sortedInnovations2.begin(), sortedInnovations2.end(),
			std::inserter(disjointInnovations1, disjointInnovations1.begin()));

		// Get the elements in innovations2 that are not in innovations1
		std::set_difference(sortedInnovations2.begin(), sortedInnovations2.end(),
			sortedInnovations1.begin(), sortedInnovations1.end(),
			std::inserter(disjointInnovations2, disjointInnovations2.begin()));

		// Calculate the maximum and minimum innovation numbers
		const int maxInnovation1 = !innovations1.empty() ? *std::max_element(innovations1.begin(), innovations1.end()) : 0;
		const int maxInnovation2 = !innovations2.empty() ? *std::max_element(innovations2.begin(), innovations2.end()) : 0;
		int minMaxInnovation = std::min(maxInnovation1, maxInnovation2);

		// Filter out disjoint genes that are beyond the range of the other genome
		const int numDisjointGenes1 = std::count_if(disjointInnovations1.begin(), disjointInnovations1.end(),
			[minMaxInnovation](int innovation) { return innovation <= minMaxInnovation; });
		const int numDisjointGenes2 = std::count_if(disjointInnovations2.begin(), disjointInnovations2.end(),
			[minMaxInnovation](int innovation) { return innovation <= minMaxInnovation; });

		return numDisjointGenes1 + numDisjointGenes2;
	}

	int Population::getNumberOfExcessGenesBetweenTwoSolutions(const SolutionPtr& solution1, const SolutionPtr& solution2)
	{
		const auto solution1InnovationNumbers = solution1->getInnovationNumbers();
		const auto solution2InnovationNumbers = solution2->getInnovationNumbers();

		if (solution1InnovationNumbers.empty() || solution2InnovationNumbers.empty())
			return 0;

		int maxInnovation1 = *std::max_element(solution1InnovationNumbers.begin(),
			solution1InnovationNumbers.end());
		int maxInnovation2 = *std::max_element(solution2InnovationNumbers.begin(),
			solution2InnovationNumbers.end());

		// Calculate excess genes in both genomes
		const int excessCount1 = std::count_if(solution1InnovationNumbers.begin(), solution1InnovationNumbers.end(),
			[maxInnovation2](uint16_t innovation) { return innovation > maxInnovation2; });
		const int excessCount2 = std::count_if(solution2InnovationNumbers.begin(), solution2InnovationNumbers.end(),
			[maxInnovation1](uint16_t innovation) { return innovation > maxInnovation1; });

		// Return the total number of excess genes
		return excessCount1 + excessCount2;
	}

	void Population::createSpeciesAndAddSolutionToIt(const SolutionPtr& solution)
	{
		species.emplace_back();
		species.back().setRepresentative(solution);
		solution->setSpecies(species.back().getId());
	}

	void Population::updateGenerationAndAges()
	{
		currentGeneration++;
		for(const auto& solution : solutions)
			solution->incrementAge();
	}

	double Population::getAverageConnectionDifferenceBetweenTwoSolutions(const SolutionPtr& solution1, const SolutionPtr& solution2) const
	{
		const auto connectionGenes1 = solution1->getGenome().getConnectionGenes();
		const auto connectionGenes2 = solution2->getGenome().getConnectionGenes();

		if (connectionGenes1.empty() || connectionGenes2.empty())
			return 0;

		double sumAmplitudeDifference = 0.0;
		double sumWidthDifference = 0.0;
		static constexpr double amplitudeDifferenceInfluence = 0.8;
		static constexpr double widthDifferenceInfluence = 0.2;

		for (const auto& connectionGene1 : connectionGenes1)
		{
			for (const auto& connectionGene2 : connectionGenes2)
			{
				if (connectionGene1.getInnovationNumber() == connectionGene2.getInnovationNumber())
				{
					const double amplitudeDifference = std::abs(connectionGene1.getKernelAmplitude() - connectionGene2.getKernelAmplitude());
					const double widthDifference = std::abs(connectionGene1.getKernelSigma() - connectionGene2.getKernelSigma());
					sumAmplitudeDifference += amplitudeDifference;
					sumWidthDifference += widthDifference;
				}
			}
		}

		const double totalDifference = sumAmplitudeDifference * amplitudeDifferenceInfluence + sumWidthDifference * widthDifferenceInfluence;

		return totalDifference;
	}

}
