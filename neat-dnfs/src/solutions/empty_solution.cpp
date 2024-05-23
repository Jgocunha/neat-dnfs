#include "solutions/empty_solution.h"

namespace neat_dnfs
{
	EmptySolution::EmptySolution(const SolutionTopology& topology)
		: Solution(topology)
	{}

	SolutionPtr EmptySolution::clone() const
	{
		EmptySolution solution(initialTopology);
		auto clonedSolution = std::make_shared<EmptySolution>(solution);

		return clonedSolution;
	}

	void EmptySolution::testPhenotype()
	{
		updateFitness();
	}

	void EmptySolution::updateFitness()
	{
		const double newFitness = tools::utils::generateRandomDouble(0.0, 1.0);
		if (newFitness > parameters.fitness)
			parameters.fitness = newFitness;
	}
}
