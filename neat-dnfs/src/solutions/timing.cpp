#include "solutions/timing.h"


namespace neat_dnfs
{
	Timing::Timing(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "Timing";
	}

	Timing::Timing(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype)
		: Solution(initialTopology, phenotype)
	{
		name = "Timing";
	}

	SolutionPtr Timing::clone() const
	{
		Timing solution(initialTopology);
		auto clonedSolution = std::make_shared<Timing>(solution);

		return clonedSolution;
	}

	void Timing::testPhenotype()
	{
		using namespace dnf_composer::element;
		parameters.fitness = 0.0;
		parameters.partialFitness.clear();

		static constexpr double wf1 = 1 / 4.f;
		static constexpr double wf2 = 1 / 4.f;
		static constexpr double wf3 = 1 / 4.f;
		static constexpr double wf4 = 1 / 4.f;
		// static constexpr double wf5 = 1 / 6.f;
		// static constexpr double wf6 = 1 / 6.f;

		static int count  = 0;
		count++;

		initSimulation();
		addGaussianStimulus("nf 1", 
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0,
				true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		const double f1 = iterationsUntilBump("nf 1", 100, 150, 10);
		const double f2 = iterationsUntilBump("nf 2", 200, 250, 20);
		//const double f3 = iterationsUntilBump("nf 3", 2000, 2500, 200);
		runSimulation(1000);
		removeGaussianStimuli();
		const double f3 = iterationsUntilNoBump("nf 1", 100, 150, 10);
		const double f4 = iterationsUntilNoBump("nf 2", 1000, 1500, 100);
		//const double f6 = iterationsUntilNoBump("nf 3", 2500, 3000, 100);
		removeGaussianStimuli();

		parameters.partialFitness.push_back(f1);
		parameters.partialFitness.push_back(f2);
		parameters.partialFitness.push_back(f3);
		parameters.partialFitness.push_back(f4);

		parameters.fitness = wf1 * f1 + wf2 * f2 + wf3 * f3 + wf4 * f4;
	}

	void Timing::createPhenotypeEnvironment()
	{
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0,
				true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
	}
}