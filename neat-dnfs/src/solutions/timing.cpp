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

		static constexpr double wf1 = 1 / 8.f;
		static constexpr double wf2 = 1 / 8.f;
		static constexpr double wf3 = 1 / 8.f;
		static constexpr double wf4 = 1 / 8.f;
		static constexpr double wf5 = 1 / 8.f;
		static constexpr double wf6 = 1 / 8.f;
		static constexpr double wf7 = 1 / 8.f;
		static constexpr double wf8 = 1 / 8.f;

		static constexpr double in_amp		= 13.0;
		static constexpr double in_width	= 12.0;

		initSimulation();
		addGaussianStimulus("nf 1", 
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0,
				true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(100);
		const double f1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 1", 50.0, in_amp, in_width);
		runSimulation(1000);
		const double f2 = noBumps("nf 2");

		runSimulation(1000);
		const double f3 = noBumps("nf 2");

		runSimulation(1000);
		const double f4 = noBumps("nf 2");

		runSimulation(1000);
		const double f5 = noBumps("nf 2");

		runSimulation(1000);
		const double f6= oneBumpAtPositionWithAmplitudeAndWidth("nf 2", 50.0, 3, 6);

		removeGaussianStimuli();
		runSimulation(100);
		const double f7 = closenessToRestingLevel("nf 1");
		runSimulation(5000);
		const double f8 = closenessToRestingLevel("nf 2");

		parameters.partialFitness.push_back(f1);
		parameters.partialFitness.push_back(f2);
		parameters.partialFitness.push_back(f3);
		parameters.partialFitness.push_back(f4);
		parameters.partialFitness.push_back(f5);
		parameters.partialFitness.push_back(f6);
		parameters.partialFitness.push_back(f7);
		parameters.partialFitness.push_back(f8);

		parameters.fitness = wf1 * f1 + wf2 * f2 + wf3 * f3 + wf4 * f4 + wf5 * f5 + wf6 * f6 + wf7 * f7 + wf8 * f8;
	}

	void Timing::createPhenotypeEnvironment()
	{
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0,
				true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
	}
}