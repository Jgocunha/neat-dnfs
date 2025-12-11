#include "solutions/single_bump.h"

namespace neat_dnfs
{
	SingleBumpSolution::SingleBumpSolution(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "Single bump (self-stabilized)";
	}

	SingleBumpSolution::SingleBumpSolution(const SolutionTopology& initialTopology,
		const dnf_composer::Simulation& phenotype)
		: Solution(initialTopology, phenotype)
	{
		name = "Single bump (self-stabilized)";
	}

	SolutionPtr SingleBumpSolution::clone() const
	{
		SingleBumpSolution solution(initialTopology);
		auto clonedSolution = std::make_shared<SingleBumpSolution>(solution);

		return clonedSolution;
	}

	void SingleBumpSolution::testPhenotype()
	{
		using namespace dnf_composer::element;
		parameters.fitness = 0.0;
		static constexpr int iterations = SimulationConstants::maxSimulationSteps;
		parameters.partialFitness.clear();


		addGaussianStimulus("nf 1", 
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		initSimulation();
		runSimulation(iterations);

		const double f1_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 1", 50.0, 20, 10);
		parameters.partialFitness.emplace_back(f1_1);
		const double f1_2 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", 50.0, 15, 5);
		parameters.partialFitness.emplace_back(f1_2);

		removeGaussianStimuli();
		runSimulation(iterations);

		const double f2_1 = closenessToRestingLevel("nf 1");
		parameters.partialFitness.emplace_back(f2_1);
		const double f2_2 = closenessToRestingLevel("nf 2");
		parameters.partialFitness.emplace_back(f2_2);

		// f1_1 only one bump at the input field
		// f1_2 only one bump at the output field
		// f2_1 closeness to resting level after removing the stimulus
		// f2_2 closeness to resting level after removing the stimulus
		static constexpr double wf1_1 = 0.25;
		static constexpr double wf1_2 = 0.25;
		static constexpr double wf2_1 = 0.25;
		static constexpr double wf2_2 = 0.25;

		parameters.fitness = wf1_1 * f1_1 + wf1_2 * f1_2 + wf2_1 * f2_1 + wf2_2 * f2_2;
	}

	void SingleBumpSolution::createPhenotypeEnvironment()
	{
		addGaussianStimulus("nf 1",
{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0,
	GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
{ DimensionConstants::xSize, DimensionConstants::dx });
	}
}