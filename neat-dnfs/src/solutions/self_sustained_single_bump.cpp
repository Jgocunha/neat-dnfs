#include "solutions/self_sustained_single_bump.h"

namespace neat_dnfs
{
	SelfSustainedSingleBumpSolution::SelfSustainedSingleBumpSolution(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "Single bump (self-sustained)";
		// target fitness is 0.85
		// same parameters as single bump
	}

	SolutionPtr SelfSustainedSingleBumpSolution::clone() const
	{
		SelfSustainedSingleBumpSolution solution(initialTopology);
		auto clonedSolution = std::make_shared<SelfSustainedSingleBumpSolution>(solution);

		return clonedSolution;
	}

	void SelfSustainedSingleBumpSolution::testPhenotype()
	{
		using namespace dnf_composer::element;
		parameters.fitness = 0.0;
		parameters.partialFitness.clear();
		static constexpr int iterations = SimulationConstants::maxSimulationSteps;

		initSimulation();
		addGaussianStimulus("nf 1",
			{ 5.0, 15.0, 25.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);

		const double f1_1_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 1", 25.0, 20, 16);
		const double f1_2_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", 25.0, 20, 16);
		parameters.partialFitness.emplace_back(f1_1_1);
		parameters.partialFitness.emplace_back(f1_2_1);

		removeGaussianStimuli();
		runSimulation(iterations);

		const double f2_1_1 = closenessToRestingLevel("nf 1");
		const double f2_2_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", 25.0, 10, 15);
		parameters.partialFitness.emplace_back(f2_1_1);
		parameters.partialFitness.emplace_back(f2_2_1);

		// f1_1 only one bump at the input field
		// f1_2 only one bump at the output field
		// f2_1 closeness to resting level after removing the stimulus
		// f2_2 only one bump at the output field after removing the stimulus
		static constexpr double wf1_1 = 0.25;
		static constexpr double wf1_2 = 0.25;
		static constexpr double wf2_1 = 0.25;
		static constexpr double wf2_2 = 0.25;

		parameters.fitness = wf1_1 * f1_1_1 + wf1_2 * f1_2_1 + wf2_1 * f2_1_1 + wf2_2 * f2_2_1;
	}

	void SelfSustainedSingleBumpSolution::createPhenotypeEnvironment()
	{
		addGaussianStimulus("nf 1",
			{ 5.0, 15.0, 25.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
	}
}
