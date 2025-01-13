#include "solutions/self_sustained_single_bump.h"

namespace neat_dnfs
{
	SelfSustainedSingleBumpSolution::SelfSustainedSingleBumpSolution(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "Single bump (self-sustained)";
		// target fitness is 0.85
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

		addGaussianStimulus("nf 1",
			{ 5.0, 15.0, 25.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		initSimulation();
		runSimulationUntilFieldStable("nf 1");
		runSimulationUntilFieldStable("nf 2");

		const double f1_1_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 1", 25.0, 20, 10);
		const double f1_2_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", 25.0, 15, 5);

		removeGaussianStimuli();
		runSimulationUntilFieldStable("nf 1");
		runSimulationUntilFieldStable("nf 2");

		const double f2_1_1 = closenessToRestingLevel("nf 1");
		const double f2_2_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", 25.0, 10, 5);

		// f1_1 only one bump at the input field
		// f1_2 only one bump at the output field
		// f2_1 closeness to resting level after removing the stimulus
		// f2_2 only one bump at the output field after removing the stimulus
		static constexpr double wf1_1 = 0.10;
		static constexpr double wf1_2 = 0.40;
		static constexpr double wf2_1 = 0.10;
		static constexpr double wf2_2 = 0.40;

		parameters.fitness = 0.5 * (wf1_1 * f1_1_1 + wf1_2 * f1_2_1 + wf2_1 * f2_1_1 + wf2_2 * f2_2_1);

		addGaussianStimulus("nf 1",
			{ 5.0, 15.0, 75.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		initSimulation();
		runSimulationUntilFieldStable("nf 1");
		runSimulationUntilFieldStable("nf 2");

		const double f1_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 1", 75.0, 20, 10);
		const double f1_2 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", 75.0, 15, 5);

		removeGaussianStimuli();
		runSimulationUntilFieldStable("nf 1");
		runSimulationUntilFieldStable("nf 2");

		const double f2_1 = closenessToRestingLevel("nf 1");
		const double f2_2 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", 75.0, 10, 5);

		parameters.fitness += 0.5 * (wf1_1 * f1_1 + wf1_2 * f1_2 + wf2_1 * f2_1 + wf2_2 * f2_2);
	}
}
