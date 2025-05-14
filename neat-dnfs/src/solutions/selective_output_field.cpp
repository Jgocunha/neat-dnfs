#include "solutions/selective_output_field.h"

namespace neat_dnfs
{
	SelectiveOutputSolution::SelectiveOutputSolution(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "Selective output";
		// target fitness is 0.85
	}

	SolutionPtr SelectiveOutputSolution::clone() const
	{
		SelectiveOutputSolution solution(initialTopology);
		auto clonedSolution = std::make_shared<SelectiveOutputSolution>(solution);

		return clonedSolution;
	}

	void SelectiveOutputSolution::testPhenotype()
	{
		using namespace dnf_composer::element;
		parameters.fitness = 0.0;
		parameters.partialFitness.clear();
		static constexpr int iterations = SimulationConstants::maxSimulationSteps;

		static constexpr double in_amp = 8.0;
		static constexpr double in_width = 10.0;
		static constexpr double out_amp = 6.0;
		static constexpr double out_width = 5.0;

		initSimulation();
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);

		const double f1_1_1 = twoBumpsAtPositionWithAmplitudeAndWidth("nf 1", 
			20.0, in_amp, in_width, 
			50.0, in_amp, in_width);
		parameters.partialFitness.emplace_back(f1_1_1);
		const double f1_4_1 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 2", { 20.0, 50.0 }, out_amp, out_width);
		parameters.partialFitness.emplace_back(f1_4_1);

		removeGaussianStimuli();
		runSimulation(iterations);

		const double f2_1_1 = closenessToRestingLevel("nf 1");
		const double f2_2_1 = closenessToRestingLevel("nf 2");
		parameters.partialFitness.emplace_back(f2_1_1);
		parameters.partialFitness.emplace_back(f2_2_1);

		static constexpr double wf1_1_1 = 0.30;
		static constexpr double wf1_4_1 = 0.50;
		static constexpr double wf2_1_1 = 0.10;
		static constexpr double wf2_2_1 = 0.10;

		parameters.fitness = wf1_1_1 * f1_1_1 + wf1_4_1 * f1_4_1 + wf2_1_1 * f2_1_1 + wf2_2_1 * f2_2_1;
	}

	void SelectiveOutputSolution::createPhenotypeEnvironment()
	{
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });

		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude,50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });

		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
	}
}
