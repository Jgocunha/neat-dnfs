#include "solutions/selective_output_field.h"

namespace neat_dnfs
{
	SelectiveOutputSolution::SelectiveOutputSolution(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "Selective output";
		// target fitness is 0.85
		// for better results, reduce the search space by:
		// - not allowing inhibitory connections;
		// - not allowing all types of inter-field kernels;
		// - not allowing adding field genes.
		// make sure these settings are:
		// - set noise amplitude to 0.2;
		// - set stabilityThreshold to 0.9.
		// - setting delta_t to 25;
		// - setting max iterations to 200.
		//static constexpr double addFieldGeneProbability = 0.00;
		//static constexpr double mutateFieldGeneProbability = 0.75;
		//static constexpr double addConnectionGeneProbability = 0.10;
		//static constexpr double mutateConnectionGeneProbability = 0.15;
		//static constexpr double toggleConnectionGeneProbability = 0.00;
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

		static constexpr double wbehaviour = 1.f / 3.f;
		static constexpr int iterations = 200;

		static constexpr double in_amp = 8.0;
		static constexpr double in_width = 10.0;
		static constexpr double out_amp = 6.0;
		static constexpr double out_width = 5.0;

		initSimulation();
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);

		const double f1_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 1", 50.0, in_amp, in_width);
		const double f1_2 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", 50.0, out_amp, out_width);

		removeGaussianStimuli();
		runSimulation(iterations);

		const double f2_1 = closenessToRestingLevel("nf 1");
		const double f2_2 = closenessToRestingLevel("nf 2");

		// f1_1 only one bump at the input field
		// f1_2 only one bump at the output field
		// f2_1 closeness to resting level after removing the stimulus
		// f2_2 closeness to resting level after removing the stimulus
		static constexpr double wf1_1 = 0.20;
		static constexpr double wf1_2 = 0.50;
		static constexpr double wf2_1 = 0.10;
		static constexpr double wf2_2 = 0.20;

		parameters.fitness = wbehaviour * (wf1_1 * f1_1 + wf1_2 * f1_2 + wf2_1 * f2_1 + wf2_2 * f2_2);

		initSimulation();
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulationUntilFieldStable("nf 1");
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);

		const double f1_1_1 = twoBumpsAtPositionWithAmplitudeAndWidth("nf 1", 
			20.0, in_amp, in_width, 
			50.0, in_amp, in_width);
		const double f1_4_1 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 2", { 20.0, 50.0 }, out_amp, out_width);

		removeGaussianStimuli();
		runSimulation(iterations);

		const double f2_1_1 = closenessToRestingLevel("nf 1");
		const double f2_2_1 = closenessToRestingLevel("nf 2");

		static constexpr double wf1_1_1 = 0.30;
		static constexpr double wf1_4_1 = 0.50;
		static constexpr double wf2_1_1 = 0.10;
		static constexpr double wf2_2_1 = 0.10;

		parameters.fitness += wbehaviour * (wf1_1_1 * f1_1_1 + wf1_4_1 * f1_4_1 + wf2_1_1 * f2_1_1 + wf2_2_1 * f2_2_1);

		initSimulation();
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulationUntilFieldStable("nf 1");
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulationUntilFieldStable("nf 1");
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);

		const double f1_1_2 = threeBumpsAtPositionWithAmplitudeAndWidth("nf 1",
			20.0, in_amp, in_width,
			50.0, in_amp, in_width,
			80.0, in_amp, in_width);
		const double f1_5_2 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 2", { 20.0, 50.0, 80.0 }, out_amp, out_width);

		removeGaussianStimuli();
		runSimulation(iterations);

		const double f2_1_2 = closenessToRestingLevel("nf 1");
		const double f2_2_2 = closenessToRestingLevel("nf 2");

		static constexpr double wf1_1_2 = 0.30;
		static constexpr double wf1_5_2 = 0.50;
		static constexpr double wf2_1_2 = 0.10;
		static constexpr double wf2_2_2 = 0.10;

		parameters.fitness += wbehaviour * (wf1_1_2 * f1_1_2 + wf1_5_2 * f1_5_2 + wf2_1_2 * f2_1_2 + wf2_2_2 * f2_2_2);
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
