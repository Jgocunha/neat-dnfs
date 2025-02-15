#include "solutions/action_execution_layer.h"

namespace neat_dnfs
{
	ActionExecutionSimulation::ActionExecutionSimulation(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "Action execution";
		// target fitness is 0.85
		// deltaT = 25, iterations = 200
	}

	SolutionPtr ActionExecutionSimulation::clone() const
	{
		ActionExecutionSimulation solution(initialTopology);
		auto clonedSolution = std::make_shared<ActionExecutionSimulation>(solution);

		return clonedSolution;
	}

	void ActionExecutionSimulation::testPhenotype()
	{
		using namespace dnf_composer::element;
		parameters.fitness = 0.0;

		static constexpr int iterations = SimulationConstants::maxSimulationSteps;

		static constexpr double in_amp = 9.0;
		static constexpr double in_width = 10.0;
		static constexpr double out_amp = 9.0;
		static constexpr double out_width = 10.0;

		static constexpr double wf1 = 0.20; // multi bump ORL
		static constexpr double wf2 = 0.20; // ORL creates a selective single bump in AEL
		static constexpr double wf3 = 0.20; // AOL single bump
		static constexpr double wf4 = 0.20; // AOL inhibits AEL
		static constexpr double wf5 = 0.20; // travelling bump in AEL

		initSimulation();
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 1",
						{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
						{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f1 = threeBumpsAtPositionWithAmplitudeAndWidth("nf 1",
				20.0, in_amp, in_width,
				50.0, in_amp, in_width,
				80.0, in_amp, in_width);
		parameters.fitness = wf1 * f1;

		const double f2 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 3", { 20.0, 50.0, 80.0 }, out_amp, out_width);
		parameters.fitness += wf2 * f2;

		removeGaussianStimuli();
		initSimulation();
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 2",
						{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
						{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f3 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", 20.0, in_amp, in_width);
		parameters.fitness += wf3 * f3;

		const double f4 = negativePreShapednessAtPosition("nf 3", 20.0);
		parameters.fitness += wf4 * f4;

		const double f5 = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", 50.0, out_amp, out_width);
		parameters.fitness += wf5 * f5;

		removeGaussianStimuli();
	}

	void ActionExecutionSimulation::createPhenotypeEnvironment()
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

		addGaussianStimulus("nf 2",
			{ GaussStimulusConstants::width, 0.0, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
	}
}
