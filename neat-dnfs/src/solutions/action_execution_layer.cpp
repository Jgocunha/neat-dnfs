#include "solutions/action_execution_layer.h"

namespace neat_dnfs
{
	ActionExecutionSimulation::ActionExecutionSimulation(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "Action execution";
		// target fitness is 0.90
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

		static constexpr double wbehaviour = 1.f / 1.f;
		static constexpr int iterations = SimulationConstants::maxSimulationSteps;

		static constexpr double in_amp = 8.0;
		static constexpr double in_width = 10.0;
		static constexpr double out_amp = 6.0;
		static constexpr double out_width = 5.0;

		///*Selectivity*/
		//initSimulation();
		//addGaussianStimulus("nf 1",
		//	{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
		//	{ DimensionConstants::xSize, DimensionConstants::dx });
		//runSimulation(iterations);

		//const double f1_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 1", 50.0, in_amp, in_width);
		//const double f1_2 = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", 50.0, out_amp, out_width);

		//removeGaussianStimuli();
		//runSimulation(iterations);

		//const double f2_1 = closenessToRestingLevel("nf 1");
		//const double f2_2 = closenessToRestingLevel("nf 3");

		//// f1_1 only one bump at the input field
		//// f1_2 only one bump at the output field
		//// f2_1 closeness to resting level after removing the stimulus
		//// f2_2 closeness to resting level after removing the stimulus
		//static constexpr double wf1_1 = 0.30;
		//static constexpr double wf1_2 = 0.50;
		//static constexpr double wf2_1 = 0.10;
		//static constexpr double wf2_2 = 0.10;

		//// behaviour 1
		//parameters.fitness = wbehaviour * (wf1_1 * f1_1 + wf1_2 * f1_2 + wf2_1 * f2_1 + wf2_2 * f2_2);

		//initSimulation();
		//addGaussianStimulus("nf 1",
		//	{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
		//	{ DimensionConstants::xSize, DimensionConstants::dx });
		//runSimulation(iterations);
		//addGaussianStimulus("nf 1",
		//	{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
		//	{ DimensionConstants::xSize, DimensionConstants::dx });
		//runSimulation(iterations);

		//const double f1_1_1 = twoBumpsAtPositionWithAmplitudeAndWidth("nf 1",
		//	20.0, in_amp, in_width,
		//	50.0, in_amp, in_width);
		//const double f1_4_1 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 3", { 20.0, 50.0 }, out_amp, out_width);

		//removeGaussianStimuli();
		//runSimulation(iterations);

		//const double f2_1_1 = closenessToRestingLevel("nf 1");
		//const double f2_2_1 = closenessToRestingLevel("nf 3");

		//static constexpr double wf1_1_1 = 0.40;
		//static constexpr double wf1_4_1 = 0.40;
		//static constexpr double wf2_1_1 = 0.10;
		//static constexpr double wf2_2_1 = 0.10;

		//// behaviour 2
		//parameters.fitness += wbehaviour * (wf1_1_1 * f1_1_1 + wf1_4_1 * f1_4_1 + wf2_1_1 * f2_1_1 + wf2_2_1 * f2_2_1);

		//initSimulation();
		//addGaussianStimulus("nf 1",
		//	{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
		//	{ DimensionConstants::xSize, DimensionConstants::dx });
		//runSimulation(iterations);
		//addGaussianStimulus("nf 1",
		//	{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
		//	{ DimensionConstants::xSize, DimensionConstants::dx });
		//runSimulation(iterations);
		//addGaussianStimulus("nf 1",
		//	{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
		//	{ DimensionConstants::xSize, DimensionConstants::dx });
		//runSimulation(iterations);

		//const double f1_1_2 = threeBumpsAtPositionWithAmplitudeAndWidth("nf 1",
		//	20.0, in_amp, in_width,
		//	50.0, in_amp, in_width,
		//	80.0, in_amp, in_width);
		//const double f1_5_2 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 3", { 20.0, 50.0, 80.0 }, out_amp, out_width);

		//removeGaussianStimuli();
		//runSimulation(iterations);

		//const double f2_1_2 = closenessToRestingLevel("nf 1");
		//const double f2_2_2 = closenessToRestingLevel("nf 3");

		//static constexpr double wf1_1_2 = 0.40;
		//static constexpr double wf1_5_2 = 0.40;
		//static constexpr double wf2_1_2 = 0.10;
		//static constexpr double wf2_2_2 = 0.10;

		//// behaviour 3
		//parameters.fitness += wbehaviour * (wf1_1_2 * f1_1_2 + wf1_5_2 * f1_5_2 + wf2_1_2 * f2_1_2 + wf2_2_2 * f2_2_2);
		///*End of selectivity*/

		/*Negative interaction from nf 2*/
		initSimulation();
		//addGaussianStimulus("nf 1",
			//{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			//{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 2",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		//const double f4_0 = oneBumpAtPositionWithAmplitudeAndWidth("nf 1", 20.0, in_amp, in_width);
		const double f4_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", 20.0, in_amp, in_width);
		const double f4_2 = negativePreShapednessAtPosition("nf 3", 20.0);

		removeGaussianStimuli();
		runSimulation(iterations);
		const double f4_3 = closenessToRestingLevel("nf 2");
		const double f4_4 = closenessToRestingLevel("nf 3");

		//static constexpr double wf4_0 = 0.10;
		static constexpr double wf4_1 = 0.10;
		static constexpr double wf4_2 = 0.70;
		static constexpr double wf4_3 = 0.10;
		static constexpr double wf4_4 = 0.10;

		// behaviour 4
		parameters.fitness += wbehaviour * (/*wf4_0 * f4_0 + */ wf4_1 * f4_1 + wf4_2 * f4_2 + wf4_3 * f4_3 + wf4_4 * f4_4);
		/*End of negative interaction from nf 3*/

		/*initSimulation();
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);

		const double f5_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 1", 50.0, in_amp, in_width);
		const double f5_2 = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", 50.0, out_amp, out_width);

		addGaussianStimulus("nf 2",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);

		const double f5_3 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", 50.0, in_amp, in_width);
		const double f5_4 = closenessToRestingLevel("nf 3");

		removeGaussianStimuliFromField("nf 2");
		runSimulation(iterations);

		const double f5_5 = closenessToRestingLevel("nf 2");
		const double f5_6 = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", 50.0, out_amp, out_width);

		static constexpr double wf5_1 = 0.05;
		static constexpr double wf5_2 = 0.10;
		static constexpr double wf5_3 = 0.05;
		static constexpr double wf5_4 = 0.35;
		static constexpr double wf5_5 = 0.10;
		static constexpr double wf5_6 = 0.35;

		parameters.fitness += wbehaviour * (wf5_1 * f5_1 + wf5_2 * f5_2 + wf5_3 * f5_3 + wf5_4 * f5_4 + wf5_5 * f5_5 + wf5_6 * f5_6);*/
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
