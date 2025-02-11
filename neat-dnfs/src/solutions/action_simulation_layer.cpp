#include "solutions/action_simulation_layer.h"

namespace neat_dnfs
{
	ActionSimulationSolution::ActionSimulationSolution(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "Action simulation";
		// target fitness is 0.85
		// same parameters as single bump
	}

	SolutionPtr ActionSimulationSolution::clone() const
	{
		ActionSimulationSolution solution(initialTopology);
		auto clonedSolution = std::make_shared<ActionSimulationSolution>(solution);

		return clonedSolution;
	}

	void ActionSimulationSolution::testPhenotype()
	{
		using namespace dnf_composer::element;
		parameters.fitness = 0.0;
		static constexpr int iterations = SimulationConstants::maxSimulationSteps;

		static constexpr double wf1 = 0.15; // multi bump ORL
		static constexpr double wf2 = 0.20; // ORL pre-shapes ASL
		static constexpr double wf3 = 0.15; // AOL single bump
		static constexpr double wf4 = 0.20; // AOL pre-shapes ASL
		static constexpr double wf5 = 0.30; // ASL = AOL & ORL

		initSimulation();
		addGaussianStimulus("nf 1",
			{ 5.0, 15.0, 20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 1",
			{ 5.0, 15.0, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 1",
			{ 5.0, 15.0, 80.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f1 = threeBumpsAtPositionWithAmplitudeAndWidth("nf 1",
			20.0, 9, 10,
			50.0, 9, 10,
			80.0, 9, 10
		);
		removeGaussianStimuli();
		parameters.fitness = wf1 * f1;

		initSimulation();
		addGaussianStimulus("nf 1",
			{ 5.0, 15.0, 20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f2 = preShapedness("nf 3");
		removeGaussianStimuli();
		parameters.fitness += wf2 * f2;

		initSimulation();
		addGaussianStimulus("nf 2",
			{ 5.0, 15.0, 20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f3 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", 20.0, 9, 10);
		removeGaussianStimuli();
		parameters.fitness += wf3 * f3;

		initSimulation();
		addGaussianStimulus("nf 2",
			{ 5.0, 15.0, 20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f4 = preShapedness("nf 3");
		removeGaussianStimuli();
		parameters.fitness += wf4 * f4;

		initSimulation();
		addGaussianStimulus("nf 1",
			{ 5.0, 15.0, 20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 2",
			{ 5.0, 15.0, 20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f5 = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", 20.0, 9, 10);
		removeGaussianStimuli();
		parameters.fitness += wf5 * f5;


		//initSimulation();
		//addGaussianStimulus("nf 1",
		//	{ 5.0, 15.0, 20.0, true, false },
		//	{ DimensionConstants::xSize, DimensionConstants::dx });
		//addGaussianStimulus("nf 1",
		//	{ 5.0, 15.0, 50.0, true, false },
		//	{ DimensionConstants::xSize, DimensionConstants::dx });
		//addGaussianStimulus("nf 1",
		//	{ 5.0, 15.0, 80.0, true, false },
		//	{ DimensionConstants::xSize, DimensionConstants::dx });
		//runSimulation(iterations);

		//const double f0_1 = threeBumpsAtPositionWithAmplitudeAndWidth("nf 1", 
		//	20.0, 9, 10,
		//	50.0, 9, 10, 
		//	80.0, 9, 10
		//);

		//removeGaussianStimuli();
		//runSimulation(iterations);

		//const double f0_2 = closenessToRestingLevel("nf 1");
		//const double f0_3 = closenessToRestingLevel("nf 3");

		//static constexpr double wf0_1 = 0.70;
		//static constexpr double wf0_2 = 0.10;
		//static constexpr double wf0_3 = 0.20;

		//parameters.fitness = w_behaviour * (wf0_1 * f0_1 + wf0_2 * f0_2 + wf0_3 * f0_3);

		//initSimulation();
		//addGaussianStimulus("nf 2",
		//	{ 5.0, 15.0, 20.0, true, false },
		//	{ DimensionConstants::xSize, DimensionConstants::dx });
		//runSimulation(iterations);

		//const double f1_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", 20.0, 9, 10);
		//const double f1_2 = preShapedness("nf 3");

		//removeGaussianStimuli();

		//static constexpr double wf1_1 = 0.50;
		//static constexpr double wf1_2 = 0.50;

		//parameters.fitness += w_behaviour * (wf1_1 * f1_1 + wf1_2 * f1_2);

		//initSimulation();
		//addGaussianStimulus("nf 1",
		//	{ 5.0, 15.0, 20.0, true, false },
		//	{ DimensionConstants::xSize, DimensionConstants::dx });
		//runSimulation(iterations);

		//const double f3_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 1", 20.0, 9, 10);
		//const double f3_2 = preShapedness("nf 3");

		//removeGaussianStimuli();

		//static constexpr double wf3_1 = 0.50;
		//static constexpr double wf3_2 = 0.50;

		//parameters.fitness += w_behaviour * (wf3_1 * f3_1 + wf3_2 * f3_2);

		////initSimulation();
		////addGaussianStimulus("nf 1",
		////	{ 5.0, 15.0, 50.0, true, false },
		////	{ DimensionConstants::xSize, DimensionConstants::dx });

		//////runSimulationUntilFieldStable("nf 1");
		//////runSimulationUntilFieldStable("nf 2");
		//////runSimulationUntilFieldStable("nf 3");
		////runSimulation(iterations);

		////const double f1_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 1", 50.0, 20, 10);
		////const double f1_2 = closenessToRestingLevel("nf 2");
		////const double f1_3 = preShapedness("nf 3");

		////addGaussianStimulus("nf 2",
		////	{ 5.0, 15.0, 50.0, true, false },
		////	{ DimensionConstants::xSize, DimensionConstants::dx });

		//////runSimulationUntilFieldStable("nf 1");
		//////runSimulationUntilFieldStable("nf 2");
		//////runSimulationUntilFieldStable("nf 3");
		////runSimulation(iterations);


		////const double f2_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", 50.0, 20, 10);
		////const double f2_3 = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", 50.0, 10, 5);

		////removeGaussianStimuli();
		//////runSimulationUntilFieldStable("nf 1");
		//////runSimulationUntilFieldStable("nf 2");
		//////runSimulationUntilFieldStable("nf 3");
		////runSimulation(iterations);

		////const double f3_1 = closenessToRestingLevel("nf 1");
		////const double f3_2 = closenessToRestingLevel("nf 2");
		////const double f3_3 = closenessToRestingLevel("nf 3");

		////// f1_1 only one bump at one of the input fields after adding the stimulus to it
		////// f1_2 closeness to resting level at the other input field
		////// f1_3 closeness to resting level at the output field
		////// f2_1 only one bump at one of the input fields after adding the stimulus to it
		////// f2_3 only one bump at the output field after adding the stimuli to the input fields
		////// f3_1 closeness to resting level at the input field after removing the stimulus
		////// f3_2 closeness to resting level at the other input field after removing the stimulus
		////// f3_3 closeness to resting level at the output field after removing the stimuli
		////static constexpr double wf1_1 = 0.10;
		////static constexpr double wf1_2 = 0.05;
		////static constexpr double wf1_3 = 0.25;

		////static constexpr double wf2_1 = 0.05;
		////static constexpr double wf2_3 = 0.30;

		////static constexpr double wf3_1 = 0.05;
		////static constexpr double wf3_2 = 0.05;
		////static constexpr double wf3_3 = 0.15;

		////parameters.fitness = w_behaviour * (wf1_1 * f1_1 + wf1_2 * f1_2 + wf1_3 * f1_3 +
		////	wf2_1 * f2_1 +  wf2_3 * f2_3 +
		////	wf3_1 * f3_1 + wf3_2 * f3_2 + wf3_3 * f3_3);

		////initSimulation();
		////addGaussianStimulus("nf 2",
		////	{ 5.0, 15.0, 50.0, true, false },
		////	{ DimensionConstants::xSize, DimensionConstants::dx });

		//////runSimulationUntilFieldStable("nf 1");
		//////runSimulationUntilFieldStable("nf 2");
		//////runSimulationUntilFieldStable("nf 3");
		////runSimulation(iterations);

		////const double f1_1_ = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", 50.0, 20, 10);
		////const double f1_2_ = closenessToRestingLevel("nf 1");
		////const double f1_3_ = preShapedness("nf 3");

		////addGaussianStimulus("nf 1",
		////	{ 5.0, 15.0, 50.0, true, false },
		////	{ DimensionConstants::xSize, DimensionConstants::dx });

		//////runSimulationUntilFieldStable("nf 1");
		//////runSimulationUntilFieldStable("nf 2");
		//////runSimulationUntilFieldStable("nf 3");
		////runSimulation(iterations);

		////const double f2_1_ = oneBumpAtPositionWithAmplitudeAndWidth("nf 1", 50.0, 20, 10);
		////const double f2_3_ = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", 50.0, 10, 5);

		////removeGaussianStimuli();
		//////runSimulationUntilFieldStable("nf 1");
		//////runSimulationUntilFieldStable("nf 2");
		//////runSimulationUntilFieldStable("nf 3");
		////runSimulation(iterations);

		////const double f3_1_ = closenessToRestingLevel("nf 1");
		////const double f3_2_ = closenessToRestingLevel("nf 2");
		////const double f3_3_ = closenessToRestingLevel("nf 3");

		////parameters.fitness += w_behaviour * (wf1_1 * f1_1_ + wf1_2 * f1_2_ + wf1_3 * f1_3_ +
		////	wf2_1 * f2_1_ + wf2_3 * f2_3_ +
		////	wf3_1 * f3_1_ + wf3_2 * f3_2_ + wf3_3 * f3_3_);


		////initSimulation();
		////addGaussianStimulus("nf 1",
		////	{ 5.0, 15.0, 20.0, true, false },
		////	{ DimensionConstants::xSize, DimensionConstants::dx });
		////addGaussianStimulus("nf 1",
		////	{ 5.0, 15.0, 50.0, true, false },
		////	{ DimensionConstants::xSize, DimensionConstants::dx });
		////addGaussianStimulus("nf 1",
		////	{ 5.0, 15.0, 80.0, true, false },
		////	{ DimensionConstants::xSize, DimensionConstants::dx });

		/////*runSimulationUntilFieldStable("nf 1");
		////runSimulationUntilFieldStable("nf 2");
		////runSimulationUntilFieldStable("nf 3");*/
		////runSimulation(iterations);

		////addGaussianStimulus("nf 2",
		////	{ 5.0, 15.0, 50.0, true, false },
		////	{ DimensionConstants::xSize, DimensionConstants::dx });

		//////runSimulationUntilFieldStable("nf 1");
		//////runSimulationUntilFieldStable("nf 2");
		//////runSimulationUntilFieldStable("nf 3");
		////runSimulation(iterations);

		////const double fa = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", 50.0, 10, 5);

		////removeGaussianStimuli();
		/////*runSimulationUntilFieldStable("nf 1");
		////runSimulationUntilFieldStable("nf 2");
		////runSimulationUntilFieldStable("nf 3");*/
		////runSimulation(iterations);

		////const double fb_1 = closenessToRestingLevel("nf 1");
		////const double fb_2 = closenessToRestingLevel("nf 2");
		////const double fb_3 = closenessToRestingLevel("nf 3");

		////// fa only one bump at the output field at the middle position after adding the stimuli to the input fields
		////// fb_1_2_3 closeness to resting level after removing the stimuli
		////static constexpr double wfa = 0.70;
		////static constexpr double wfb_1 = 0.10;
		////static constexpr double wfb_2 = 0.10;
		////static constexpr double wfb_3 = 0.10;

		////parameters.fitness += 0.333 * (wfa * fa + wfb_1 * fb_1 + wfb_2 * fb_2 + wfb_3 * fb_3);
	}

	void ActionSimulationSolution::createPhenotypeEnvironment()
	{
		addGaussianStimulus("nf 1",
			{ 5.0, 15.0, 20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 1",
			{ 5.0, 15.0, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 1",
			{ 5.0, 15.0, 80.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });

		addGaussianStimulus("nf 2",
			{ 5.0, 0.0, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
	}
}
