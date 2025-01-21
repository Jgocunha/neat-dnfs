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


		static constexpr double wbehaviour = 1.f / 2.f;
		static constexpr double amp_tar = 10;
		static constexpr double width_tar = 12;

		initSimulation();
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulationUntilFieldStable("nf 1");
		runSimulationUntilFieldStable("nf 3");

		const double f1_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 1", 50.0, amp_tar, width_tar);
		const double f1_2 = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", 50.0, amp_tar, width_tar);

		removeGaussianStimuli();
		runSimulationUntilFieldStable("nf 1");
		runSimulationUntilFieldStable("nf 3");

		const double f2_1 = closenessToRestingLevel("nf 1");
		const double f2_2 = closenessToRestingLevel("nf 3");

		// f1_1 only one bump at the input field
		// f1_2 only one bump at the output field
		// f2_1 closeness to resting level after removing the stimulus
		// f2_2 closeness to resting level after removing the stimulus
		static constexpr double wf1_1 = 0.20;
		static constexpr double wf1_2 = 0.50;
		static constexpr double wf2_1 = 0.10;
		static constexpr double wf2_2 = 0.20;

		/*std::cout << "Stimulus at nf 1 should create a bump in nf 1 and nf3" << '\n';
		std::cout << "f1_1 a bump at nf 1: " << f1_1 << '\n';
		std::cout << "f1_2 a bump at nf 3: " << f1_2 << '\n';
		std::cout << "f2_1 resting level at nf 1: " << f2_1 << '\n';
		std::cout << "f2_2 resting level at nf 3: " << f2_2 << '\n';*/

		parameters.fitness = wbehaviour * (wf1_1 * f1_1 + wf1_2 * f1_2 + wf2_1 * f2_1 + wf2_2 * f2_2);

		initSimulation();
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude,20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulationUntilFieldStable("nf 1");
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulationUntilFieldStable("nf 1");
		runSimulationUntilFieldStable("nf 3");

		const double f1_1_1 = twoBumpsAtPositionWithAmplitudeAndWidth("nf 1", 
			20.0, amp_tar, width_tar,
			50.0, amp_tar, width_tar);
		const double f1_4_1 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 3", 
			{ 20.0, 50.0 }, amp_tar, width_tar);

		removeGaussianStimuli();
		runSimulationUntilFieldStable("nf 1");
		runSimulationUntilFieldStable("nf 3");

		const double f2_1_1 = closenessToRestingLevel("nf 1");
		const double f2_2_1 = closenessToRestingLevel("nf 3");

		static constexpr double wf1_1_1 = 0.30;
		static constexpr double wf1_4_1 = 0.50;
		static constexpr double wf2_1_1 = 0.10;
		static constexpr double wf2_2_1 = 0.10;

		/*std::cout << "Two stimuli at nf 1 should create two bumps in nf 1 and one bump in nf3 (selection)" << '\n';
		std::cout << "f1_1_1 two bumps at nf 1: " << f1_1_1 << '\n';
		std::cout << "f1_4_1 one bump at nf 3: " << f1_4_1 << '\n';
		std::cout << "f2_1_1 resting level at nf 1: " << f2_1_1 << '\n';
		std::cout << "f2_2_1 resting level at nf 3: " << f2_2_1 << '\n';*/

		parameters.fitness += wbehaviour * (wf1_1_1 * f1_1_1 + wf1_4_1 * f1_4_1 + wf2_1_1 * f2_1_1 + wf2_2_1 * f2_2_1);

		/*initSimulation();
		addGaussianStimulus("nf 1",
			{ 5.0, 15.0, 20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulationUntilFieldStable("nf 1");
		addGaussianStimulus("nf 1",
			{ 5.0, 15.0, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulationUntilFieldStable("nf 1");
		addGaussianStimulus("nf 1",
			{ 5.0, 15.0, 80.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulationUntilFieldStable("nf 1");
		runSimulationUntilFieldStable("nf 3");

		const double f1_1_2 = threeBumpsAtPositionWithAmplitudeAndWidth("nf 1", 
			20.0, 5.0, 10.0,
			50.0, 5.0, 10.0,
			80.0, 5.0, 10.0);
		const double f1_5_2 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 3", 
			{ 20.0, 50.0, 80.0 }, 5.0, 8.0);

		removeGaussianStimuli();
		runSimulationUntilFieldStable("nf 1");
		runSimulationUntilFieldStable("nf 3");

		const double f2_1_2 = closenessToRestingLevel("nf 1");
		const double f2_2_2 = closenessToRestingLevel("nf 3");

		static constexpr double wf1_1_2 = 0.30;
		static constexpr double wf1_5_2 = 0.50;
		static constexpr double wf2_1_2 = 0.10;
		static constexpr double wf2_2_2 = 0.10;

		parameters.fitness += wbehaviour * (wf1_1_2 * f1_1_2 + wf1_5_2 * f1_5_2 + wf2_1_2 * f2_1_2 + wf2_2_2 * f2_2_2);*/


		//initSimulation();
		//addGaussianStimulus("nf 2",
		//			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
		//			{ DimensionConstants::xSize, DimensionConstants::dx });
		//runSimulationUntilFieldStable("nf 2");
		//runSimulationUntilFieldStable("nf 3");
		//const double f3_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", 20.0, amp_tar, width_tar);
		//const double f3_2 = closenessToRestingLevel("nf 3");
		//addGaussianStimulus("nf 1",
		//	{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
		//	{ DimensionConstants::xSize, DimensionConstants::dx });
		//runSimulationUntilFieldStable("nf 1");
		//runSimulationUntilFieldStable("nf 2");
		//runSimulationUntilFieldStable("nf 3");
		//const double f3_3 = negativePreShapednessAtPosition("nf 3", 20.0);
		//addGaussianStimulus("nf 1",
		//	{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
		//	{ DimensionConstants::xSize, DimensionConstants::dx });
		//runSimulationUntilFieldStable("nf 1");
		//runSimulationUntilFieldStable("nf 2");
		//runSimulationUntilFieldStable("nf 3");
		//const double f3_4 = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", 50.0, amp_tar, width_tar);

		//removeGaussianStimuli();
		//runSimulationUntilFieldStable("nf 1");
		//runSimulationUntilFieldStable("nf 2");
		//runSimulationUntilFieldStable("nf 3");
		//const double f3_5 = closenessToRestingLevel("nf 1");
		//const double f3_6 = closenessToRestingLevel("nf 2");
		//const double f3_7 = closenessToRestingLevel("nf 3");

		///*std::cout << "Stimulus at nf 2 should create a bump in nf 2 and a negative pre-shapedness in nf3" << '\n';
		//std::cout << "f3_1 a bump at nf 2: " << f3_1 << '\n';
		//std::cout << "f3_2 negative pre-shapedness at nf 3: " << f3_2 << '\n';
		//std::cout << "f3_3 resting level at nf 2: " << f3_3 << '\n';
		//std::cout << "f3_4 resting level at nf 3: " << f3_4 << '\n';*/

		//static constexpr double wf3_1 = 0.10;
		//static constexpr double wf3_2 = 0.20;
		//static constexpr double wf3_3 = 0.20;
		//static constexpr double wf3_4 = 0.20;
		//static constexpr double wf3_5 = 0.10;
		//static constexpr double wf3_6 = 0.10;
		//static constexpr double wf3_7 = 0.10;

		//parameters.fitness += wbehaviour * (wf3_1 * f3_1 + wf3_2 * f3_2 + wf3_3 * f3_3 + wf3_4 * f3_4 + wf3_5 * f3_5 + wf3_6 * f3_6 + wf3_7 * f3_7);

		//std::cout << "Total fitness: " << parameters.fitness << " of solution " << std::to_string(id) << '\n' << '\n';



		//constexpr double stages = 1.f / 5.f;

		//// stage 1 multi-bump input field 1
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
		//runSimulationUntilFieldStable("nf 1");
		//const double f1_a = threeBumpsAtPositionWithAmplitudeAndWidth("nf 1", 
		//	20.0, 10.0, 12.0, 
		//	50.0, 10.0, 12.0, 
		//	80.0, 10.0, 12.0);
		//removeGaussianStimuli();
		//runSimulationUntilFieldStable("nf 1");
		//const double f1_b = closenessToRestingLevel("nf 1");
		//static constexpr double wf1_a = 0.70;
		//static constexpr double wf1_b = 0.30;
		//parameters.fitness = stages * (wf1_a * f1_a + wf1_b * f1_b);

		//// stage 1.1 output field must select from the input field 1
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
		//runSimulationUntilFieldStable("nf 1");
		//runSimulationUntilFieldStable("nf 3");
		//const double f1_1_a = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 3",
		//	{ 20.0, 50.0, 80.0 }, 10, 12);
		//removeGaussianStimuli();
		//runSimulationUntilFieldStable("nf 1");
		//runSimulationUntilFieldStable("nf 3");
		//const double f1_1_b = closenessToRestingLevel("nf 3");
		//static constexpr double wf1_1_a = 0.80;
		//static constexpr double wf1_1_b = 0.20;
		//parameters.fitness += stages * (wf1_1_a * f1_1_a + wf1_1_b * f1_1_b);

		//// stage 2 one-bump input field 2
		//initSimulation();
		//addGaussianStimulus("nf 2",
		//				{ 5.0, 15.0, 20.0, true, false },
		//				{ DimensionConstants::xSize, DimensionConstants::dx });
		//runSimulationUntilFieldStable("nf 2");
		//const double f2_a = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", 20.0, 10, 12);
		//removeGaussianStimuli();
		//const double f2_b = closenessToRestingLevel("nf 2");
		//static constexpr double wf2_a = 0.70;
		//static constexpr double wf2_b = 0.30;
		//parameters.fitness += stages * (wf2_a * f2_a + wf2_b * f2_b);

		//// stage 3 negative pre-shapedness output field from input field 2
		//initSimulation();
		////addGaussianStimulus("nf 1",
		////	{ 5.0, 15.0, 20.0, true, false },
		////	{ DimensionConstants::xSize, DimensionConstants::dx });
		////runSimulationUntilFieldStable("nf 1");
		////const double f3_a = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", 20.0, 10, 12);
		//addGaussianStimulus("nf 2",
		//				{ 5.0, 15.0, 20.0, true, false },
		//				{ DimensionConstants::xSize, DimensionConstants::dx });
		//runSimulationUntilFieldStable("nf 2");
		//runSimulationUntilFieldStable("nf 3");

		//const double f3_b = negativePreShapednessAtPosition("nf 3", 20);
		//removeGaussianStimuli();
		//static constexpr double wf3_a = 0.00;
		//static constexpr double wf3_b = 1.00;
		//parameters.fitness += stages * (/*wf3_a * f3_a +*/ wf3_b * f3_b);

		//// stage 4 
		//initSimulation();
		//addGaussianStimulus("nf 1",
		//	{ 5.0, 15.0, 20.0, true, false },
		//	{ DimensionConstants::xSize, DimensionConstants::dx });
		//runSimulationUntilFieldStable("nf 1");
		//const double f4_a = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", 20.0, 10, 12);
		//addGaussianStimulus("nf 2",
		//	{ 5.0, 15.0, 20.0, true, false },
		//	{ DimensionConstants::xSize, DimensionConstants::dx });
		//runSimulationUntilFieldStable("nf 2");
		//runSimulationUntilFieldStable("nf 3");

		//const double f4_b = closenessToRestingLevel("nf 3");
		//removeGaussianStimuli();
		//static constexpr double wf4_a = 0.50;
		//static constexpr double wf4_b = 0.50;
		//parameters.fitness += stages * (wf4_a * f4_a + wf4_b * f4_b);

		//initSimulation();
		//addGaussianStimulus("nf 1",
		//	{ 5.0, 15.0, 50.0, true, false },
		//	{ DimensionConstants::xSize, DimensionConstants::dx });

		//runSimulationUntilFieldStable("nf 1");
		//runSimulationUntilFieldStable("nf 2");
		//runSimulationUntilFieldStable("nf 3");

		//const double f1_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 1", 50.0, 5, 8);
		//const double f1_2 = closenessToRestingLevel("nf 2");
		//const double f1_3 = preShapedness("nf 3");
		////const double f1_3 = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", 50.0, 10, 10);

		//addGaussianStimulus("nf 2",
		//	{ 5.0, 15.0, 50.0, true, false },
		//	{ DimensionConstants::xSize, DimensionConstants::dx });

		//runSimulationUntilFieldStable("nf 1");
		//runSimulationUntilFieldStable("nf 2");
		//runSimulationUntilFieldStable("nf 3");

		//const double f2_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", 50.0, 5, 8);
		//const double f2_3 = closenessToRestingLevel("nf 3");

		//removeGaussianStimuli();
		//runSimulationUntilFieldStable("nf 1");
		//runSimulationUntilFieldStable("nf 2");
		//runSimulationUntilFieldStable("nf 3");

		//const double f3_1 = closenessToRestingLevel("nf 1");
		//const double f3_2 = closenessToRestingLevel("nf 2");
		//const double f3_3 = closenessToRestingLevel("nf 3");

		//// f1_1 only one bump at one of the input fields after adding the stimulus to it
		//// f1_2 closeness to resting level at the other input field
		//// f1_3 closeness to resting level at the output field
		//// f2_1 only one bump at one of the input fields after adding the stimulus to it
		//// f2_3 only one bump at the output field after adding the stimuli to the input fields
		//// f3_1 closeness to resting level at the input field after removing the stimulus
		//// f3_2 closeness to resting level at the other input field after removing the stimulus
		//// f3_3 closeness to resting level at the output field after removing the stimuli
		//static constexpr double wf1_1 = 0.10;
		//static constexpr double wf1_2 = 0.05;
		//static constexpr double wf1_3 = 0.25;

		//static constexpr double wf2_1 = 0.05;
		//static constexpr double wf2_3 = 0.30;

		//static constexpr double wf3_1 = 0.05;
		//static constexpr double wf3_2 = 0.05;
		//static constexpr double wf3_3 = 0.15;

		//parameters.fitness = stages * (wf1_1 * f1_1 + wf1_2 * f1_2 + wf1_3 * f1_3 +
		//	wf2_1 * f2_1 +  wf2_3 * f2_3 +
		//	wf3_1 * f3_1 + wf3_2 * f3_2 + wf3_3 * f3_3);

		//initSimulation();
		//addGaussianStimulus("nf 2",
		//	{ 5.0, 15.0, 50.0, true, false },
		//	{ DimensionConstants::xSize, DimensionConstants::dx });

		//runSimulationUntilFieldStable("nf 1");
		//runSimulationUntilFieldStable("nf 2");
		//runSimulationUntilFieldStable("nf 3");

		//const double f1_1_ = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", 50.0, 10, 8);
		//const double f1_2_ = closenessToRestingLevel("nf 1");
		//const double f1_3_ = negativePreShapedness("nf 3");

		//addGaussianStimulus("nf 1",
		//	{ 5.0, 15.0, 20.0, true, false },
		//	{ DimensionConstants::xSize, DimensionConstants::dx });
		//addGaussianStimulus("nf 1",
		//	{ 5.0, 15.0, 50.0, true, false },
		//	{ DimensionConstants::xSize, DimensionConstants::dx });

		//runSimulationUntilFieldStable("nf 1");
		//runSimulationUntilFieldStable("nf 2");
		//runSimulationUntilFieldStable("nf 3");

		////const double f2_1_ = oneBumpAtPositionWithAmplitudeAndWidth("nf 1", 20.0, 20, 10);
		//const double f2_3_ = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 3", { 20.0, 50.0 }, 5, 8);

		//removeGaussianStimuli();
		//runSimulationUntilFieldStable("nf 1");
		//runSimulationUntilFieldStable("nf 2");
		//runSimulationUntilFieldStable("nf 3");

		//const double f3_1_ = closenessToRestingLevel("nf 1");
		//const double f3_2_ = closenessToRestingLevel("nf 2");
		//const double f3_3_ = closenessToRestingLevel("nf 3");

		//static constexpr double wf1_1_ = 0.10;
		//static constexpr double wf1_2_ = 0.05;
		//static constexpr double wf1_3_ = 0.30;
		//							 
		//static constexpr double wf2_3_ = 0.30;
		//							 
		//static constexpr double wf3_1_ = 0.05;
		//static constexpr double wf3_2_ = 0.05;
		//static constexpr double wf3_3_ = 0.15;

		//parameters.fitness += stages * (wf1_1_ * f1_1_ + wf1_2_ * f1_2_ + wf1_3_ * f1_3_ +
		//				wf2_3_ * f2_3_ +
		//				wf3_1_ * f3_1_ + wf3_2_ * f3_2_ + wf3_3_ * f3_3_);


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

		//runSimulationUntilFieldStable("nf 1");
		//runSimulationUntilFieldStable("nf 2");
		//runSimulationUntilFieldStable("nf 3");

		//addGaussianStimulus("nf 2",
		//	{ 5.0, 15.0, 50.0, true, false },
		//	{ DimensionConstants::xSize, DimensionConstants::dx });

		//runSimulationUntilFieldStable("nf 1");
		//runSimulationUntilFieldStable("nf 2");
		//runSimulationUntilFieldStable("nf 3");

		////const double fa_a = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", 20.0, 10, 5);
		////const double fa_b = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", 80.0, 10, 5);

		//const double fa = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 3", { 20.0, 50.0, 80.0 }, 5, 8);

		//removeGaussianStimuli();
		//runSimulationUntilFieldStable("nf 1");
		//runSimulationUntilFieldStable("nf 2");
		//runSimulationUntilFieldStable("nf 3");

		//const double fb_1 = closenessToRestingLevel("nf 1");
		//const double fb_2 = closenessToRestingLevel("nf 2");
		//const double fb_3 = closenessToRestingLevel("nf 3");

		//// fa only one bump at the output field at the middle position after adding the stimuli to the input fields
		//// fb_1_2_3 closeness to resting level after removing the stimuli
		//static constexpr double wfa = 0.70;
		//static constexpr double wfb_1 = 0.10;
		//static constexpr double wfb_2 = 0.10;
		//static constexpr double wfb_3 = 0.10;

		//parameters.fitness += stages * (wfa * fa + wfb_1 * fb_1 + wfb_2 * fb_2 + wfb_3 * fb_3);


		//initSimulation();
		//addGaussianStimulus("nf 1",
		//	{ 5.0, 15.0, 80.0, true, false },
		//	{ DimensionConstants::xSize, DimensionConstants::dx });
		//addGaussianStimulus("nf 1",
		//	{ 5.0, 15.0, 50.0, true, false },
		//	{ DimensionConstants::xSize, DimensionConstants::dx });
		////addGaussianStimulus("nf 1",
		//	//{ 5.0, 15.0, 80.0, true, false },
		//	//{ DimensionConstants::xSize, DimensionConstants::dx });

		//runSimulationUntilFieldStable("nf 1");
		//runSimulationUntilFieldStable("nf 2");
		//runSimulationUntilFieldStable("nf 3");

		//addGaussianStimulus("nf 2",
		//	{ 5.0, 15.0, 50.0, true, false },
		//	{ DimensionConstants::xSize, DimensionConstants::dx });

		//runSimulationUntilFieldStable("nf 1");
		//runSimulationUntilFieldStable("nf 2");
		//runSimulationUntilFieldStable("nf 3");

		//const double fa_a_ = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", 80.0, 10, 5);
		//const double fa_b_ = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", 80.0, 10, 5);

		//const double fa_ = std::max(fa_a_, fa_b_);

		//removeGaussianStimuli();
		//runSimulationUntilFieldStable("nf 1");
		//runSimulationUntilFieldStable("nf 2");
		//runSimulationUntilFieldStable("nf 3");

		//const double fb_1_ = closenessToRestingLevel("nf 1");
		//const double fb_2_ = closenessToRestingLevel("nf 2");
		//const double fb_3_ = closenessToRestingLevel("nf 3");

		//parameters.fitness += stages * (wfa * fa_ + wfb_1 * fb_1_ + wfb_2 * fb_2_ + wfb_3 * fb_3_);
	}

	void ActionExecutionSimulation::createPhenotypeEnvironment()
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
