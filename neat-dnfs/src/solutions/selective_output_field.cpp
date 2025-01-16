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

		/*initSimulation();

		addGaussianStimulus("nf 1",
			{ 5.0, 15.0, 20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 1",
			{ 5.0, 15.0, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 1",
			{ 5.0, 15.0, 80.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulationUntilFieldStable("nf 1");
		runSimulationUntilFieldStable("nf 2");

		const double fa_1 = threeBumpsAtPositionWithAmplitudeAndWidth("nf 1", 20.0, 15.0, 5.0, 50.0, 15.0, 5.0, 80.0, 15.0, 5.0);
		const double fa_2 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 2", { 20.0, 50.0, 80.0 }, 10.0, 5.0);

		removeGaussianStimuli();
		runSimulationUntilFieldStable("nf 1");
		runSimulationUntilFieldStable("nf 2");

		const double fb_1 = closenessToRestingLevel("nf 1");
		const double fb_2 = closenessToRestingLevel("nf 2");

		static constexpr double wfa_1 = 0.20;
		static constexpr double wfa_2 = 0.60;
		static constexpr double wfb_1 = 0.10;
		static constexpr double wfb_2 = 0.10;

		parameters.fitness = (wfa_1 * fa_1 + wfa_2 * fa_2 + wfb_1 * fb_1 + wfb_2 * fb_2);*/


		initSimulation();

		addGaussianStimulus("nf 1",
			{ 5.0, 15.0, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulationUntilFieldStable("nf 1");
		runSimulationUntilFieldStable("nf 2");

		const double f1_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 1", 50.0, 20, 10);
		const double f1_2 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", 50.0, 15, 5);

		removeGaussianStimuli();
		runSimulationUntilFieldStable("nf 1");
		runSimulationUntilFieldStable("nf 2");

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

		parameters.fitness = 0.3333 * (wf1_1 * f1_1 + wf1_2 * f1_2 + wf2_1 * f2_1 + wf2_2 * f2_2);

		initSimulation();

		addGaussianStimulus("nf 1",
			{ 5.0, 15.0, 20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 1",
			{ 5.0, 15.0, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulationUntilFieldStable("nf 1");
		runSimulationUntilFieldStable("nf 2");

		const double f1_1_1 = twoBumpsAtPositionWithAmplitudeAndWidth("nf 1", 20.0, 15.0, 5.0, 50.0, 15.0, 5.0);
		/*const double f1_2_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", 20.0, 15, 5);
		const double f1_3_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", 50.0, 15, 5);
		const double f1_4_1 = std::max(f1_2_1, f1_3_1);*/
		const double f1_4_1 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 2", { 20.0, 50.0 }, 10.0, 5.0);

		removeGaussianStimuli();
		runSimulationUntilFieldStable("nf 1");
		runSimulationUntilFieldStable("nf 2");

		const double f2_1_1 = closenessToRestingLevel("nf 1");
		const double f2_2_1 = closenessToRestingLevel("nf 2");

		static constexpr double wf1_1_1 = 0.20;
		static constexpr double wf1_4_1 = 0.60;
		static constexpr double wf2_1_1 = 0.10;
		static constexpr double wf2_2_1 = 0.10;

		parameters.fitness += 0.3333 * (wf1_1_1 * f1_1_1 + wf1_4_1 * f1_4_1 + wf2_1_1 * f2_1_1 + wf2_2_1 * f2_2_1);

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
		runSimulationUntilFieldStable("nf 1");
		runSimulationUntilFieldStable("nf 2");

		const double f1_1_2 = threeBumpsAtPositionWithAmplitudeAndWidth("nf 1", 20.0, 15.0, 5.0, 50.0, 15.0, 5.0, 80.0, 15.0, 5.0);
		/*const double f1_2_2 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", 20.0, 15, 5);
		const double f1_3_2 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", 50.0, 15, 5);
		const double f1_4_2 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", 80.0, 15, 5);
		const double f1_5_2 = std::max( std::max(f1_2_2, f1_3_2), f1_4_2);*/
		const double f1_5_2 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 2", { 20.0, 50.0, 80.0 }, 10.0, 5.0);

		removeGaussianStimuli();
		runSimulationUntilFieldStable("nf 1");
		runSimulationUntilFieldStable("nf 2");

		const double f2_1_2 = closenessToRestingLevel("nf 1");
		const double f2_2_2 = closenessToRestingLevel("nf 2");

		static constexpr double wf1_1_2 = 0.20;
		static constexpr double wf1_5_2 = 0.60;
		static constexpr double wf2_1_2 = 0.10;
		static constexpr double wf2_2_2 = 0.10;

		parameters.fitness += 0.3333 * (wf1_1_2 * f1_1_2 + wf1_5_2 * f1_5_2 + wf2_1_2 * f2_1_2 + wf2_2_2 * f2_2_2);
	}

	void SelectiveOutputSolution::createPhenotypeEnvironment()
	{
		addGaussianStimulus("nf 1",
			{ 5.0, 0.0, 20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });

		addGaussianStimulus("nf 1",
			{ 5.0, 0.0, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });

		addGaussianStimulus("nf 1",
			{ 5.0, 0.0, 80.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
	}
}
