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
		static constexpr double wf3 = 0.10; // AOL single bump
		static constexpr double wf4 = 0.20; // AOL pre-shapes ASL
		static constexpr double wf5 = 0.10; // ASL no bumps with just ORL
		static constexpr double wf6 = 0.25; // ASL = AOL & ORL

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
		addGaussianStimulus("nf 1",
			{ 5.0, 15.0, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 1",
			{ 5.0, 15.0, 80.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f5 = noBumps("nf 3");
		parameters.fitness += wf5 * f5;

		addGaussianStimulus("nf 2",
			{ 5.0, 15.0, 20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f6 = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", 20.0, 20, 15);
		removeGaussianStimuli();
		parameters.fitness += wf6 * f6;
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
