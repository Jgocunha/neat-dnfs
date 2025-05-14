#include "solutions/action_simulation_layer.h"

namespace neat_dnfs
{
	ActionSimulationSolution::ActionSimulationSolution(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "Action simulation";
		// target fitness is 0.85
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
		parameters.partialFitness.clear();

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
			20.0, 15, 10,
			50.0, 15, 10,
			80.0, 15, 10
		);
		const double f2 = preShapedness("nf 3");
		parameters.partialFitness.emplace_back(f1);
		parameters.partialFitness.emplace_back(f2);

		removeGaussianStimuli();
		addGaussianStimulus("nf 2",
				{ 5.0, 15.0, 50.0, true, false },
				{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f3 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", 50.0, 15, 10);
		const double f4 = preShapedness("nf 3");
		parameters.partialFitness.emplace_back(f3);
		parameters.partialFitness.emplace_back(f4);

		addGaussianStimulus("nf 1",
			{ 5.0, 15.0, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });

		runSimulation(iterations);

		const double f5 = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", 50.0, 10, 10);
		parameters.partialFitness.emplace_back(f5);

		removeGaussianStimuli();
		runSimulation(iterations);

		const double f6_1 = closenessToRestingLevel("nf 1");
		const double f6_2 = closenessToRestingLevel("nf 2");
		const double f6_3 = closenessToRestingLevel("nf 3");
		parameters.partialFitness.emplace_back(f6_1);
		parameters.partialFitness.emplace_back(f6_2);
		parameters.partialFitness.emplace_back(f6_3);

		static constexpr double wf1 = 0.10;
		static constexpr double wf2 = 0.20;

		static constexpr double wf3 = 0.10;
		static constexpr double wf4 = 0.20;

		static constexpr double wf5 = 0.25;

		static constexpr double wf6_1 = 0.05;
		static constexpr double wf6_2 = 0.05;
		static constexpr double wf6_3 = 0.05;

		parameters.fitness = wf1 * f1 + wf2 * f2 +
			wf3 * f3 + wf4 * f4 +
			wf5 * f5 +
			wf6_1 * f6_1 + wf6_2 * f6_2 + wf6_3 * f6_3;
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
