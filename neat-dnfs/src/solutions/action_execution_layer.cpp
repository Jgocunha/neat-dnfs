#include "solutions/action_execution_layer.h"
#include <format>

namespace neat_dnfs
{
	ActionExecutionSimulation::ActionExecutionSimulation(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "Action execution";
		// target fitness is 0.85
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
		parameters.partialFitness.clear();
		static constexpr int iterations = SimulationConstants::maxSimulationSteps;

		static constexpr double wf1 = 0.15; // multi bump ORL
		static constexpr double wf2 = 0.20; // ORL creates a selective single bump in AEL
		static constexpr double wf3 = 0.15; // AOL single bump
		static constexpr double wf4 = 0.20; // AOL inhibits AEL
		static constexpr double wf5 = 0.15; // travelling bump in AEL i
		static constexpr double wf6 = 0.15; // travelling bump in AEL ii

		// behaviour 1 - selection from multi-bump ORL
		initSimulation();
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 1",
						dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
						dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f1 = threeBumpsAtPositionWithAmplitudeAndWidth("nf 1",
			20.0, 15, 10,
			50.0, 15, 10,
			80.0, 15, 10
		);
		parameters.fitness = wf1 * f1;
		parameters.partialFitness.emplace_back(f1);

		const double f2 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 3", { 20.0, 50.0, 80.0 }, 10, 10);
		parameters.fitness += wf2 * f2;
		parameters.partialFitness.emplace_back(f2);

		// behaviour 2 - inhibition from AOL
		removeGaussianStimuli();
		initSimulation();
		addGaussianStimulus("nf 2",
						dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
						dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f3 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", 20.0, 15, 10);
		parameters.fitness += wf3 * f3;
		parameters.partialFitness.emplace_back(f3);

		const double f4 = negativePreShapednessAtPosition("nf 3", 20.0);
		parameters.fitness += wf4 * f4;
		parameters.partialFitness.emplace_back(f4);

		// behaviour 3 - both connections
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f5 = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", 50.0, 10, 10);
		parameters.fitness += wf5 * f5;
		parameters.partialFitness.emplace_back(f5);

		moveGaussianStimulusContinuously(std::format("gs nf 2 {}", 20.0), 50.0, +0.5);
		const double f6 = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", 20.0, 10, 10);
		parameters.fitness += wf6 * f6;
		parameters.partialFitness.emplace_back(f6);

		removeGaussianStimuli();


		


		//using namespace dnf_composer::element;
		//parameters.fitness = 0.0;
		//parameters.partialFitness.clear();
		//static constexpr int iterations = SimulationConstants::maxSimulationSteps;

		//static constexpr double wf1 = 0.20; // multi bump ORL
		//static constexpr double wf2 = 0.20; // ORL creates a selective single bump in AEL
		//static constexpr double wf3 = 0.20; // AOL single bump
		//static constexpr double wf4 = 0.20; // AOL inhibits AEL
		//static constexpr double wf5 = 0.20; // travelling bump in AEL

		//initSimulation();
		//addGaussianStimulus("nf 1",
		//	dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
		//	dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		//addGaussianStimulus("nf 1",
		//				dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
		//				dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		//addGaussianStimulus("nf 1",
		//	dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
		//	dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		//runSimulation(iterations);
		//const double f1 = threeBumpsAtPositionWithAmplitudeAndWidth("nf 1",
		//	20.0, 15, 10,
		//	50.0, 15, 10,
		//	80.0, 15, 10
		//);
		//parameters.fitness = wf1 * f1;
		//parameters.partialFitness.emplace_back(f1);

		//const double f2 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 3", { 20.0, 50.0, 80.0 }, 10, 10);
		//parameters.fitness += wf2 * f2;
		//parameters.partialFitness.emplace_back(f2);

		//removeGaussianStimuli();
		//initSimulation();
		//addGaussianStimulus("nf 1",
		//	dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
		//	dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		//addGaussianStimulus("nf 1",
		//	dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
		//	dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		//addGaussianStimulus("nf 2",
		//				dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
		//				dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		//runSimulation(iterations);
		//const double f3 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", 20.0, 15, 10);
		//parameters.fitness += wf3 * f3;
		//parameters.partialFitness.emplace_back(f3);

		//const double f4 = negativePreShapednessAtPosition("nf 3", 20.0);
		//parameters.fitness += wf4 * f4;
		//parameters.partialFitness.emplace_back(f4);

		//const double f5 = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", 50.0, 10, 10);
		//parameters.fitness += wf5 * f5;
		//parameters.partialFitness.emplace_back(f5);

		//removeGaussianStimuli();
	}

	void ActionExecutionSimulation::createPhenotypeEnvironment()
	{
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });

		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude,50.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });

		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });

		addGaussianStimulus("nf 2",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, 0.0, 50.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
	}
}
