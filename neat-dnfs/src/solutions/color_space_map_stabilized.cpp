#include "solutions/color_space_map_stabilized.h"

namespace neat_dnfs
{
	ColorSpaceMapStabilizedSolution::ColorSpaceMapStabilizedSolution(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "Color space map";
	}

	SolutionPtr ColorSpaceMapStabilizedSolution::clone() const
	{
		ColorSpaceMapStabilizedSolution solution(initialTopology);
		auto clonedSolution = std::make_shared<ColorSpaceMapStabilizedSolution>(solution);
		return clonedSolution;
	}

	void ColorSpaceMapStabilizedSolution::testPhenotype()
	{
		initSimulation();
		learning();
		testing();
	}

	void ColorSpaceMapStabilizedSolution::learning()
	{
		if (isThereAFieldCoupling())
		{
			for (const auto& inputOutputBump : inputOutputBumps)
			{
				addGaussianStimulus("nf 1", { GaussStimulusConstants::width,
					GaussStimulusConstants::amplitude,
					inputOutputBump.front(),
					GaussStimulusConstants::circularity,
					GaussStimulusConstants::normalization },
					phenotype.getElement("nf 1")->getElementCommonParameters().dimensionParameters);
				addGaussianStimulus("nf 2", { GaussStimulusConstants::width,
					GaussStimulusConstants::amplitude,
					inputOutputBump.back(),
					GaussStimulusConstants::circularity,
					GaussStimulusConstants::normalization },
					phenotype.getElement("nf 2")->getElementCommonParameters().dimensionParameters);

				initSimulation();
				setLearningForFieldCouplings(true);

				runSimulationUntilFieldStable("nf 1");
				runSimulationUntilFieldStable("nf 2");

				setLearningForFieldCouplings(false);
				removeGaussianStimuli();

				runSimulationUntilFieldStable("nf 1");
				runSimulationUntilFieldStable("nf 2");
			}
		}
	}

	void ColorSpaceMapStabilizedSolution::testing()
	{
		parameters.fitness = 0.0;
		for (const auto& inputOutputBump : inputOutputBumps)
		{
			addGaussianStimulus("nf 1", { GaussStimulusConstants::width,
				GaussStimulusConstants::amplitude,
				inputOutputBump.front(),
				GaussStimulusConstants::circularity,
				GaussStimulusConstants::normalization },
				phenotype.getElement("nf 1")->getElementCommonParameters().dimensionParameters);

			initSimulation();

			runSimulationUntilFieldStable("nf 1");
			runSimulationUntilFieldStable("nf 2");

			const double expectedInput = inputOutputBump.front();
			const double expectedOutput = inputOutputBump.back();
			const double f1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 1", expectedInput, 15.0, 10.0);
			const double f2 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", expectedOutput, 15.0, 10.0);

			removeGaussianStimuli();

			runSimulationUntilFieldStable("nf 1");
			runSimulationUntilFieldStable("nf 2");

			const double f3 = closenessToRestingLevel("nf 1");
			const double f4 = closenessToRestingLevel("nf 2");

			// f1 only one bump at the input field
			// f2 only one bump at the output field
			// f3 closeness to resting level at the input field
			// f4 closeness to resting level at the output field
			static constexpr double wf1 = 0.2;
			static constexpr double wf2 = 0.6;
			static constexpr double wf3 = 0.1;
			static constexpr double wf4 = 0.1;
			std::cout << " f1: " << f1 << " f2: " << f2 << " f3: " << f3 << " f4: " << f4 << std::endl;
			parameters.fitness += (1.0 / (inputOutputBumps.size())) * (wf1 * f1 + wf2 * f2 + wf3 * f3 + wf4 * f4);
			std::cout << "Behaviour <" << expectedInput << ", " << expectedOutput << "> fitness: " << parameters.fitness << std::endl;
		}
		std::cout << "Total fitness: " << parameters.fitness << std::endl;
	}

	void ColorSpaceMapStabilizedSolution::updateFitness()
	{

	}
}
