#include "solutions/color_space_map_out_sustained.h"

namespace neat_dnfs
{
	ColorSpaceMapOutputSustainedSolution::ColorSpaceMapOutputSustainedSolution(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "Color space map (self-sustained input)";
	}

	SolutionPtr ColorSpaceMapOutputSustainedSolution::clone() const
	{
		ColorSpaceMapOutputSustainedSolution solution(initialTopology);
		auto clonedSolution = std::make_shared<ColorSpaceMapOutputSustainedSolution>(solution);
		return clonedSolution;
	}

	void ColorSpaceMapOutputSustainedSolution::testPhenotype()
	{
		initSimulation();
		learning();
		testing();
	}

	void ColorSpaceMapOutputSustainedSolution::learning()
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

	void ColorSpaceMapOutputSustainedSolution::testing()
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

			const double expectedOutput = inputOutputBump.back();
			const double f1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 1", expectedOutput, 25.0, 10.0);
			double f2 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", expectedOutput, 15.0, 8.0);

			removeGaussianStimuli();

			runSimulationUntilFieldStable("nf 1");
			runSimulationUntilFieldStable("nf 2");

			const double f3 = closenessToRestingLevel("nf 1");
			const double f4 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", expectedOutput, 15.0, 8.0);

			// f1 only one bump at the input field
			// f2 only one bump at the output field
			// f3 closeness to resting level
			// f4 only one bump at the output field (after removing the stimulus)
			static constexpr double wf1 = 0.15;
			static constexpr double wf2 = 0.35;
			static constexpr double wf3 = 0.15;
			static constexpr double wf4 = 0.35;
			parameters.fitness += (1.0 / (inputOutputBumps.size())) * (wf1 * f1 + wf2 * f2 + wf3 * f3 + wf4 * f4);
		}
	}

	void ColorSpaceMapOutputSustainedSolution::updateFitness()
	{

	}
}
