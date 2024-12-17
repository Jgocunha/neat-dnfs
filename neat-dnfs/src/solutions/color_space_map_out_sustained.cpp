#include "solutions/color_space_map_out_sustained.h"

namespace neat_dnfs
{
	ColorSpaceMapOutputSustainedSolution::ColorSpaceMapOutputSustainedSolution(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "Color space map (self-sustained output)";
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
			
			addGaussianStimulus("nf 1", {
				GaussStimulusConstants::width,
				GaussStimulusConstants::amplitude,
				inputOutputBump.front(),
				GaussStimulusConstants::circularity,
				GaussStimulusConstants::normalization },
				phenotype.getElement("nf 1")->getElementCommonParameters().dimensionParameters);

			initSimulation();

			const double f0_1 = closenessToRestingLevel("nf 1");
			const double f0_2 = closenessToRestingLevel("nf 2");

			runSimulationUntilFieldStable("nf 1");
			runSimulationUntilFieldStable("nf 2");

			const double expectedInput = inputOutputBump.front();
			const double expectedOutput = inputOutputBump.back();
			const double f1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 1", expectedInput, 20.0, 20.0);
			const double f2 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", expectedOutput, 15.0, 17.0);

			removeGaussianStimuli();

			runSimulationUntilFieldStable("nf 1");
			runSimulationUntilFieldStable("nf 2");

			const double f3 = closenessToRestingLevel("nf 1");
			const double f4 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", expectedOutput, 5.0, 9.0);

			// f0_1 closeness to resting level before any stimulus
			// f0_2 closeness to resting level before any stimulus
			// f1 only one bump at the input field
			// f2 only one bump at the output field
			// f3 closeness to resting level
			// f4 only one bump at the output field (after removing the stimulus)
			static constexpr double wf0_1 = 0.05;
			static constexpr double wf0_2 = 0.15;
			static constexpr double wf1 = 0.10;
			static constexpr double wf2 = 0.30;
			static constexpr double wf3 = 0.15;
			static constexpr double wf4 = 0.30;
			std::cout << "f0_1: " << f0_1 << " f0_2: " << f0_2 << " f1: " << f1 << " f2: " << f2 << " f3: " << f3 << " f4: " << f4 << std::endl;
			parameters.fitness += (1.0 / (inputOutputBumps.size())) * (wf1 * f1 + wf2 * f2 + wf3 * f3 + wf4 * f4 + wf0_1 * f0_1 + wf0_2 * f0_2);
			std::cout << "Behaviour <" << expectedInput << ", " << expectedOutput << "> fitness: " << parameters.fitness << std::endl;
		}
		std::cout << "Total fitness: " << parameters.fitness << std::endl;
	}

	void ColorSpaceMapOutputSustainedSolution::updateFitness()
	{

	}
}
