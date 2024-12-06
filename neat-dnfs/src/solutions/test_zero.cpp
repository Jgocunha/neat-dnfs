#include "solutions/test_zero.h"

namespace neat_dnfs
{
	TestZeroSolution::TestZeroSolution(const SolutionTopology& topology)
		: Solution(topology)
	{}

	SolutionPtr TestZeroSolution::clone() const
	{
		TestZeroSolution solution(initialTopology);
		auto clonedSolution = std::make_shared<TestZeroSolution>(solution);
		return clonedSolution;
	}

	void TestZeroSolution::testPhenotype()
	{
		initSimulation();
		learning();
		testing();
	}

	void TestZeroSolution::learning()
	{
		bool isThereACoupling = false;
		for (const auto& coupling : phenotype.getElements())
			if (coupling->getLabel() == dnf_composer::element::ElementLabel::FIELD_COUPLING)
				isThereACoupling = true;

		if (isThereACoupling)
		{
			for( const auto& inputOutputBump : inputOutputBumps)
			{
				addGaussianStimulus("nf 1", { GaussStimulusConstants::width,
					GaussStimulusConstants::amplitude,
					inputOutputBump.front(),
					GaussStimulusConstants::circularity,
					GaussStimulusConstants::normalization
					});
				addGaussianStimulus("nf 2", { GaussStimulusConstants::width,
					GaussStimulusConstants::amplitude,
					inputOutputBump.back(),
					GaussStimulusConstants::circularity,
					GaussStimulusConstants::normalization
					});

				initSimulation();

				for (const auto& coupling : phenotype.getElements())
				{
					if (coupling->getLabel() == dnf_composer::element::ElementLabel::FIELD_COUPLING)
					{
						const auto fieldCoupling = std::dynamic_pointer_cast<dnf_composer::element::FieldCoupling>(coupling);
						fieldCoupling->setLearning(true);
					}
				}
				// run simulation
				runSimulation(400); //250
				// remove stimuli
				removeGaussianStimuli();
				// deactivate learning for all the couplings
				for (const auto& coupling : phenotype.getElements())
				{
					if (coupling->getLabel() == dnf_composer::element::ElementLabel::FIELD_COUPLING)
					{
						const auto fieldCoupling = std::dynamic_pointer_cast<dnf_composer::element::FieldCoupling>(coupling);
						fieldCoupling->setLearning(false);
					}
				}
			}
			
		}
	}

	void TestZeroSolution::testing()
	{
		parameters.fitness = 0.0;
		for (const auto& inputOutputBump : inputOutputBumps)
		{
			addGaussianStimulus("nf 1", { GaussStimulusConstants::width,
				GaussStimulusConstants::amplitude,
				inputOutputBump.front(),
				GaussStimulusConstants::circularity,
				GaussStimulusConstants::normalization
				});

			initSimulation();
			runSimulation(30);
			//updateFitness();
			const double expectedOutput = inputOutputBump.back();
			const auto field =
				std::dynamic_pointer_cast<dnf_composer::element::NeuralField>(phenotype.getElement("nf 2"));
			const auto fieldBumps = field->getBumps();

			parameters.bumps = fieldBumps;
			if (fieldBumps.empty())
			{
				parameters.fitness = 0.0;
				return;
			}

			if(fieldBumps.size() == 1)
			{
				const auto bump = fieldBumps.front();
				const double centroidDifference = std::abs(bump.centroid - expectedOutput);
				parameters.fitness += (1/inputOutputBumps.size()) / (1.0 + centroidDifference);
			}
			else
			{
				// calculate avg centroid
				double avgCentroid = 0.0;
				for (const auto& bump : fieldBumps)
					avgCentroid += bump.centroid;
				avgCentroid /= fieldBumps.size();

				parameters.fitness += (1/(inputOutputBumps.size()*2)) / (1.0 + std::abs(avgCentroid - expectedOutput));
			}
			removeGaussianStimuli();
			runSimulation(10);
		}
	}

	void TestZeroSolution::updateFitness()
	{
		
	}
}
