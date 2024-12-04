#include "solutions/single_bump.h"

namespace neat_dnfs
{
	SingleBumpSolution::SingleBumpSolution(const SolutionTopology& topology)
		: Solution(topology)
	{}

	SolutionPtr SingleBumpSolution::clone() const
	{
		SingleBumpSolution solution(initialTopology);
		auto clonedSolution = std::make_shared<SingleBumpSolution>(solution);

		return clonedSolution;
	}

	void SingleBumpSolution::testPhenotype()
	{
		learningPhase();

		const dnf_composer::element::GaussStimulusParameters stimParams = {
			5.0,
			15.0,
			50.0,
			false,
			false
		};
		addGaussianStimulus("nf 1", stimParams);
		initSimulation();
		//runSimulationUntilFieldStable("nf 1");
		//runSimulationUntilFieldStable("nf 2");
		runSimulation(30);
		updateFitness();
		removeGaussianStimuli();
		//runSimulationUntilFieldStable("nf 1");
		//runSimulationUntilFieldStable("nf 2");
		runSimulation(10);
	}

	void SingleBumpSolution::updateFitness()
	{
		using namespace dnf_composer::element;

		static constexpr double expectedBumpPosition = 50.0;
		static constexpr double expectedBumpWidth = 5.0;
		static constexpr double expectedBumpAmplitude = 10.0;

		const auto field =
			std::dynamic_pointer_cast<NeuralField>(phenotype.getElement("nf 2"));
		const auto fieldBumps = field->getBumps();

		parameters.bumps = fieldBumps;
		if (fieldBumps.empty())
		{
			parameters.fitness = 0.0;
			return;
		}

		const auto bump = fieldBumps.front();
		const double centroidDifference = std::abs(bump.centroid - expectedBumpPosition);
		const double widthDifference = std::abs(bump.width - expectedBumpWidth);
		const double amplitudeDifference = std::abs(bump.amplitude - expectedBumpAmplitude);

		const double positionFitness = 0.5 / (1.0 + centroidDifference);
		const double widthFitness = 0.25 / (1.0 + widthDifference);
		const double amplitudeFitness = 0.25 / (1.0 + amplitudeDifference);

		parameters.fitness = positionFitness + widthFitness + amplitudeFitness;
	}

	bool SingleBumpSolution::isHighestActivationOfFieldEqualTo(const std::string& field, double target_u) const
	{
		const auto neuralField = std::dynamic_pointer_cast<dnf_composer::element::NeuralField>(phenotype.getElement(field));
		const double u_max = neuralField->getHighestActivation();
		constexpr double epsilon = 1e-6;
		return  std::abs(u_max - target_u) < epsilon;
	}

	void SingleBumpSolution::learningPhase()
	{
		// activate learning for all the couplings
		for (const auto& coupling : phenotype.getElements())
		{
			if (coupling->getLabel() == dnf_composer::element::ElementLabel::FIELD_COUPLING)
			{
				const auto fieldCoupling = std::dynamic_pointer_cast<dnf_composer::element::FieldCoupling>(coupling);
				fieldCoupling->setLearning(true);
			}
		}
		// show inout stimulus
		const dnf_composer::element::GaussStimulusParameters stimParams = {
			5.0,
			15.0,
			50.0,
			false,
			false
		};
		addGaussianStimulus("nf 1", stimParams);
		// show output stimulus
		addGaussianStimulus("nf 2", stimParams);
		initSimulation();
		// run simulation
		runSimulation(200);
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
