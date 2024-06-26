#include "solutions/asl.h"

namespace neat_dnfs
{
	ActionSimulationLayerSolution::ActionSimulationLayerSolution(const SolutionTopology& topology)
		: Solution(topology)
	{}

	SolutionPtr ActionSimulationLayerSolution::clone() const
	{
		ActionSimulationLayerSolution solution(initialTopology);
		auto clonedSolution = std::make_shared<ActionSimulationLayerSolution>(solution);

		return clonedSolution;
	}

	void ActionSimulationLayerSolution::testPhenotype()
	{
		parameters.fitness = 0.0;
		updateFitness();
	}

	void ActionSimulationLayerSolution::updateFitness()
	{
		evaluateConditionWithCoincidentalStimulus();
		if (parameters.fitness > 0)
			evaluateConditionWithNonCoincidentalStimulus();
	}

	bool ActionSimulationLayerSolution::isHighestActivationOfFieldEqualTo(const std::string& field, double target_u) const
	{
		const auto neuralField = std::dynamic_pointer_cast<dnf_composer::element::NeuralField>(phenotype.getElement(field));
		const double u_max = neuralField->getHighestActivation();
		constexpr double epsilon = 1e-6;
		return  std::abs(u_max - target_u) < epsilon;
	}

	void ActionSimulationLayerSolution::evaluateConditionWithCoincidentalStimulus()
	{
		const dnf_composer::element::GaussStimulusParameters stimParams = {
			5.0,
			15.0,
			25.0,
			false,
			false
		};

		addGaussianStimulus("nf 1", stimParams);
		addGaussianStimulus("nf 2", stimParams);
		initSimulation();
		bool isActivationValidBeforeStimulus = isHighestActivationOfFieldEqualTo("nf 1", 0.0);
		isActivationValidBeforeStimulus = isActivationValidBeforeStimulus && isHighestActivationOfFieldEqualTo("nf 2", 0.0);
		runSimulationUntilFieldStable("nf 1");
		runSimulationUntilFieldStable("nf 2");
		runSimulationUntilFieldStable("nf 3");

		bool isActivationValidAfterStimulus = isHighestActivationOfFieldEqualTo("nf 1", 9.9859044930311960);
		isActivationValidAfterStimulus = isActivationValidAfterStimulus && isHighestActivationOfFieldEqualTo("nf 2", 9.9859044930311960);

		parameters.fitness += 0.5 * fitnessCoincidentalStimulus();

		if (!isActivationValidBeforeStimulus || !isActivationValidAfterStimulus)
			tools::logger::log(tools::logger::ERROR, "Activation of field is not valid.");

		removeGaussianStimuli();
		runSimulationUntilFieldStable("nf 1");
		runSimulationUntilFieldStable("nf 2");
		runSimulationUntilFieldStable("nf 3");
		stopSimulation();
	}

	double ActionSimulationLayerSolution::fitnessCoincidentalStimulus() const
	{
		using namespace dnf_composer::element;

		const auto field = std::dynamic_pointer_cast<NeuralField>(phenotype.getElement("nf 3"));
		const auto fieldBumps = field->getBumps();

		if (fieldBumps.empty())
			return 0.0;

		const auto bump = fieldBumps.front();

		constexpr double w = 1.0 / 3.0;
		const double score = w * 1 + // presence of bump
			w * tools::utils::normalizeWithGaussian(bump.amplitude, 10, 3) +
			w * tools::utils::normalizeWithGaussian(bump.width, 10, 3);
		return score;
	}

	void ActionSimulationLayerSolution::evaluateConditionWithNonCoincidentalStimulus()
	{
		const dnf_composer::element::GaussStimulusParameters stim_1 = {
			5.0,
			15.0,
			25.0,
			false,
			false
		};

		const dnf_composer::element::GaussStimulusParameters stim_2 = {
			5.0,
			15.0,
			50.0,
			false,
			false
		};

		addGaussianStimulus("nf 1", stim_1);
		addGaussianStimulus("nf 2", stim_2);
		initSimulation();
		bool isActivationValidBeforeStimulus = isHighestActivationOfFieldEqualTo("nf 1", -10.0);
		isActivationValidBeforeStimulus = isActivationValidBeforeStimulus && isHighestActivationOfFieldEqualTo("nf 2", -10.0);
		runSimulationUntilFieldStable("nf 1");
		runSimulationUntilFieldStable("nf 2");
		runSimulationUntilFieldStable("nf 3");

		bool isActivationValidAfterStimulus = isHighestActivationOfFieldEqualTo("nf 1", 9.9859044930311960);
		isActivationValidAfterStimulus = isActivationValidAfterStimulus && isHighestActivationOfFieldEqualTo("nf 2", 9.9859044930311960);

		parameters.fitness += 0.5 * fitnessNonCoincidentalStimulus();

		if (!isActivationValidBeforeStimulus || !isActivationValidAfterStimulus)
			tools::logger::log(tools::logger::ERROR, "Activation of field is not valid.");

		removeGaussianStimuli();
		runSimulationUntilFieldStable("nf 1");
		runSimulationUntilFieldStable("nf 2");
		runSimulationUntilFieldStable("nf 3");
		stopSimulation();
	}

	double ActionSimulationLayerSolution::fitnessNonCoincidentalStimulus() const
	{
		using namespace dnf_composer::element;
		const auto field = std::dynamic_pointer_cast<NeuralField>(phenotype.getElement("nf 3"));
		const double u_max = field->getHighestActivation();
		const double score = tools::utils::normalizeWithGaussian(u_max, -5.0, 3);
		return score;
	}

}