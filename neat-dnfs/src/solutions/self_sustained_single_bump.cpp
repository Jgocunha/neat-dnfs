#include "solutions/self_sustained_single_bump.h"

namespace neat_dnfs
{
	SelfSustainedSingleBumpSolution::SelfSustainedSingleBumpSolution(const SolutionTopology& topology)
		: Solution(topology)
	{}

	SolutionPtr SelfSustainedSingleBumpSolution::clone() const
	{
		SelfSustainedSingleBumpSolution solution(initialTopology);
		auto clonedSolution = std::make_shared<SelfSustainedSingleBumpSolution>(solution);

		return clonedSolution;
	}

	void SelfSustainedSingleBumpSolution::testPhenotype()
	{
		parameters.fitness = 0.0;
		updateFitness();
	}

	void SelfSustainedSingleBumpSolution::updateFitness()
	{
		const dnf_composer::element::GaussStimulusParameters stimParams = {
			5.0,
			15.0,
			50.0,
			false,
			false
		};

		addGaussianStimulus("nf 1", stimParams);
		initSimulation();
		bool isActivationValidBeforeStimulus = isHighestActivationOfFieldEqualTo("nf 1", 0.0);
		isActivationValidBeforeStimulus = isActivationValidBeforeStimulus && isHighestActivationOfFieldEqualTo("nf 2", 0.0);
		runSimulation(20);
		const bool isActivationValidAfterStimulus = isHighestActivationOfFieldEqualTo("nf 1", 9.9859044930311960);
		//runSimulationUntilFieldStable("nf 1");
		const double bumpFormationScore = selfStabilityFitness();
		removeGaussianStimuli();
		runSimulation(20);
		const bool isActivationValidAfterRemovalOfStimulus = isHighestActivationOfFieldEqualTo("nf 1", -10.0);
		//runSimulationUntilFieldStable("nf 1");
		const double bumpPersistenceScore = selfSustainabilityFitness();
		//runSimulationUntilFieldStable("nf 1");
		//runSimulation(50);

		if (!isActivationValidBeforeStimulus || !isActivationValidAfterStimulus || !isActivationValidAfterRemovalOfStimulus)
			tools::logger::log(tools::logger::ERROR, "Activation of field is not valid.");

		stopSimulation();
		parameters.fitness = 0.5*bumpFormationScore + 0.5*bumpPersistenceScore;
	}

	double SelfSustainedSingleBumpSolution::selfStabilityFitness() const
	{
		using namespace dnf_composer::element;

		const auto field = std::dynamic_pointer_cast<NeuralField>(phenotype.getElement("nf 2"));
		const auto fieldBumps = field->getBumps();

		if (fieldBumps.empty())
			return 0.0;

		const auto bump = fieldBumps.front();

		constexpr double w = 1.0 / 3.0;
		const double score = w * 1 + 
			w * tools::utils::normalizeWithGaussian(bump.amplitude,20, 8.5) +
			w * tools::utils::normalizeWithGaussian(bump.width, 20, 8.5);
		return score;
	}

	double SelfSustainedSingleBumpSolution::selfSustainabilityFitness() const
	{
		using namespace dnf_composer::element;

		const auto field = std::dynamic_pointer_cast<NeuralField>(phenotype.getElement("nf 2"));
		const auto fieldBumps = field->getBumps();

		if (fieldBumps.empty())
			return 0.0;

		const auto bump = fieldBumps.front();

		constexpr double w = 1.0 / 3.0;
		const double score = w * 1 +
			w * tools::utils::normalizeWithGaussian(bump.amplitude, 10, 8.5) +
			w * tools::utils::normalizeWithGaussian(bump.width, 10, 8.5);
		return score;
	}

	bool SelfSustainedSingleBumpSolution::isHighestActivationOfFieldEqualTo(const std::string& field, double target_u) const
	{
		const auto neuralField = std::dynamic_pointer_cast<dnf_composer::element::NeuralField>(phenotype.getElement(field));
		const double u_max = neuralField->getHighestActivation();
		constexpr double epsilon = 1e-6;
		return  std::abs(u_max - target_u) < epsilon;
	}
}
