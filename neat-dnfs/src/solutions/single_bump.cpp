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
		const dnf_composer::element::GaussStimulusParameters stimParams = {
			5.0,
			25.0,
			50.0,
			false,
			false
		};
		addGaussianStimulus("nf 1", stimParams);
		initSimulation();
		runSimulation(500);
		updateFitness();
		removeGaussianStimuli();
		runSimulation(500);
		stopSimulation();
	}

	void SingleBumpSolution::updateFitness()
	{
		using namespace dnf_composer::element;

		static constexpr double expectedBumpPosition = 50.0;
		static constexpr double expectedBumpWidth = 10.0;
		static constexpr double expectedBumpAmplitude = 10.0;

		const auto field =
			std::dynamic_pointer_cast<NeuralField>(phenotype.getElement("nf 2"));
		const auto fieldBumps = field->getBumps();

		if (fieldBumps.empty())
		{
			parameters.fitness = 0.0;
			return;
		}

		const auto bump = fieldBumps.front();
		const double centroidDifference = std::abs(std::ceil(bump.centroid) - expectedBumpPosition);
		const double widthDifference = std::abs(std::ceil(bump.width) - expectedBumpWidth);
		const double amplitudeDifference = std::abs(std::ceil(bump.amplitude) - expectedBumpAmplitude);

		parameters.fitness = 0.8 / (1.0 + centroidDifference);
		//parameters.fitness += 0.1 / (1.0 + widthDifference);
		//parameters.fitness += 0.1 / (1.0 + amplitudeDifference);
	}
}
