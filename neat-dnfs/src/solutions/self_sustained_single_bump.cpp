#include "solutions/self_sustained_single_bump.h"

namespace neat_dnfs
{
	SelfSustainedSingleBumpSolution::SelfSustainedSingleBumpSolution(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "Single bump (self-sustained)";
		// target fitness is 0.85
		// same parameters as single bump
	}

	SolutionPtr SelfSustainedSingleBumpSolution::clone() const
	{
		SelfSustainedSingleBumpSolution solution(initialTopology);
		auto clonedSolution = std::make_shared<SelfSustainedSingleBumpSolution>(solution);

		return clonedSolution;
	}

	void SelfSustainedSingleBumpSolution::testPhenotype()
	{
		using namespace dnf_composer::element;
		parameters.fitness = 0.0;
		parameters.partialFitness.clear();
		static constexpr int iterations = SimulationConstants::maxSimulationSteps;

		initSimulation();
		addGaussianStimulus("nf 1",
			{ 5.0, 15.0, 25.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);

		const double oneBumpAtInputFieldFitness = oneBumpAtPositionWithAmplitudeAndWidth("nf 1", 25.0, 20, 10);
		const double oneBumpAtOutputFieldFitness = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", 25.0, 20, 10);
		parameters.partialFitness.emplace_back(oneBumpAtInputFieldFitness);
		parameters.partialFitness.emplace_back(oneBumpAtOutputFieldFitness);

		removeGaussianStimuli();
		runSimulation(iterations);

		const double closenessToRestingLevelFitness = closenessToRestingLevel("nf 1");
		const double oneBumpAtOutputFieldAfterStimulusRemovedFitness = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", 25.0, 15,12);
		parameters.partialFitness.emplace_back(closenessToRestingLevelFitness);
		parameters.partialFitness.emplace_back(oneBumpAtOutputFieldAfterStimulusRemovedFitness);

		static constexpr double weightOneBumpAtInputField = 0.25;
		static constexpr double weightOneBumpAtOutputField = 0.25;
		static constexpr double weightClosenessToRestingLevel = 0.25;
		static constexpr double weightOneBumpAtOutputFieldAfterStimulusRemoved = 0.25;

		parameters.fitness = weightOneBumpAtInputField * oneBumpAtInputFieldFitness
			+ weightOneBumpAtOutputField * oneBumpAtOutputFieldFitness
			+ weightClosenessToRestingLevel * closenessToRestingLevelFitness
			+ weightOneBumpAtOutputFieldAfterStimulusRemoved * oneBumpAtOutputFieldAfterStimulusRemovedFitness;
	}

	void SelfSustainedSingleBumpSolution::createPhenotypeEnvironment()
	{
		addGaussianStimulus("nf 1",
			{ 5.0, 15.0, 25.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
	}
}
