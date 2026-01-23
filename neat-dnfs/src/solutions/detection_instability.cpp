#include "solutions/detection_instability.h"

namespace neat_dnfs
{
	DetectionInstability::DetectionInstability(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "Detection Instability";
	}

	DetectionInstability::DetectionInstability(const SolutionTopology& initialTopology,
		const dnf_composer::Simulation& phenotype)
		: Solution(initialTopology, phenotype)
	{
		name = "Detection Instability";
	}

	SolutionPtr DetectionInstability::clone() const
	{
		DetectionInstability solution(initialTopology);
		auto clonedSolution = std::make_shared<DetectionInstability>(solution);

		return clonedSolution;
	}

	SolutionPtr DetectionInstability::copy() const
	{
		DetectionInstability solution(initialTopology, phenotype);
		auto copy = std::make_shared<DetectionInstability>(solution);

		return copy;
	}

	void DetectionInstability::testPhenotype()
	{
		using namespace dnf_composer::element;
		parameters.fitness = 0.0;
		static constexpr int iterations = SimulationConstants::maxSimulationSteps;
		parameters.partialFitness.clear();

		initSimulation();
		addGaussianStimulus("nf 1",
					{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude,
						50.0, true, false },
					{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);

		const double f1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 1", 50.0, 20, 10);
		const double f2 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", 50.0, 15, 5);
		parameters.partialFitness.emplace_back(f1);
		parameters.partialFitness.emplace_back(f2);

		removeGaussianStimuli();
		runSimulation(iterations*2);

		const double f3 = closenessToRestingLevel("nf 1");
		const double f4 = closenessToRestingLevel("nf 2");
		parameters.partialFitness.emplace_back(f3);
		parameters.partialFitness.emplace_back(f4);

		// f1 only one bump at the input field
		// f2 only one bump at the output field
		// f3 closeness to resting level after removing the stimulus
		// f4 closeness to resting level after removing the stimulus
		static constexpr double wf1 = 1 / 4.f;
		static constexpr double wf2 = 1 / 4.f;
		static constexpr double wf3 = 1 / 4.f;
		static constexpr double wf4 = 1 / 4.f;

		parameters.fitness = wf1*f1 + wf2*f2 + wf3*f3 + wf4*f4;
	}

	void DetectionInstability::createPhenotypeEnvironment()
	{
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
				{ DimensionConstants::xSize, DimensionConstants::dx });
	}
}