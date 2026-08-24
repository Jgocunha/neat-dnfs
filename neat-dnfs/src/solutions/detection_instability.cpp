#include "solutions/detection_instability.h"

namespace neat_dnfs
{
	DetectionInstability::DetectionInstability(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "Detection Instability";
		loadFitnessWeights("detection-instability", 4);
	}

	DetectionInstability::DetectionInstability(const SolutionTopology& initialTopology,
		const dnf_composer::Simulation& phenotype)
		: Solution(initialTopology, phenotype)
	{
		name = "Detection Instability";
		loadFitnessWeights("detection-instability", 4);
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
		const int iterations = SimulationConstants::maxSimulationSteps;
		parameters.partialFitness.clear();

		static constexpr double position = 50.0;

		initSimulation();
		addGaussianStimulus("nf 1",
					dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude,
						position, true, false },
					dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);

		const double f1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 1", position, 20, 10);
		const double f2 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", position, 15, 5);
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
		const auto& w = fitnessWeights;

		parameters.fitness = w[0]*f1 + w[1]*f2 + w[2]*f3 + w[3]*f4;
	}

	void DetectionInstability::createPhenotypeEnvironment()
	{
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
				dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
	}
}
