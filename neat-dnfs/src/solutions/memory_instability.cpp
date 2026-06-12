#include "solutions/memory_instability.h"

namespace neat_dnfs
{
	MemoryInstability::MemoryInstability(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "Memory Instability";
	}

	MemoryInstability::MemoryInstability(const SolutionTopology& initialTopology,
		const dnf_composer::Simulation& phenotype)
			:Solution(initialTopology, phenotype)
	{
		name = "Memory Instability";
	}

	SolutionPtr MemoryInstability::clone() const
	{
		MemoryInstability solution(initialTopology);
		auto clonedSolution = std::make_shared<MemoryInstability>(solution);

		return clonedSolution;
	}

	SolutionPtr MemoryInstability::copy() const
	{
		MemoryInstability solution(initialTopology, phenotype);
		auto copy = std::make_shared<MemoryInstability>(solution);

		return copy;
	}

	void MemoryInstability::testPhenotype()
	{
		using namespace dnf_composer::element;
		parameters.fitness = 0.0;
		parameters.partialFitness.clear();
		static constexpr int iterations = SimulationConstants::maxSimulationSteps;

		initSimulation();
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude,
				50.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);

		const double f1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 1", 50.0, 20, 10);
		const double f2 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", 50.0, 20, 10);
		parameters.partialFitness.emplace_back(f1);
		parameters.partialFitness.emplace_back(f2);

		removeGaussianStimuli();
		runSimulation(iterations);

		const double f3 = closenessToRestingLevel("nf 1");
		const double f4 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", 50.0, 15,12);
		parameters.partialFitness.emplace_back(f3);
		parameters.partialFitness.emplace_back(f4);

		// f1_1 only one bump at the input field
		// f1_2 only one bump at the output field
		// f2_1 closeness to resting level after removing the stimulus
		// f2_2 only one bump at the output field after removing the stimulus
		static constexpr double wf1 = 1 / 4.f;
		static constexpr double wf2 = 1 / 4.f;
		static constexpr double wf3 = 1 / 4.f;
		static constexpr double wf4 = 1 / 4.f;

		parameters.fitness = wf1*f1 + wf2*f2 + wf3*f3 + wf4*f4;
	}

	void MemoryInstability::createPhenotypeEnvironment()
	{
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0,
					GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
	}
}
