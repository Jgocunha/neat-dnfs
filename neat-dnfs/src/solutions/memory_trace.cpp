#include "solutions/memory_trace.h"


namespace neat_dnfs
{
	MemoryTrace::MemoryTrace(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "Memory Trace";
		loadFitnessWeights("memory-trace", 8);
	}

	MemoryTrace::MemoryTrace(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype)
		: Solution(initialTopology, phenotype)
	{
		name = "Memory Trace";
		loadFitnessWeights("memory-trace", 8);
	}

	SolutionPtr MemoryTrace::clone() const
	{
		MemoryTrace solution(initialTopology);
		auto clonedSolution = std::make_shared<MemoryTrace>(solution);

		return clonedSolution;
	}

	SolutionPtr MemoryTrace::copy() const
	{
		MemoryTrace solution(initialTopology, phenotype);
		auto copy = std::make_shared<MemoryTrace>(solution);

		return copy;
	}

	void MemoryTrace::testPhenotype()
	{
		using namespace dnf_composer::element;
	    parameters.fitness = 0.0;
	    parameters.partialFitness.clear();
		const int iterations = SimulationConstants::maxSimulationSteps;

		static constexpr double posA = 20.0;
		static constexpr double posB = 80.0;

		// =========================
		// Phase A: No encoding, output pre-shape
		// =========================
		initSimulation();
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, posA,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f1 = preShapednessAtPosition("nf 3", posA, BumpFitnessDefaults::preShapednessSigma, BumpFitnessDefaults::preShapednessEpsilon);
		parameters.partialFitness.push_back(f1);
		removeGaussianStimuli();
		runSimulation(iterations);
		const double f2 = closenessToRestingLevel("nf 1");
		parameters.partialFitness.push_back(f2);

		// =========================
		// Phase B: Encoding
		// =========================
		addGaussianStimulus("nf 2",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, posB,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations*5);
		const double f3 = preShapednessAtPosition("nf 3", posB, BumpFitnessDefaults::preShapednessSigma, BumpFitnessDefaults::preShapednessEpsilon);
		parameters.partialFitness.push_back(f3);

		// =========================
		// Phase C: Probing
		// =========================
		removeGaussianStimuli();
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, posA,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, posB,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f4 = closenessToRestingLevel("nf 2");
		parameters.partialFitness.push_back(f4);
		const double f5 = twoBumpsAtPositionWithAmplitudeAndWidth("nf 1", posA, 10.0, 10.0, posB, 10.0, 10.0, BumpFitnessWeights{ 0.70, 0.20, 0.05, 0.05 });
		parameters.partialFitness.push_back(f5);
		const double f6 = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", posB, 10.0, 10.0, BumpFitnessWeights{ 0.45, 0.45, 0.05, 0.05 });
		parameters.partialFitness.push_back(f6);
		runSimulation(iterations);
		const double f7 = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", posB, 10.0, 10.0, BumpFitnessWeights{ 0.45, 0.45, 0.05, 0.05 });
		parameters.partialFitness.push_back(f7);

		runSimulation(iterations*2);
		const double f8 = noBumps("nf 3", BumpFitnessDefaults::noBumpsDecayRate);
		parameters.partialFitness.push_back(f8);

		const auto& w = fitnessWeights;

		parameters.fitness = w[0] * f1
		+ w[1] * f2
		+ w[2] * f3
		+ w[3] * f4
		+ w[4] * f5
		+ w[5] * f6
		+ w[6] * f7
		+ w[7] * f8;
	}

	void MemoryTrace::createPhenotypeEnvironment()
	{
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, 0.0, 20.0,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, 0.0, 80.0,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 2",
		   dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0,
			 GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
		   dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
	}
}
