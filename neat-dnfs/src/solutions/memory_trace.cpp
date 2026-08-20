#include "solutions/memory_trace.h"


namespace neat_dnfs
{
	MemoryTrace::MemoryTrace(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "Memory Trace";
	}

	MemoryTrace::MemoryTrace(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype)
		: Solution(initialTopology, phenotype)
	{
		name = "Memory Trace";
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
		static constexpr int iterations = SimulationConstants::maxSimulationSteps;

		static constexpr double posA = 20.0;
		static constexpr double posB = 80.0;

		// =========================
		// Phase A: No encoding, no output bump
		// =========================
		initSimulation();
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, posA,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f1 = preShapednessAtPosition("nf 3", posA);
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
		const double f5 = preShapednessAtPosition("nf 3", posB);
		parameters.partialFitness.push_back(f5);

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
		const double f6 = closenessToRestingLevel("nf 2");
		parameters.partialFitness.push_back(f6);
		const double f7 = twoBumpsAtPositionWithAmplitudeAndWidth("nf 1", posA, 10.0, 10.0, posB, 10.0, 10.0);
		parameters.partialFitness.push_back(f7);
		const double f8 = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", posB, 10.0, 10.0);
		parameters.partialFitness.push_back(f8);
		runSimulation(iterations);
		const double f9 = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", posB, 10.0, 10.0);
		parameters.partialFitness.push_back(f9);

		runSimulation(iterations*2);
		const double f10 = noBumps("nf 3");
		parameters.partialFitness.push_back(f10);

		static constexpr double wf1 =  1 / 8.f;
		static constexpr double wf2 =  1 / 8.f;
		static constexpr double wf5 =  1 / 8.f;
		static constexpr double wf6 =  1 / 8.f;
		static constexpr double wf7 =  1 / 8.f;
		static constexpr double wf8 =  1 / 8.f;
		static constexpr double wf9 =  1 / 8.f;
		static constexpr double wf10 = 1 / 8.f;

		parameters.fitness = wf1 * f1 + wf2 * f2 + wf5 * f5 + wf6 * f6 + wf7 * f7 + wf8 * f8 + wf9 * f9 + wf10 * f10;
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