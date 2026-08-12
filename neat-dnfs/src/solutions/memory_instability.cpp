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

		// ---------------------------
		// DFT Memory Instability Task
		// ---------------------------
		// Phase A (encode): strong input -> bump forms in memory field (nf 2)
		// Phase B (delay): input removed -> bump persists in nf 2 across time
		// Additionally: input field (nf 1) should relax back to resting level

		// Targets for memory bump in nf2 (tune if your bump detector differs)
		static constexpr double pos = 50.0;
		static constexpr double memAmpTargetEncode = 20.0;
		static constexpr double memWidthTargetEncode = 10.0;
		static constexpr double memAmpTargetDelay = 15.0;
		static constexpr double memWidthTargetDelay = 12.0;

		// Split delay into checkpoints to discourage "slow decay" solutions
		constexpr int chunk = (iterations >= 3) ? (iterations / 3) : 1;
		constexpr int rem = iterations - 2 * chunk;

		// =========================
		// Phase A: Encode
		// =========================
		initSimulation();
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, pos,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });

		runSimulation(iterations);

		// Core encoding requirement: memory bump should exist in nf2
		const double f_encode = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", pos,
			memAmpTargetEncode, memWidthTargetEncode);
		parameters.partialFitness.emplace_back(f_encode);

		// =========================
		// Phase B: Delay (input removed)
		// =========================
		removeGaussianStimuli();

		// Checkpoint 1
		runSimulation(chunk);
		const double f_mem_t1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", pos,
			memAmpTargetDelay, memWidthTargetDelay);
		parameters.partialFitness.emplace_back(f_mem_t1);

		// Checkpoint 2
		runSimulation(chunk);
		const double f_mem_t2 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", pos,
			memAmpTargetDelay, memWidthTargetDelay);
		parameters.partialFitness.emplace_back(f_mem_t2);

		// Checkpoint 3 (end of delay)
		runSimulation(rem);
		const double f_mem_t3 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", pos,
			memAmpTargetDelay, memWidthTargetDelay);
		parameters.partialFitness.emplace_back(f_mem_t3);

		// Average persistence score
		const double f_memory = (f_mem_t1 + f_mem_t2 + f_mem_t3) / 3.0;
		parameters.partialFitness.emplace_back(f_memory);

		// Input field should relax back to baseline (helps prevent "sticky perception")
		const double f_input_relax = closenessToRestingLevel("nf 1");
		parameters.partialFitness.emplace_back(f_input_relax);

		// -------------------------
		// Fitness weights
		// -------------------------
		// Make persistence dominant: this is the defining signature of memory instability.
		static constexpr double w_encode = 0.25;
		static constexpr double w_memory = 0.65;
		static constexpr double w_relax  = 0.10;

		parameters.fitness =
			w_encode * f_encode +
			w_memory * f_memory +
			w_relax  * f_input_relax;
	}

	void MemoryInstability::createPhenotypeEnvironment()
	{
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0,
					GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
	}
}
