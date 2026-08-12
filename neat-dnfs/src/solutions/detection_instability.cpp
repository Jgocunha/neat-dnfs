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


		// -------------------------
		// DFT Detection Instability
		// -------------------------
		// Phase A: weak (subthreshold) stimulus -> no bump in output field
		// Phase B: strong (suprathreshold) stimulus -> fast bump formation in output field
		// Phase C: stimulus removed -> return to resting level (avoid memory solutions)

		static constexpr double pos = 50.0;

		// Use a weak stimulus relative to the default amplitude to test subthreshold stability
		static constexpr double weakAmpFactor = 0.15;
		constexpr double weakAmp = GaussStimulusConstants::amplitude * weakAmpFactor;
		constexpr double strongAmp = GaussStimulusConstants::amplitude;

		// Targets for bump shape (keep modestly strict; adjust if your bump detector differs)
		static constexpr double outAmpTarget = 35.0;
		static constexpr double outWidthTarget = 8.0;

		// Target for detection speed (iterations until first bump)
		static constexpr double targetIt = 50.0;

		// =========================
		// Phase A: subthreshold
		// =========================
		initSimulation();

		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, weakAmp, pos,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });

		runSimulation(iterations);

		const double fA_outNoBump = noBumps("nf 2");
		parameters.partialFitness.emplace_back(fA_outNoBump);

		// =========================
		// Phase B: suprathreshold detection
		// =========================
		removeGaussianStimuli();
		runSimulation(iterations); // Reset the simulation to resting level

		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, strongAmp, pos,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });

		// Reward fast bump onset in the output field
		const double fB_speed = iterationsUntilBump("nf 2", targetIt, targetIt*1.25f, targetIt/10.0f);
		parameters.partialFitness.emplace_back(fB_speed);

		// Let the field settle a bit after detection
		runSimulation(iterations);

		const double fB_shape = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", pos, outAmpTarget, outWidthTarget);
		parameters.partialFitness.emplace_back(fB_shape);

		// =========================
		// Phase C: remove stimulus and relax (avoid memory instability)
		// =========================
		removeGaussianStimuli();
		runSimulation(iterations);

		const double fC_relax = closenessToRestingLevel("nf 2");
		parameters.partialFitness.emplace_back(fC_relax);

		// -------------------------
		// Fitness weights
		// -------------------------
		static constexpr double w_noBump = 1 / 4.f;
		static constexpr double w_speed  = 1 / 4.f;
		static constexpr double w_shape  = 1 / 4.f;
		static constexpr double w_relax  = 1 / 4.f;

		parameters.fitness =
			w_noBump * fA_outNoBump +
			w_speed  * fB_speed +
			w_shape  * fB_shape +
			w_relax  * fC_relax;
	}

	void DetectionInstability::createPhenotypeEnvironment()
	{
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
				dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
	}
}