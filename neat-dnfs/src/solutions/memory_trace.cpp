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

	void MemoryTrace::testPhenotype()
	{
		using namespace dnf_composer::element;
	    parameters.fitness = 0.0;
	    parameters.partialFitness.clear();

	    // Positions
	    static constexpr double A = 50.0;   // primed position
	    static constexpr double C = 80.0;   // novel position (far enough to avoid overlap)

	    // Stimulus amplitudes
		constexpr double strongAmp = GaussStimulusConstants::amplitude;
		constexpr double weakAmp   = 0.35 * GaussStimulusConstants::amplitude;

	    // Timing targets (tune once based on your dt / dynamics)
	    static constexpr double primeTarget   = 220;  // when nf2 should detect under strong input
	    static constexpr double primeMax      = 500;
	    static constexpr double primeTol      = 40;

	    static constexpr double decayTarget   = 500;  // when nf2 should lose bump after input removed
	    static constexpr double decayMax      = 1200;
	    static constexpr double decayTol      = 120;

	    static constexpr double primedTarget  = 120;  // a weak probe at A should be detected relatively fast
	    static constexpr double primedMax     = 350;
	    static constexpr double primedTol     = 35;

	    static constexpr double novelEarlyWindow = 120; // for first 120 steps, novel should NOT bump (early)
	    static constexpr int    settleBetweenProbes = 150;

	    static constexpr double novelTarget   = 260;  // novel weak probe should be detected later (if at all)
	    static constexpr double novelMax      = 350;
	    static constexpr double novelTol      = 30;

	    // Weights (trace signature should dominate)
	    static constexpr double w_prime = 0.15;
	    static constexpr double w_decay = 0.20;
	    static constexpr double w_primed = 0.35;
	    static constexpr double w_novel = 0.30;

	    initSimulation();

	    // --------------------------
	    // Phase 1: PRIME at A (strong)
	    // --------------------------
	    addGaussianStimulus("nf 1",
	        { GaussStimulusConstants::width, strongAmp, A,
	          GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
	        { DimensionConstants::xSize, DimensionConstants::dx });

	    // Encourage nf2 to form bump under strong input
	    const double f_prime = iterationsUntilBump("nf 2", primeTarget, primeMax, primeTol);
	    parameters.partialFitness.emplace_back(f_prime);

	    // --------------------------
	    // Phase 2: REMOVE input and enforce "no working memory"
	    // --------------------------
	    removeGaussianStimuli();

	    // Encourage bump in nf2 to disappear (trace must be subthreshold, not a maintained bump)
	    const double f_decay = iterationsUntilNoBump("nf 2", decayTarget, decayMax, decayTol);
	    parameters.partialFitness.emplace_back(f_decay);

	    // --------------------------
	    // Phase 3: PROBE primed location A (weak) -> should detect early
	    // --------------------------
	    addGaussianStimulus("nf 1",
	        { GaussStimulusConstants::width, weakAmp, A,
	          GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
	        { DimensionConstants::xSize, DimensionConstants::dx });

	    const double f_primed = iterationsUntilBump("nf 2", primedTarget, primedMax, primedTol);
	    parameters.partialFitness.emplace_back(f_primed);

	    // --------------------------
	    // Phase 4: reset, PROBE novel location C (weak)
	    //   - early window: should NOT bump
	    //   - later: if it bumps, it should be late
	    // --------------------------
	    removeGaussianStimuli();
	    runSimulation(settleBetweenProbes);

	    addGaussianStimulus("nf 1",
	        { GaussStimulusConstants::width, weakAmp, C,
	          GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
	        { DimensionConstants::xSize, DimensionConstants::dx });

	    // Early suppression requirement: in the first window, should remain no-bump
	    // We implement this by stepping manually for a short window and checking bumps stay empty.
	    // (This avoids rewarding early bump on the novel location.)
	    double f_novelEarly = 1.0;
	    {
	        const auto nf2 = std::dynamic_pointer_cast<NeuralField>(phenotype.getElement("nf 2"));
	        for (int i = 0; i < static_cast<int>(novelEarlyWindow); ++i) {
	            phenotype.step();
	            if (!nf2->getBumps().empty()) { f_novelEarly = 0.0; break; }
	        }
	    }

	    // Then reward (if it happens) being late rather than early
	    const double f_novelLate = iterationsUntilBump("nf 2", novelTarget, novelMax, novelTol);

	    // Combine: must pass early no-bump AND have late/weak responsiveness
	    const double f_novel = 0.6 * f_novelEarly + 0.4 * f_novelLate;
	    parameters.partialFitness.emplace_back(f_novel);

	    // --------------------------
	    // Final fitness
	    // --------------------------
	    parameters.fitness =
	        w_prime  * f_prime +
	        w_decay  * f_decay +
	        w_primed * f_primed +
	        w_novel  * f_novel;
	}

	void MemoryTrace::createPhenotypeEnvironment()
	{
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0,
				true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 1",
		   { GaussStimulusConstants::width, GaussStimulusConstants::amplitude*0.35, 80.0,
			 GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
		   { DimensionConstants::xSize, DimensionConstants::dx });
	}
}