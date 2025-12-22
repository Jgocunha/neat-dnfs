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
        static constexpr double posA = 30.0;
        static constexpr double posB = 70.0;

        // Timing (tune to your deltaT / bump detector)
        static constexpr int t_encode = SimulationConstants::maxSimulationSteps / 3;   // strong stimulus
        static constexpr int t_clear  = SimulationConstants::maxSimulationSteps / 6;   // relax after removal
        static constexpr int t_delay  = SimulationConstants::maxSimulationSteps / 3;   // latent period
        static constexpr int t_probe  = SimulationConstants::maxSimulationSteps / 3;   // probe window

        // Strong vs. weak stimulus
        static constexpr double strongAmp = GaussStimulusConstants::amplitude; // your default (e.g., 20)
        static constexpr double strongWid = GaussStimulusConstants::width;

        // Weak should be below “easy win” threshold so trace matters
        static constexpr double weakAmp = GaussStimulusConstants::amplitude * 0.40;
        static constexpr double weakWid = GaussStimulusConstants::width;

        // Output bump targets (rough; your bump detector already tolerates)
        static constexpr double outAmpEncode  = 18.0;
        static constexpr double outWidEncode  = 10.0;

        static constexpr double outAmpProbe   = 12.0;
        static constexpr double outWidProbe   = 12.0;

        // -------------------------
        // Phase A: Encode (strong input at A)
        // -------------------------
        initSimulation();

        addGaussianStimulus("nf 1",
            { strongWid, strongAmp, posA,
              GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
            { DimensionConstants::xSize, DimensionConstants::dx });

        runSimulation(t_encode);

        const double f_encode = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", posA, outAmpEncode, outWidEncode);
        parameters.partialFitness.emplace_back(f_encode);

        // -------------------------
        // Phase B: Clear (remove input; nf2 should NOT keep a bump)
        // -------------------------
        removeGaussianStimuli();
        runSimulation(t_clear);

        const double f_noWM_1 = noBumps("nf 2");                 // punish working memory in output
        const double f_in_relax = closenessToRestingLevel("nf 1"); // optional sanity: perception relaxes
        parameters.partialFitness.emplace_back(f_noWM_1);
        parameters.partialFitness.emplace_back(f_in_relax);

        // -------------------------
        // Phase C: Delay (still no bump in nf2)
        // -------------------------
        runSimulation(t_delay);
        const double f_noWM_2 = noBumps("nf 2");
        parameters.partialFitness.emplace_back(f_noWM_2);

        // -------------------------
        // Phase D: Probe (two equal weak inputs A and B)
        // Expect bias toward A due to latent trace (hidden field).
        // -------------------------
        addGaussianStimulus("nf 1",
            { weakWid, weakAmp, posA,
              GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
            { DimensionConstants::xSize, DimensionConstants::dx });

        addGaussianStimulus("nf 1",
            { weakWid, weakAmp, posB,
              GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
            { DimensionConstants::xSize, DimensionConstants::dx });

        // Reward fast decision AND correct location
        const double f_fast = iterationsUntilBump("nf 2",
            /*targetIterations=*/ t_probe * 0.25,
            /*maxIterations=*/    t_probe,
            /*tolerance=*/        t_probe * 0.15);

        // After probe window, enforce that the bump is at A (history-biased choice)
        runSimulation(t_probe);
        const double f_choice = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", posA, outAmpProbe, outWidProbe);

        parameters.partialFitness.emplace_back(f_fast);
        parameters.partialFitness.emplace_back(f_choice);

        // -------------------------
        // Fitness weights
        // -------------------------
        // Dominant term: history-dependent biased choice under symmetric input.
        // Strong penalties for output holding working memory.
        static constexpr double w_encode   = 0.15;
        static constexpr double w_noWM     = 0.35; // average of noWM checkpoints
        static constexpr double w_fast     = 0.10;
        static constexpr double w_choice   = 0.35;
        static constexpr double w_relax    = 0.05;

        const double f_noWM = 0.5 * (f_noWM_1 + f_noWM_2);

        parameters.fitness =
            w_encode * f_encode +
            w_noWM   * f_noWM +
            w_fast   * f_fast +
            w_choice * f_choice +
            w_relax  * f_in_relax;
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