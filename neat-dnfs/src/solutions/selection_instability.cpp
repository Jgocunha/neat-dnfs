#include "solutions/selection_instability.h"

namespace neat_dnfs
{
	SelectionInstability::SelectionInstability(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "Selection Instability";
	}

	SelectionInstability::SelectionInstability(const SolutionTopology& initialTopology,
		const dnf_composer::Simulation& phenotype)
		:Solution(initialTopology, phenotype)
	{
		name = "Selection Instability";
	}

	SolutionPtr SelectionInstability::clone() const
	{
		SelectionInstability solution(initialTopology);
		auto clonedSolution = std::make_shared<SelectionInstability>(solution);

		return clonedSolution;
	}

	SolutionPtr SelectionInstability::copy() const
	{
		SelectionInstability solution(initialTopology, phenotype);
		auto copy = std::make_shared<SelectionInstability>(solution);

		return copy;
	}

	void SelectionInstability::testPhenotype()
	{
		using namespace dnf_composer::element;
		parameters.fitness = 0.0;
		parameters.partialFitness.clear();
		static constexpr int iterations = SimulationConstants::maxSimulationSteps;

		static constexpr double p1 = 20.0;
	    static constexpr double p2 = 80.0;

	    static constexpr double in_amp   = 8.0;
	    static constexpr double in_width = 10.0;

	    static constexpr double out_amp   = 6.0;
	    static constexpr double out_width = 5.0;

	    initSimulation();

	    // Phase 1: two competing stimuli
	    addGaussianStimulus("nf 1",
	        dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, p1,
	          GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
	        dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });

	    addGaussianStimulus("nf 1",
	        dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, p2,
	          GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
	        dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });

	    runSimulation(iterations);

	    // Input sanity: two bumps present
	    const double f_in = twoBumpsAtPositionWithAmplitudeAndWidth("nf 1",
	        p1, in_amp, in_width,
	        p2, in_amp, in_width);
	    parameters.partialFitness.emplace_back(f_in);

	    // Output: WTA selection (check twice for stability)
	    const double f_sel_1 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth(
	        "nf 2", { p1, p2 }, out_amp, out_width);

	    runSimulation(iterations / 2);

		const double f_sel_2 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth(
	         "nf 2", { p1, p2 }, out_amp, out_width);

	    const double f_sel = 0.5 * (f_sel_1 + f_sel_2);
		parameters.partialFitness.emplace_back(f_sel);

	    // --- Phase 2: remove stimuli and relax (avoid memory solutions) ---
	    removeGaussianStimuli();
	    runSimulation(iterations);

	    const double f_relax_nf1 = closenessToRestingLevel("nf 1");
	    const double f_relax_nf2 = closenessToRestingLevel("nf 2");
	    parameters.partialFitness.emplace_back(f_relax_nf1);
	    parameters.partialFitness.emplace_back(f_relax_nf2);

	    // Weights: keep focus on selection
	    static constexpr double w_in    = 1 / 4.f;
	    static constexpr double w_sel   = 1 / 4.f;
	    static constexpr double w_r1    = 1 / 4.f;
	    static constexpr double w_r2    = 1 / 4.f;

	    parameters.fitness =
	        w_in  * f_in +
	        w_sel * f_sel +
	        w_r1  * f_relax_nf1 +
	        w_r2  * f_relax_nf2;
	}

	void SelectionInstability::createPhenotypeEnvironment()
	{
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0,
		GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
		dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });

		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0,
	GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
	}
}
