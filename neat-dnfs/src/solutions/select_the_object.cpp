#include "solutions/select_the_object.h"

namespace neat_dnfs
{
	SelectTheObject::SelectTheObject(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "Select the object";
		// target fitness is 0.85
	}

	SolutionPtr SelectTheObject::clone() const
	{
		SelectTheObject solution(initialTopology);
		auto clonedSolution = std::make_shared<SelectTheObject>(solution);

		return clonedSolution;
	}

	void SelectTheObject::testPhenotype()
	{
		using namespace dnf_composer::element;
		parameters.fitness = 0.0;

		static constexpr int iterations = SimulationConstants::maxSimulationSteps;

		static constexpr double in_amp = 9.0;
		static constexpr double in_width = 10.0;
		static constexpr double out_amp = 9.0;
		static constexpr double out_width = 10.0;

		// nf 1 - input - small objects field (sof)
		// nf 2 - input - large object field (lof)
		// nf 3 - input - hand position field (hpf)
		// nf 4 - output - target action field (taf)

		static constexpr double wf1 = 0.125; // multi bump sof
		static constexpr double wf2 = 0.125; // sof creates a selective single bump in taf
		static constexpr double wf3 = 0.125; // lof single bump
		static constexpr double wf4 = 0.125; // lof pre-shapes taf
		static constexpr double wf5 = 0.125; // hpf inhibits sof activation in taf
		static constexpr double wf6 = 0.125; // lof wins over sof in taf
		static constexpr double wf7 = 0.125; // without hpf, sof wins over lof in taf
		static constexpr double wf8 = 0.125; // hpf inhibits lof activation in taf

		initSimulation();
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 2",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f1 = twoBumpsAtPositionWithAmplitudeAndWidth("nf 1",
			20.0, in_amp, in_width,
			80.0, in_amp, in_width);
		parameters.fitness = wf1 * f1;

		const double f2 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 4", { 20.0, 80.0 }, out_amp, out_width);
		parameters.fitness += wf2 * f2;
		const double f3 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", 50.0, in_amp, in_width);
		parameters.fitness += wf3 * f3;

		removeGaussianStimuli();
		initSimulation();
		addGaussianStimulus("nf 2",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
	
		runSimulation(iterations);
		const double f4 = preShapedness("nf 4");
		parameters.fitness += wf4 * f4;

		removeGaussianStimuli();
		initSimulation();
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 3",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f5 = oneBumpAtPositionWithAmplitudeAndWidth("nf 1", 80.0, in_amp, in_width);
		parameters.fitness += wf5 * f5;

		removeGaussianStimuliFromField("nf 3");
		addGaussianStimulus("nf 2",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 3",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f6 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", 50.0, in_amp, in_width);
		parameters.fitness += wf6 * f6;

		removeGaussianStimuliFromField("nf 3");
		runSimulation(iterations);
		const double f7 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 4", { 20.0, 80.0 }, out_amp, out_width);
		parameters.fitness += wf7 * f7;

		removeGaussianStimuli();
		initSimulation();
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 3",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f8 = negativePreShapednessAtPosition("nf 4", 20);
		parameters.fitness += wf8 * f8;

		removeGaussianStimuli();
	}

	void SelectTheObject::createPhenotypeEnvironment()
	{
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });

		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });

		addGaussianStimulus("nf 2",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });

		addGaussianStimulus("nf 3",
			{ GaussStimulusConstants::width, 0.0, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
	}
}
