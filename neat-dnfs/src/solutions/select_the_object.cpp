#include "solutions/select_the_object.h"

namespace neat_dnfs
{
	SelectTheObject::SelectTheObject(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "Select the object";
		// target fitness is 0.95
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

		//initSimulation();
		//addGaussianStimulus("nf 1",
		//	{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
		//	{ DimensionConstants::xSize, DimensionConstants::dx });
		//addGaussianStimulus("nf 1",
		//	{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
		//	{ DimensionConstants::xSize, DimensionConstants::dx });
		//addGaussianStimulus("nf 2",
		//	{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
		//	{ DimensionConstants::xSize, DimensionConstants::dx });
		//runSimulation(iterations);

		//const double f1 = twoBumpsAtPositionWithAmplitudeAndWidth("nf 1",
		//	20.0, in_amp, in_width,
		//	80.0, in_amp, in_width);
		//const double f2 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 2", { 50.0 }, out_amp, out_width);
		//const double f3 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 4", { 20.0, 80.0 }, out_amp, out_width);

		//addGaussianStimulus("nf 3",
		//	{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
		//	{ DimensionConstants::xSize, DimensionConstants::dx });
		//runSimulation(iterations);

		//const double f4 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 4", { 80.0 }, out_amp, out_width);

		//removeGaussianStimuliFromField("nf 1");
		//removeGaussianStimuliFromField("nf 3");

		//addGaussianStimulus("nf 3",
		//	{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
		//	{ DimensionConstants::xSize, DimensionConstants::dx });
		//runSimulation(iterations);

		//const double f5 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 4", { 50.0 }, out_amp, out_width);

		//removeGaussianStimuli();

		//initSimulation();
		//addGaussianStimulus("nf 1",
		//	{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
		//	{ DimensionConstants::xSize, DimensionConstants::dx });
		//addGaussianStimulus("nf 1",
		//	{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
		//	{ DimensionConstants::xSize, DimensionConstants::dx });
		//addGaussianStimulus("nf 2",
		//	{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
		//	{ DimensionConstants::xSize, DimensionConstants::dx });
		//addGaussianStimulus("nf 3",
		//	{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
		//	{ DimensionConstants::xSize, DimensionConstants::dx });
		//runSimulation(iterations);

		//const double f6 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 4", { 50.0 }, out_amp, out_width);

		//removeGaussianStimuliFromField("nf 2");
		//removeGaussianStimuliFromField("nf 3");

		//addGaussianStimulus("nf 3",
		//	{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
		//	{ DimensionConstants::xSize, DimensionConstants::dx });
		//runSimulation(iterations);

		//const double f7 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 4", { 20.0 }, out_amp, out_width);

		//static constexpr double wf1 = 1 / 7.f;
		//static constexpr double wf2 = 1 / 7.f;
		//static constexpr double wf3 = 1 / 7.f;
		//static constexpr double wf4 = 1 / 7.f;
		//static constexpr double wf5 = 1 / 7.f;
		//static constexpr double wf6 = 1 / 7.f;
		//static constexpr double wf7 = 1 / 7.f;


		//parameters.fitness = wf1 * f1 + wf2 * f2 + wf3 * f3 + wf4 * f4 + wf5 * f5 + wf6 * f6 + wf7 * f7;

		//removeGaussianStimuli();
		// nf 1 - input - small objects field (sof)
		// nf 2 - input - large object field (lof)
		// nf 3 - input - hand position field (hpf)
		// nf 4 - output - target action field (taf)

		static constexpr double wf1		= 1 / 14.f; // multi bump sof
		static constexpr double wf2		= 1 / 14.f; // sof creates a selective single bump in taf
		static constexpr double wf3		= 1 / 14.f; // lof single bump
		static constexpr double wf4		= 1 / 14.f; // lof pre-shapes taf
		static constexpr double wf5		= 1 / 14.f; // sof + hpf create a selective single bump in taf (pos. 20)
		static constexpr double wf6		= 1 / 14.f; // sof + hpf create a selective single bump in taf (pos. 80)
		static constexpr double wf7		= 1 / 14.f; // hpf single bump
		static constexpr double wf8		= 1 / 14.f; // taf should be close to resting level just with hpf
		static constexpr double wf9		= 1 / 14.f; // lof + hpf create a single bump in taf
		static constexpr double wf10	= 1 / 14.f; // lof + sof + hpf create a selective single bump in taf (pos. 50)
		static constexpr double wf11	= 1 / 14.f; // 
		static constexpr double wf11_1	= 1 / 14.f; //
		static constexpr double wf11_2	= 1 / 14.f; //
		static constexpr double wf12	= 1 / 14.f; //


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
		const double f5 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", 80.0, in_amp, in_width);
		parameters.fitness += wf5 * f5;

		removeGaussianStimuliFromField("nf 3");
		addGaussianStimulus("nf 3",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f6 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", 20.0, in_amp, in_width);
		parameters.fitness += wf6 * f6;

		removeGaussianStimuli();
		initSimulation();
		addGaussianStimulus("nf 3",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f7 = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", 50.0, in_amp, in_width);
		parameters.fitness += wf7 * f7;

		const double f8 = closenessToRestingLevel("nf 4");
		parameters.fitness += wf8 * f8;

		addGaussianStimulus("nf 2",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f9 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", 50.0, in_amp, in_width);
		parameters.fitness += wf9 * f9;

		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f10 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", 50.0, in_amp, in_width);
		parameters.fitness += wf10 * f10;

		removeGaussianStimuliFromField("nf 3");
		addGaussianStimulus("nf 3",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f11 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", 80.0, in_amp, in_width);
		parameters.fitness += wf11 * f11;

		// new f11_1
		removeGaussianStimuliFromField("nf 3");
		addGaussianStimulus("nf 3",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f11_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", 50.0, in_amp, in_width);
		parameters.fitness += wf11_1 * f11_1;

		// new f11_2
		removeGaussianStimuliFromField("nf 3");
		addGaussianStimulus("nf 3",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f11_2 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", 20.0, in_amp, in_width);
		parameters.fitness += wf11_2 * f11_2;

		removeGaussianStimuli();
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
		const double f12 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 4", { 20.0, 80.0 }, out_amp, out_width);
		parameters.fitness += wf12 * f12;

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
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
	}
}
