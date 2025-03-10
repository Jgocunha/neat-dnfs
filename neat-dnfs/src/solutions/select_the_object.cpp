#include "solutions/select_the_object.h"

namespace neat_dnfs
{
	SelectTheObject::SelectTheObject(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "Select the object";
		// target fitness is 0.95
	}

	SelectTheObject::SelectTheObject(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype)
		: Solution(initialTopology, phenotype)
	{
		name = "Select the object";
		// target fitness is 0.95
	}

	SolutionPtr SelectTheObject::clone() const
	{
		SelectTheObject solution(initialTopology, phenotype);
		auto clonedSolution = std::make_shared<SelectTheObject>(solution);

		return clonedSolution;
	}

	void SelectTheObject::testPhenotype()
	{
		using namespace dnf_composer::element;
		parameters.fitness = 0.0;
		parameters.partialFitness.clear();

		static constexpr int iterations = SimulationConstants::maxSimulationSteps;

		static constexpr double in_amp = 10.0;
		static constexpr double in_width = 12.0;
		static constexpr double out_amp = 3.0;
		static constexpr double out_width = 10.0;

		// nf 1 - input - small objects field (sof)
		// nf 2 - input - large object field (lof)
		// nf 3 - input - hand position field (hpf)
		// nf 4 - output - target action field (taf)

		static constexpr double wf1		= 0.3; // multi bump sof
		static constexpr double wf2		= 0.7; // sof creates a selective single bump in taf
		static constexpr double wf_1_2		= 1 / 5.f; // behaviour 1

		static constexpr double wf3		= 0.2; // lof single bump
		static constexpr double wf4		= 0.8; // lof pre-shapes taf
		static constexpr double wf_3_4		= 1 / 5.f; // behaviour 2

		static constexpr double wf5		= 0.4; // sof + hpf create a selective single bump in taf (pos. 20)
		static constexpr double wf6		= 0.4; // sof + hpf create a selective single bump in taf (pos. 80)
		static constexpr double wf7		= 0.2; // hpf single bump
		static constexpr double wf_5_6_7	= 1 / 5.f; // behaviour 3

		static constexpr double wf8		= 0.2; // taf should be close to resting level just with hpf
		static constexpr double wf9		= 0.4; // lof + hpf create a single bump in taf
		static constexpr double wf10	= 0.4; // lof + sof + hpf create a selective single bump in taf (pos. 50)
		static constexpr double wf_8_9_10	= 0.5; // behaviour 4

		static constexpr double wf11	= 0.25f; // switch from lof bump position to sof bump position 1 in taf
		static constexpr double wf12	= 0.25f; // switch from sof position 1 bump position to sof bump position 2 in taf
		static constexpr double wf13	= 0.25f; // switch from sof position 2 bump position to sof bump position 1 in taf
		static constexpr double wf14	= 0.25f; // without hpf bump, sof + lof create a selective single bump in taf
		static constexpr double wf_11_12_13_14 = 1 / 5.f; // behaviour 5

		initSimulation();
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		addGaussianStimulus("nf 2",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f1 = twoBumpsAtPositionWithAmplitudeAndWidth("nf 1",
			20.0, in_amp, in_width,
			80.0, in_amp, in_width);
		parameters.partialFitness.emplace_back(f1);
		//parameters.fitness = wf1 * f1;

		const double f2 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 4", { 20.0, 80.0 }, out_amp, out_width);
		parameters.partialFitness.emplace_back(f2);
		//parameters.fitness += wf2 * f2;

		//behaviour 1
		parameters.fitness += wf_1_2 * ( wf1 * f1 + wf2 * f2);

		const double f3 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 2", { 50.0 }, in_amp, in_width);
		parameters.partialFitness.emplace_back(f3);
		//parameters.fitness += wf3 * f3;

		removeGaussianStimuli();
		initSimulation();
		addGaussianStimulus("nf 2",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });

		runSimulation(iterations);
		const double f4 = preShapedness("nf 4");
		parameters.partialFitness.emplace_back(f4);
		//parameters.fitness += wf4 * f4;

		// behaviour 2
		parameters.fitness += wf_3_4 * (wf3 * f3 + wf4 * f4);

		removeGaussianStimuli();
		initSimulation();
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		addGaussianStimulus("nf 3",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f5 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", 80.0, out_amp, out_width);
		parameters.partialFitness.emplace_back(f5);
		//parameters.fitness += wf5 * f5;
		const double f7_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", 20.0, in_amp, in_width);

		removeGaussianStimuliFromField("nf 3");
		addGaussianStimulus("nf 3",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f6 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", 20.0, out_amp, out_width);
		parameters.partialFitness.emplace_back(f6);
		//parameters.fitness += wf6 * f6;
		const double f7_2 = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", 80.0, in_amp, in_width);

		const double f7 = (f7_1 + f7_2) / 2.0;
		parameters.partialFitness.emplace_back(f7);
		//parameters.fitness += wf7 * f7;

		// behaviour 3
		parameters.fitness += wf_5_6_7 * (wf5 * f5 + wf6 * f6 + wf7 * f7);

		removeGaussianStimuli();

		initSimulation();
		addGaussianStimulus("nf 3",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		
		const double f8 = closenessToRestingLevel("nf 4");
		//parameters.fitness += wf8 * f8;
		parameters.partialFitness.emplace_back(f8);

		addGaussianStimulus("nf 2",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f9 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", 50.0, out_amp, out_width);
		//parameters.fitness += wf9 * f9;
		parameters.partialFitness.emplace_back(f9);

		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f10 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", 50.0, out_amp, out_width);
		//parameters.fitness += wf10 * f10;
		parameters.partialFitness.emplace_back(f10);

		// behaviour 4
		parameters.fitness += wf_8_9_10 * (wf8 * f8 + wf9 * f9 + wf10 * f10);

		removeGaussianStimuliFromField("nf 3");
		addGaussianStimulus("nf 3",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f11 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", 80.0, in_amp, in_width);
		//parameters.fitness += wf11 * f11;
		parameters.partialFitness.emplace_back(f11);

		removeGaussianStimuliFromField("nf 3");
		addGaussianStimulus("nf 3",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f12 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", 50.0, in_amp, in_width);
		//parameters.fitness += wf12 * f12;
		parameters.partialFitness.emplace_back(f12);

		removeGaussianStimuliFromField("nf 3");
		addGaussianStimulus("nf 3",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f13 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", 20.0, in_amp, in_width);
		//parameters.fitness += wf13 * f13;
		parameters.partialFitness.emplace_back(f13);

		removeGaussianStimuli();
		initSimulation();
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		addGaussianStimulus("nf 2",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f14 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 4", { 20.0, 80.0 }, out_amp, out_width);
		//parameters.fitness += wf14 * f14;
		parameters.partialFitness.emplace_back(f14);

		// behaviour 5
		parameters.fitness += wf_11_12_13_14 * (wf11 * f11 + wf12 * f12 + wf13 * f13 + wf14 * f14);

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
