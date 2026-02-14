#include "solutions/hri_packaging_task.h"

namespace neat_dnfs
{
	HRIPackagingTask::HRIPackagingTask(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "HRI Packaging Task";
	}

	HRIPackagingTask::HRIPackagingTask(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype)
		: Solution(initialTopology, phenotype)
	{
		name = "HRI Packaging Task";
	}

	SolutionPtr HRIPackagingTask::clone() const
	{
		HRIPackagingTask solution(initialTopology, phenotype);
		auto clonedSolution = std::make_shared<HRIPackagingTask>(solution);

		return clonedSolution;
	}

	SolutionPtr HRIPackagingTask::copy() const
	{
		HRIPackagingTask solution(initialTopology, phenotype);
		auto copy = std::make_shared<HRIPackagingTask>(solution);

		return copy;
	}

	void HRIPackagingTask::testPhenotype()
	{
		using namespace dnf_composer::element;
		parameters.fitness = 0.0;
		parameters.partialFitness.clear();

		static constexpr int iterations = SimulationConstants::maxSimulationSteps;

		static constexpr double in_amp = 9.0;
		static constexpr double in_width = 10.0;
		static constexpr double out_amp = 9.0;
		static constexpr double out_width = 10.0;

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
		static constexpr double wf11_	= 1 / 14.f; //
		static constexpr double wf12	= 1 / 14.f; //
		static constexpr double wf13	= 1 / 14.f;


		initSimulation();
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 40.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 2",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 60.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 2",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f1 = twoBumpsAtPositionWithAmplitudeAndWidth("nf 1",
			20.0, in_amp, in_width,
			80.0, in_amp, in_width);
		parameters.fitness = wf1 * f1;
		parameters.partialFitness.emplace_back(f1);

		const double f2 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 4", { 20.0, 40.0 }, out_amp, out_width);
		parameters.fitness += wf2 * f2;
		parameters.partialFitness.emplace_back(f2);
		const double f3 = twoBumpsAtPositionWithAmplitudeAndWidth("nf 2",
			60.0, in_amp, in_width,
			80.0, in_amp, in_width);
		parameters.fitness += wf3 * f3;
		parameters.partialFitness.emplace_back(f3);

		removeGaussianStimuli();
		initSimulation();
		addGaussianStimulus("nf 2",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });

		runSimulation(iterations);
		const double f4 = preShapednessAtPosition("nf 4", 50.0);
		parameters.fitness += wf4 * f4;
		parameters.partialFitness.emplace_back(f4);

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
		parameters.partialFitness.emplace_back(f5);

		removeGaussianStimuliFromField("nf 3");
		addGaussianStimulus("nf 3",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f6 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", 20.0, in_amp, in_width);
		parameters.fitness += wf6 * f6;
		parameters.partialFitness.emplace_back(f6);

		removeGaussianStimuli();
		initSimulation();
		addGaussianStimulus("nf 3",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f7 = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", 50.0, in_amp, in_width);
		parameters.fitness += wf7 * f7;
		parameters.partialFitness.emplace_back(f7);

		//const double f8 = closenessToRestingLevel("nf 4");
		//const double f8 = negativePreShapednessAtPosition("nf 4", 50);
		const double f8 = noBumps("nf 4");
		parameters.fitness += wf8 * f8;
		parameters.partialFitness.emplace_back(f8);

		addGaussianStimulus("nf 2",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f9 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", 50.0, in_amp, in_width);
		parameters.fitness += wf9 * f9;
		parameters.partialFitness.emplace_back(f9);

		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f10 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", 50.0, in_amp, in_width);
		parameters.fitness += wf10 * f10;
		parameters.partialFitness.emplace_back(f10);

		removeGaussianStimuliFromField("nf 3");
		addGaussianStimulus("nf 3",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f11 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", 80.0, in_amp, in_width);
		parameters.fitness += wf11 * f11;
		parameters.partialFitness.emplace_back(f11);

		// new f11_1
		removeGaussianStimuliFromField("nf 3");
		addGaussianStimulus("nf 3",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f11_ = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", 50.0, in_amp, in_width);
		parameters.fitness += wf11_ * f11_;
		parameters.partialFitness.emplace_back(f11_);

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
		parameters.partialFitness.emplace_back(f12);

		removeGaussianStimuli();
		initSimulation();
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 2",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f13 = noBumps("nf 4");
		parameters.fitness += wf13 * f13;
		parameters.partialFitness.emplace_back(f13);

		removeGaussianStimuli();
	}

	void HRIPackagingTask::createPhenotypeEnvironment()
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