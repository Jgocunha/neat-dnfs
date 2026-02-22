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

		static constexpr double in_amp_nf1 = 9.0;
		static constexpr double in_width_nf1 = 7.0;

		static constexpr double in_amp_nf2_3 = 5.0;
		static constexpr double in_width_nf2_3 = 6.0;

		static constexpr double out_amp_nf4 = 2.0;
		static constexpr double out_width_nf4 = 4.0;

		// nf 1 - input - small objects field (sof)
		// nf 2 - input - large object field (lof)
		// nf 3 - input - hand position field (hpf)
		// nf 4 - output - target action field (taf)

		static constexpr double wf1		= 1 / 13.f; // multi bump sof
		static constexpr double wf2		= 1 / 13.f; // sof creates a selective single bump in taf
		static constexpr double wf3		= 1 / 13.f; // lof single bump
		static constexpr double wf4		= 1 / 13.f; // lof pre-shapes taf
		static constexpr double wf5		= 1 / 13.f; // sof + hpf create a selective single bump in taf (pos. 20)
		static constexpr double wf6		= 1 / 13.f; // sof + hpf create a selective single bump in taf (pos. 80)
		static constexpr double wf7		= 1 / 13.f; // hpf single bump
		static constexpr double wf8		= 1 / 13.f; // taf should be close to resting level just with hpf
		static constexpr double wf9		= 1 / 13.f; // lof + hpf create a single bump in taf
		static constexpr double wf10	= 1 / 13.f; // lof + sof + hpf create a selective single bump in taf (pos. 50)
		static constexpr double wf11	= 1 / 13.f; //
		static constexpr double wf11_	= 1 / 13.f; //
		static constexpr double wf12	= 1 / 13.f; //


		initSimulation();
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 10.0, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 2",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 30.0, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f1 = twoBumpsAtPositionWithAmplitudeAndWidth("nf 1",
			10.0, in_amp_nf1, in_width_nf1,
			50.0, in_amp_nf1, in_width_nf1);
		parameters.fitness = wf1 * f1;
		parameters.partialFitness.emplace_back(f1);

		const double f2 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 4", { 10.0, 50.0 }, out_amp_nf4, out_width_nf4);
		parameters.fitness += wf2 * f2;
		parameters.partialFitness.emplace_back(f2);
		const double f3 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", 30.0, in_amp_nf2_3, in_width_nf2_3);
		parameters.fitness += wf3 * f3;
		parameters.partialFitness.emplace_back(f3);

		removeGaussianStimuli();
		initSimulation();
		addGaussianStimulus("nf 2",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 30.0, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			{ DimensionConstants::xSize, DimensionConstants::dx });

		runSimulation(iterations);
		const double f4 = preShapednessAtPosition("nf 4", 30.0);
		parameters.fitness += wf4 * f4;
		parameters.partialFitness.emplace_back(f4);

		removeGaussianStimuli();
		initSimulation();
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 10.0, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 3",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 10.0, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f5 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", 50.0, out_amp_nf4, out_width_nf4);
		parameters.fitness += wf5 * f5;
		parameters.partialFitness.emplace_back(f5);

		removeGaussianStimuliFromField("nf 3");
		addGaussianStimulus("nf 3",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f6 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", 10.0, out_amp_nf4, out_width_nf4);
		parameters.fitness += wf6 * f6;
		parameters.partialFitness.emplace_back(f6);

		removeGaussianStimuli();
		initSimulation();
		addGaussianStimulus("nf 3",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 30.0, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f7 = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", 30.0, in_amp_nf2_3, in_width_nf2_3);
		parameters.fitness += wf7 * f7;
		parameters.partialFitness.emplace_back(f7);

		//const double f8 = closenessToRestingLevel("nf 4");
		//const double f8 = negativePreShapednessAtPosition("nf 4", 50);
		const double f8 = noBumps("nf 4");
		parameters.fitness += wf8 * f8;
		parameters.partialFitness.emplace_back(f8);

		addGaussianStimulus("nf 2",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 30.0, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f9 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", 30.0, out_amp_nf4, out_width_nf4);
		parameters.fitness += wf9 * f9;
		parameters.partialFitness.emplace_back(f9);

		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 10.0, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f10 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", 30.0, out_amp_nf4, out_width_nf4);
		parameters.fitness += wf10 * f10;
		parameters.partialFitness.emplace_back(f10);

		removeGaussianStimuliFromField("nf 3");
		addGaussianStimulus("nf 3",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 10.0, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f11 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", 50.0, out_amp_nf4, out_width_nf4);
		parameters.fitness += wf11 * f11;
		parameters.partialFitness.emplace_back(f11);

		// new f11_1
		removeGaussianStimuliFromField("nf 3");
		addGaussianStimulus("nf 3",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 30.0, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f11_ = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", 30.0, out_amp_nf4, out_width_nf4);
		parameters.fitness += wf11_ * f11_;
		parameters.partialFitness.emplace_back(f11_);

		removeGaussianStimuli();
		initSimulation();
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 10.0, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 2",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 30.0, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f12 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 4", { 10.0, 50.0 }, out_amp_nf4, out_width_nf4);
		parameters.fitness += wf12 * f12;
		parameters.partialFitness.emplace_back(f12);

		removeGaussianStimuli();
	 }

	void HRIPackagingTask::createPhenotypeEnvironment()
	{
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 10.0, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			{ DimensionConstants::xSize, DimensionConstants::dx });

		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			{ DimensionConstants::xSize, DimensionConstants::dx });

		addGaussianStimulus("nf 2",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 30.0, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			{ DimensionConstants::xSize, DimensionConstants::dx });

		addGaussianStimulus("nf 3",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 30.0, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			{ DimensionConstants::xSize, DimensionConstants::dx });
	}
}