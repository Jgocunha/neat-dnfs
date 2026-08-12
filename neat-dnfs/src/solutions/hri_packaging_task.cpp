#include "solutions/hri_packaging_task.h"

namespace neat_dnfs
{
	HRIPackagingTask::HRIPackagingTask(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "HRI Packaging Task" + AblationConstants::label;
	}

	HRIPackagingTask::HRIPackagingTask(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype)
		: Solution(initialTopology, phenotype)
	{
		name = "HRI Packaging Task" + AblationConstants::label;
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


		static constexpr double small_obj_pos_a = 10.0;
		static constexpr double small_obj_pos_b = 50.0;
		static constexpr double large_obj_pos = 30.0;

		removeGaussianStimuli();
		// arbitrary selection at output field based on small objects
		initSimulation();
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, small_obj_pos_a,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, small_obj_pos_b,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);

		const double f1_1 = twoBumpsAtPositionWithAmplitudeAndWidth("nf 1",
		   small_obj_pos_a, 5.0, 4.0,
		   small_obj_pos_b, 5.0, 4.0);
		const double f1_2 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth(
			"nf 4", { small_obj_pos_a, small_obj_pos_b }, 4.0, 4.0);
		const double f1 = 0.3f * f1_1 + 0.7f * f1_2;
		parameters.partialFitness.emplace_back(f1_1);
		parameters.partialFitness.emplace_back(f1_2);

		// hand position negatively pre-shapes the output field
		addGaussianStimulus("nf 3",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, large_obj_pos,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f2_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", large_obj_pos, 6.0, 6.0);
		const double f2_2 = negativePreShapednessAtPosition("nf 4", large_obj_pos);
		const double f2 = 0.3f * f2_1 + 0.7f * f2_2;
		parameters.partialFitness.emplace_back(f2_1);
		parameters.partialFitness.emplace_back(f2_2);

		// hand position biases the output field towards the non-targeted small object
		moveGaussianStimulusContinuously("gs nf 3 " + std::to_string(large_obj_pos), small_obj_pos_a, -5.0f);
		const double f3_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", small_obj_pos_b, 4.0, 4.0);
		moveGaussianStimulusContinuously("gs nf 3 " + std::to_string(large_obj_pos), small_obj_pos_b, 10.0f);
		const double f3_2 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", small_obj_pos_a, 4.0, 4.0);
		const double f3 = 0.5f * f3_1 + 0.5f * f3_2;
		parameters.partialFitness.emplace_back(f3_1);
		parameters.partialFitness.emplace_back(f3_2);

		// large object appears
		addGaussianStimulus("nf 2",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, large_obj_pos,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f4_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", large_obj_pos, 4.0, 4.0); // 2
		const double f4_2 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", small_obj_pos_a, 8.0, 4.0); // a
		moveGaussianStimulusContinuously("gs nf 3 " + std::to_string(large_obj_pos), large_obj_pos, -5.0f);
		const double f4_3 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", large_obj_pos, 10.0, 6.0); // c
		moveGaussianStimulusContinuously("gs nf 3 " + std::to_string(large_obj_pos), small_obj_pos_a, -5.0f);
		const double f4_4 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", small_obj_pos_b, 4.0, 4.0); // b
		moveGaussianStimulusContinuously("gs nf 3 " + std::to_string(large_obj_pos), large_obj_pos, 5.0f);
		const double f4_5 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", large_obj_pos, 10.0, 6.0); // c
		moveGaussianStimulusContinuously("gs nf 3 " + std::to_string(large_obj_pos), small_obj_pos_b, 5.0f);
		const double f4_6 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", small_obj_pos_a, 4.0, 4.0); // a
		const double f4 = 0.3f*f4_1 + 0.1f*f4_2 + 0.2f*f4_3 + 0.1f*f4_4 + 0.2f*f4_5 + 0.1f*f4_6;
		parameters.partialFitness.emplace_back(f4_1);
		parameters.partialFitness.emplace_back(f4_2);
		parameters.partialFitness.emplace_back(f4_3);
		parameters.partialFitness.emplace_back(f4_4);
		parameters.partialFitness.emplace_back(f4_5);
		parameters.partialFitness.emplace_back(f4_6);

		removeGaussianStimuli();
		
		parameters.fitness = 0.2f * f1 + 0.2f * f2 + 0.2f * f3 + 0.4f * f4;
	}


	// void HRIPackagingTask::testPhenotype()
	// {
	// 	using namespace dnf_composer::element;
	// 	parameters.fitness = 0.0;
	// 	parameters.partialFitness.clear();
	//
	// 	static constexpr int iterations = SimulationConstants::maxSimulationSteps;
	//
	// 	static constexpr double in_amp_nf1 = 9.0;
	// 	static constexpr double in_width_nf1 = 7.0;
	//
	// 	static constexpr double in_amp_nf2_3 = 5.0;
	// 	static constexpr double in_width_nf2_3 = 6.0;
	//
	// 	static constexpr double out_amp_nf4 = 2.0;
	// 	static constexpr double out_width_nf4 = 4.0;
	//
	// 	// nf 1 - input - small objects field (sof)
	// 	// nf 2 - input - large object field (lof)
	// 	// nf 3 - input - hand position field (hpf)
	// 	// nf 4 - output - target action field (taf)
	//
	// 	static constexpr double wf1		= 1 / 12.f; // multi bump sof
	// 	static constexpr double wf2		= 1 / 12.f; // sof creates a selective single bump in taf
	// 	static constexpr double wf3		= 1 / 12.f; // lof single bump
	// 	//static constexpr double wf4		= 1 / 12.f; // lof pre-shapes taf
	// 	static constexpr double wf5		= 1 / 12.f; // sof + hpf create a selective single bump in taf (pos. 20)
	// 	static constexpr double wf6		= 1 / 12.f; // sof + hpf create a selective single bump in taf (pos. 80)
	// 	static constexpr double wf7		= 1 / 12.f; // hpf single bump
	// 	static constexpr double wf8		= 1 / 12.f; // taf should be close to resting level just with hpf
	// 	static constexpr double wf9		= 1 / 12.f; // lof + hpf create a single bump in taf
	// 	static constexpr double wf10	= 1 / 12.f; // lof + sof + hpf create a selective single bump in taf (pos. 50)
	// 	static constexpr double wf11	= 1 / 12.f; //
	// 	static constexpr double wf11_	= 1 / 12.f; //
	// 	static constexpr double wf12	= 1 / 12.f; //
	//
	//
	// 	initSimulation();
	// 	addGaussianStimulus("nf 1",
	// 		{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 10.0, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
	// 		{ DimensionConstants::xSize, DimensionConstants::dx });
	// 	addGaussianStimulus("nf 1",
	// 		{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
	// 		{ DimensionConstants::xSize, DimensionConstants::dx });
	// 	addGaussianStimulus("nf 2",
	// 		{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 30.0, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
	// 		{ DimensionConstants::xSize, DimensionConstants::dx });
	// 	runSimulation(iterations);
	// 	const double f1 = twoBumpsAtPositionWithAmplitudeAndWidth("nf 1",
	// 		10.0, in_amp_nf1, in_width_nf1,
	// 		50.0, in_amp_nf1, in_width_nf1);
	// 	parameters.fitness = wf1 * f1;
	// 	parameters.partialFitness.emplace_back(f1);
	//
	// 	const double f2 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 4", { 10.0, 50.0 }, out_amp_nf4, out_width_nf4);
	// 	parameters.fitness += wf2 * f2;
	// 	parameters.partialFitness.emplace_back(f2);
	// 	const double f3 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", 30.0, in_amp_nf2_3, in_width_nf2_3);
	// 	parameters.fitness += wf3 * f3;
	// 	parameters.partialFitness.emplace_back(f3);
	//
	// 	// removeGaussianStimuli();
	// 	// initSimulation();
	// 	// addGaussianStimulus("nf 2",
	// 	// 	{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 30.0, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
	// 	// 	{ DimensionConstants::xSize, DimensionConstants::dx });
	// 	//
	// 	// runSimulation(iterations);
	// 	// const double f4 = preShapednessAtPosition("nf 4", 30.0);
	// 	// parameters.fitness += wf4 * f4;
	// 	// parameters.partialFitness.emplace_back(f4);
	//
	// 	removeGaussianStimuli();
	// 	initSimulation();
	// 	addGaussianStimulus("nf 1",
	// 		{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 10.0, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
	// 		{ DimensionConstants::xSize, DimensionConstants::dx });
	// 	addGaussianStimulus("nf 1",
	// 		{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
	// 		{ DimensionConstants::xSize, DimensionConstants::dx });
	// 	addGaussianStimulus("nf 3",
	// 		{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 10.0, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
	// 		{ DimensionConstants::xSize, DimensionConstants::dx });
	// 	runSimulation(iterations);
	// 	const double f5 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", 50.0, out_amp_nf4, out_width_nf4);
	// 	parameters.fitness += wf5 * f5;
	// 	parameters.partialFitness.emplace_back(f5);
	//
	// 	removeGaussianStimuliFromField("nf 3");
	// 	addGaussianStimulus("nf 3",
	// 		{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
	// 		{ DimensionConstants::xSize, DimensionConstants::dx });
	// 	runSimulation(iterations);
	// 	const double f6 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", 10.0, out_amp_nf4, out_width_nf4);
	// 	parameters.fitness += wf6 * f6;
	// 	parameters.partialFitness.emplace_back(f6);
	//
	// 	removeGaussianStimuli();
	// 	initSimulation();
	// 	addGaussianStimulus("nf 3",
	// 		{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 30.0, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
	// 		{ DimensionConstants::xSize, DimensionConstants::dx });
	// 	runSimulation(iterations);
	// 	const double f7 = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", 30.0, in_amp_nf2_3, in_width_nf2_3);
	// 	parameters.fitness += wf7 * f7;
	// 	parameters.partialFitness.emplace_back(f7);
	//
	// 	//const double f8 = closenessToRestingLevel("nf 4");
	// 	//const double f8 = negativePreShapednessAtPosition("nf 4", 50);
	// 	const double f8 = noBumps("nf 4");
	// 	parameters.fitness += wf8 * f8;
	// 	parameters.partialFitness.emplace_back(f8);
	//
	// 	addGaussianStimulus("nf 2",
	// 		{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 30.0, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
	// 		{ DimensionConstants::xSize, DimensionConstants::dx });
	// 	runSimulation(iterations);
	// 	const double f9 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", 30.0, out_amp_nf4, out_width_nf4);
	// 	parameters.fitness += wf9 * f9;
	// 	parameters.partialFitness.emplace_back(f9);
	//
	// 	addGaussianStimulus("nf 1",
	// 		{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 10.0, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
	// 		{ DimensionConstants::xSize, DimensionConstants::dx });
	// 	addGaussianStimulus("nf 1",
	// 		{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
	// 		{ DimensionConstants::xSize, DimensionConstants::dx });
	// 	runSimulation(iterations);
	// 	const double f10 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", 30.0, out_amp_nf4, out_width_nf4);
	// 	parameters.fitness += wf10 * f10;
	// 	parameters.partialFitness.emplace_back(f10);
	//
	// 	removeGaussianStimuliFromField("nf 3");
	// 	addGaussianStimulus("nf 3",
	// 		{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 10.0, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
	// 		{ DimensionConstants::xSize, DimensionConstants::dx });
	// 	runSimulation(iterations);
	// 	const double f11 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", 50.0, out_amp_nf4, out_width_nf4);
	// 	parameters.fitness += wf11 * f11;
	// 	parameters.partialFitness.emplace_back(f11);
	//
	// 	// new f11_1
	// 	removeGaussianStimuliFromField("nf 3");
	// 	addGaussianStimulus("nf 3",
	// 		{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 30.0, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
	// 		{ DimensionConstants::xSize, DimensionConstants::dx });
	// 	runSimulation(iterations);
	// 	const double f11_ = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", 30.0, out_amp_nf4, out_width_nf4);
	// 	parameters.fitness += wf11_ * f11_;
	// 	parameters.partialFitness.emplace_back(f11_);
	//
	// 	removeGaussianStimuli();
	// 	initSimulation();
	// 	addGaussianStimulus("nf 1",
	// 		{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 10.0, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
	// 		{ DimensionConstants::xSize, DimensionConstants::dx });
	// 	addGaussianStimulus("nf 1",
	// 		{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
	// 		{ DimensionConstants::xSize, DimensionConstants::dx });
	// 	addGaussianStimulus("nf 2",
	// 		{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 30.0, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
	// 		{ DimensionConstants::xSize, DimensionConstants::dx });
	// 	runSimulation(iterations);
	// 	const double f12 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 4", { 10.0, 50.0 }, out_amp_nf4, out_width_nf4);
	// 	parameters.fitness += wf12 * f12;
	// 	parameters.partialFitness.emplace_back(f12);
	//
	// 	removeGaussianStimuli();
	// }

	void HRIPackagingTask::createPhenotypeEnvironment()
	{
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 10.0f, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });

		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0f, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });

		addGaussianStimulus("nf 2",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 30.0f, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });

		addGaussianStimulus("nf 2",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, 0.0f, 0.0f, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });

		addGaussianStimulus("nf 3",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 30.0f, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
	}
}