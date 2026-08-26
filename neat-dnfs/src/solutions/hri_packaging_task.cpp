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


		static constexpr double small_obj_pos_a = 10.0;
		static constexpr double small_obj_pos_b = 50.0;
		static constexpr double large_obj_pos = 30.0;

		removeGaussianStimuli();
		// arbitrary selection at output field based on small objects
		initSimulation();
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, small_obj_pos_a,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, small_obj_pos_b,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);

		const double f1_1 = twoBumpsAtPositionWithAmplitudeAndWidth("nf 1",
		   small_obj_pos_a, 5.0, 4.0,
		   small_obj_pos_b, 5.0, 4.0);
		const double f1_2 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth(
			"nf 4", { small_obj_pos_a, small_obj_pos_b }, 4.0, 4.0);
		//const double f1 = 0.3f * f1_1 + 0.7f * f1_2;
		//parameters.partialFitness.emplace_back(f1_1); // small object detection
		//parameters.partialFitness.emplace_back(f1_2); // complementary action selection

		// hand position negatively pre-shapes the output field
		addGaussianStimulus("nf 3",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, large_obj_pos,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f2_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", large_obj_pos, 6.0, 6.0);
		const double f2_2 = negativePreShapednessAtPosition("nf 4", large_obj_pos);
		//const double f2 = 0.3f * f2_1 + 0.7f * f2_2;
		//parameters.partialFitness.emplace_back(f2_1); // hand position detection
		//parameters.partialFitness.emplace_back(f2_2); // action inhibition

		// hand position biases the output field towards the non-targeted small object
		moveGaussianStimulusContinuously("gs nf 3 " + std::to_string(large_obj_pos), small_obj_pos_a, -5.0f);
		const double f3_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", small_obj_pos_b, 4.0, 4.0);
		moveGaussianStimulusContinuously("gs nf 3 " + std::to_string(large_obj_pos), small_obj_pos_b, 10.0f);
		const double f3_2 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", small_obj_pos_a, 4.0, 4.0);
		//const double f3 = 0.5f * f3_1 + 0.5f * f3_2;
		//parameters.partialFitness.emplace_back(f3_1); // complementary action selection a
		//parameters.partialFitness.emplace_back(f3_2); // complementary action selection b

		// large object appears
		addGaussianStimulus("nf 2",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, large_obj_pos,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			{ DimensionConstants::xSize, DimensionConstants::dx });
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
		//const double f4 = 0.1f*f4_1 + 0.15f*f4_2 + 0.2f*f4_3 + 0.2f*f4_4 + 0.2f*f4_5 + 0.15f*f4_6;
		//parameters.partialFitness.emplace_back(f4_1); // large object detection
		//parameters.partialFitness.emplace_back(f4_2); // complementary action selection a
		//parameters.partialFitness.emplace_back(f4_3); // assistive action selection
		//parameters.partialFitness.emplace_back(f4_4); // complementary action selection b
		//parameters.partialFitness.emplace_back(f4_5); // assistive action selection
		//parameters.partialFitness.emplace_back(f4_6); // complementary action selection a

		removeGaussianStimuli();
		runSimulation(iterations/2);
		const double f5_1 = closenessToRestingLevel("nf 1");
		const double f5_2 = closenessToRestingLevel("nf 2");
		const double f5_3 = closenessToRestingLevel("nf 3");
		const double f5_4 = closenessToRestingLevel("nf 4");
		//const double f5 = 0.25f*f5_1 + 0.25f*f5_2 + 0.25f*f5_3 + 0.25f*f5_4;
		//parameters.partialFitness.emplace_back(f5_1); // small object detection
		//parameters.partialFitness.emplace_back(f5_2); // large object detection
		//parameters.partialFitness.emplace_back(f5_3); // hand position detection
		//parameters.partialFitness.emplace_back(f5_4); // action inhibition

		const double f1 = f1_1*0.5f + f5_1*0.5f; // small object detection
		const double f2 = f4_1*0.5f + f5_2*0.5f; // large object detection
		const double f3 = f2_1*0.5f + f5_3*0.5f; // hand object detection
		const double f4 = f4_3*0.5f + f4_5*0.5f; // assistive action selection
		const double f5 = f3_1*1/3.f + f4_2*1/3.f + f4_6*1/3.f; // complementary action selection a
		const double f6 = f3_2*1/3.f + f4_4*1/3.f + f1_2*1/3.f; // complementary action selection b
		const double f7 = f2_2*0.5f + f5_4*0.5f; // action inhibition

		parameters.partialFitness.emplace_back(f1);
		parameters.partialFitness.emplace_back(f2);
		parameters.partialFitness.emplace_back(f3);
		parameters.partialFitness.emplace_back(f4);
		parameters.partialFitness.emplace_back(f5);
		parameters.partialFitness.emplace_back(f6);
		parameters.partialFitness.emplace_back(f7);

		parameters.fitness = f1*1/7.f + f2*1/7.f + f3*1/7.f + f4*1/7.f + f5*1/7.f + f6*1/7.f + f7*1/7.f;
	}

	//void HRIPackagingTask::testPhenotype()
	//{
	//	using namespace dnf_composer::element;
	//	parameters.fitness = 0.0;
	//	parameters.partialFitness.clear();
	//	static constexpr int iterations = SimulationConstants::maxSimulationSteps;
//
//
	//	static constexpr double small_obj_pos_a = 10.0;
	//	static constexpr double small_obj_pos_b = 50.0;
	//	static constexpr double large_obj_pos = 30.0;
//
	//	removeGaussianStimuli();
	//	// arbitrary selection at output field based on small objects
	//	initSimulation();
	//	addGaussianStimulus("nf 1",
	//		{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, small_obj_pos_a,
	//			GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
	//		{ DimensionConstants::xSize, DimensionConstants::dx });
	//	addGaussianStimulus("nf 1",
	//		{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, small_obj_pos_b,
	//			GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
	//		{ DimensionConstants::xSize, DimensionConstants::dx });
	//	runSimulation(iterations);
//
	//	const double f1_1 = twoBumpsAtPositionWithAmplitudeAndWidth("nf 1",
	//	   small_obj_pos_a, 5.0, 4.0,
	//	   small_obj_pos_b, 5.0, 4.0);
	//	const double f1_2 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth(
	//		"nf 4", { small_obj_pos_a, small_obj_pos_b }, 4.0, 4.0);
	//	const double f1 = 0.3f * f1_1 + 0.7f * f1_2;
	//	parameters.partialFitness.emplace_back(f1_1); // small object detection
	//	parameters.partialFitness.emplace_back(f1_2); // output field selection
//
	//	// hand position negatively pre-shapes the output field
	//	addGaussianStimulus("nf 3",
	//		{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, large_obj_pos,
	//			GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
	//		{ DimensionConstants::xSize, DimensionConstants::dx });
	//	runSimulation(iterations);
	//	const double f2_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", large_obj_pos, 6.0, 6.0);
	//	const double f2_2 = negativePreShapednessAtPosition("nf 4", large_obj_pos);
	//	const double f2 = 0.3f * f2_1 + 0.7f * f2_2;
	//	parameters.partialFitness.emplace_back(f2_1); // hand position detection
	//	parameters.partialFitness.emplace_back(f2_2); // action inhibition
//
	//	// hand position biases the output field towards the non-targeted small object
	//	moveGaussianStimulusContinuously("gs nf 3 " + std::to_string(large_obj_pos), small_obj_pos_a, -5.0f);
	//	const double f3_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", small_obj_pos_b, 4.0, 4.0);
	//	moveGaussianStimulusContinuously("gs nf 3 " + std::to_string(large_obj_pos), small_obj_pos_b, 10.0f);
	//	const double f3_2 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", small_obj_pos_a, 4.0, 4.0);
	//	const double f3 = 0.5f * f3_1 + 0.5f * f3_2;
	//	parameters.partialFitness.emplace_back(f3_1); // complementary action selection
	//	parameters.partialFitness.emplace_back(f3_2); // complementary action selection
//
	//	// large object appears
	//	addGaussianStimulus("nf 2",
	//		{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, large_obj_pos,
	//			GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
	//		{ DimensionConstants::xSize, DimensionConstants::dx });
	//	runSimulation(iterations);
	//	const double f4_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", large_obj_pos, 4.0, 4.0); // 2
	//	const double f4_2 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", small_obj_pos_a, 8.0, 4.0); // a
	//	moveGaussianStimulusContinuously("gs nf 3 " + std::to_string(large_obj_pos), large_obj_pos, -5.0f);
	//	const double f4_3 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", large_obj_pos, 10.0, 6.0); // c
	//	moveGaussianStimulusContinuously("gs nf 3 " + std::to_string(large_obj_pos), small_obj_pos_a, -5.0f);
	//	const double f4_4 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", small_obj_pos_b, 4.0, 4.0); // b
	//	moveGaussianStimulusContinuously("gs nf 3 " + std::to_string(large_obj_pos), large_obj_pos, 5.0f);
	//	const double f4_5 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", large_obj_pos, 10.0, 6.0); // c
	//	moveGaussianStimulusContinuously("gs nf 3 " + std::to_string(large_obj_pos), small_obj_pos_b, 5.0f);
	//	const double f4_6 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", small_obj_pos_a, 4.0, 4.0); // a
	//	const double f4 = 0.1f*f4_1 + 0.15f*f4_2 + 0.2f*f4_3 + 0.2f*f4_4 + 0.2f*f4_5 + 0.15f*f4_6;
	//	parameters.partialFitness.emplace_back(f4_1); // large object detection
	//	parameters.partialFitness.emplace_back(f4_2); // complementary action selection
	//	parameters.partialFitness.emplace_back(f4_3); // assistive action selection
	//	parameters.partialFitness.emplace_back(f4_4); // complementary action selection
	//	parameters.partialFitness.emplace_back(f4_5); // assistive action selection
	//	parameters.partialFitness.emplace_back(f4_6); // complementary action selection
//
	//	removeGaussianStimuli();
	//	runSimulation(iterations/2);
	//	const double f5_1 = closenessToRestingLevel("nf 1");
	//	const double f5_2 = closenessToRestingLevel("nf 2");
	//	const double f5_3 = closenessToRestingLevel("nf 3");
	//	const double f5_4 = closenessToRestingLevel("nf 4");
	//	const double f5 = 0.25f*f5_1 + 0.25f*f5_2 + 0.25f*f5_3 + 0.25f*f5_4;
	//	parameters.partialFitness.emplace_back(f5_1); // small object detection
	//	parameters.partialFitness.emplace_back(f5_2); // large object detection
	//	parameters.partialFitness.emplace_back(f5_3); // hand position detection
	//	parameters.partialFitness.emplace_back(f5_4); // action inhibition
//
	//	parameters.fitness = 0.10f*f1 + 0.10f*f2 + 0.10f*f3 + 0.65f*f4 + 0.05f*f5;
	//}

	void HRIPackagingTask::createPhenotypeEnvironment()
	{
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 10.0f, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			{ DimensionConstants::xSize, DimensionConstants::dx });

		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0f, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			{ DimensionConstants::xSize, DimensionConstants::dx });

		addGaussianStimulus("nf 2",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 30.0f, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			{ DimensionConstants::xSize, DimensionConstants::dx });

		addGaussianStimulus("nf 2",
			{ GaussStimulusConstants::width, 0.0f, 0.0f, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			{ DimensionConstants::xSize, DimensionConstants::dx });

		addGaussianStimulus("nf 3",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 30.0f, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			{ DimensionConstants::xSize, DimensionConstants::dx });
	}
}