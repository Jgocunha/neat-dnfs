#include "solutions/hri_packaging_task.h"

namespace neat_dnfs
{
	HRIPackagingTask::HRIPackagingTask(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "HRI Packaging Task";
		loadFitnessWeights("hri-packaging", 7);
	}

	HRIPackagingTask::HRIPackagingTask(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype)
		: Solution(initialTopology, phenotype)
	{
		name = "HRI Packaging Task";
		loadFitnessWeights("hri-packaging", 7);
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
		parameters.fitness = 0.0;
		parameters.partialFitness.clear();
		const int iterations = SimulationConstants::maxSimulationSteps;


		static constexpr double small_obj_pos_a = 10.0;
		static constexpr double small_obj_pos_b = 50.0;
		static constexpr double large_obj_pos = 30.0;
		static constexpr double obj_input_field_bump_amp = 5.0;
		static constexpr double obj_input_field_bump_width = 4.0;
		static constexpr double hand_input_field_bump_amp = 5.0;
		static constexpr double hand_input_field_bump_width = 4.0;
		static constexpr double large_output_field_bump_amp = 10.0;
		static constexpr double large_output_field_bump_width = 8.0;
		static constexpr double small_output_field_bump_amp = 10.0;
		static constexpr double small_output_field_bump_width = 8.0;
		static constexpr double hand_output_field_inhibition_depth = 8.6;
		static constexpr double hand_output_field_inhibition_width = 2.0;
		const std::string handStimulus = std::format("gs nf 3 {}", large_obj_pos);

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
		   small_obj_pos_a, obj_input_field_bump_amp, obj_input_field_bump_width,
		   small_obj_pos_b, obj_input_field_bump_amp, obj_input_field_bump_width);
		const double f1_2 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth(
			"nf 4", { small_obj_pos_a, small_obj_pos_b }, small_output_field_bump_amp, small_output_field_bump_width);

		// hand position negatively pre-shapes the output field
		addGaussianStimulus("nf 3",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, large_obj_pos,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f2_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", large_obj_pos, hand_input_field_bump_amp, hand_input_field_bump_width);
		const double f2_2 = negativePreShapingDepthAtPosition("nf 4", large_obj_pos, hand_output_field_inhibition_depth, hand_output_field_inhibition_width);

		// hand position biases the output field towards the non-targeted small object
		moveGaussianStimulusContinuously(handStimulus, small_obj_pos_a, -5.0f);
		const double f3_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", small_obj_pos_b, small_output_field_bump_amp, small_output_field_bump_width);
		moveGaussianStimulusContinuously(handStimulus, small_obj_pos_b, 10.0f);
		const double f3_2 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", small_obj_pos_a, small_output_field_bump_amp, small_output_field_bump_width);

		// large object appears
		addGaussianStimulus("nf 2",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, large_obj_pos,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f4_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", large_obj_pos, obj_input_field_bump_amp, obj_input_field_bump_width);
		const double f4_2 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", small_obj_pos_a, small_output_field_bump_amp, small_output_field_bump_width);
		moveGaussianStimulusContinuously(handStimulus, large_obj_pos, -5.0f);
		const double f4_3 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", large_obj_pos, large_output_field_bump_amp, large_output_field_bump_width);
		moveGaussianStimulusContinuously(handStimulus, small_obj_pos_a, -5.0f);
		const double f4_4 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", small_obj_pos_b,  small_output_field_bump_amp, small_output_field_bump_width);
		moveGaussianStimulusContinuously(handStimulus, large_obj_pos, 5.0f);
		const double f4_5 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", large_obj_pos, large_output_field_bump_amp, large_output_field_bump_width);
		moveGaussianStimulusContinuously(handStimulus, small_obj_pos_b, 5.0f);
		const double f4_6 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", small_obj_pos_a,  small_output_field_bump_amp, small_output_field_bump_width);

		removeGaussianStimuli();
		runSimulation(iterations/2);
		const double f5_1 = closenessOfMeanActivationToRestingLevel("nf 1");
		const double f5_2 = closenessOfMeanActivationToRestingLevel("nf 2");
		const double f5_3 = closenessOfMeanActivationToRestingLevel("nf 3");
		const double f5_4 = closenessOfMeanActivationToRestingLevel("nf 4"); 

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

		const auto& w = fitnessWeights;

		parameters.fitness = w[0] * f1
		+ w[1] * f2
		+ w[2] * f3
		+ w[3] * f4
		+ w[4] * f5
		+ w[5] * f6
		+ w[6] * f7;
	}

	void HRIPackagingTask::createPhenotypeEnvironment()
	{
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 
				10.0f, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });

		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 
				50.0f, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });

		addGaussianStimulus("nf 2",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 
				30.0f, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });

		addGaussianStimulus("nf 2",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, 0.0f, 0.0f, 
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });

		addGaussianStimulus("nf 3",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 
				30.0f, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
	}
}