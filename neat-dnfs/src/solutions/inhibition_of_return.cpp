#include "solutions/inhibition_of_return.h"

namespace neat_dnfs
{
	InhibitionOfReturn::InhibitionOfReturn(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "Inhibition of Return";
	}

	InhibitionOfReturn::InhibitionOfReturn(const SolutionTopology& initialTopology,
		const dnf_composer::Simulation& phenotype)
		: Solution(initialTopology, phenotype)
	{
		name = "Inhibition of Return";
	}

	SolutionPtr InhibitionOfReturn::clone() const
	{
		InhibitionOfReturn solution(initialTopology);
		auto clonedSolution = std::make_shared<InhibitionOfReturn>(solution);

		return clonedSolution;
	}

	SolutionPtr InhibitionOfReturn::copy() const
	{
		InhibitionOfReturn solution(initialTopology, phenotype);
		auto copy = std::make_shared<InhibitionOfReturn>(solution);

		return copy;
	}

	void InhibitionOfReturn::testPhenotype()
	{
		using namespace dnf_composer::element;
		parameters.fitness = 0.0f;
		static constexpr int iterations = SimulationConstants::maxSimulationSteps;
		parameters.partialFitness.clear();

		static constexpr double left = 20.0; 
		static constexpr double right = 80.0; 

		// cue activates spatial location
		initSimulation();
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, left,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
				{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(200);
		const double f1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 1", left, 15.0, 12.0);
		const double f2 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", left, 3.0, 10.0);
		parameters.partialFitness.push_back(f1);
		parameters.partialFitness.push_back(f2);

		// cue is removed
		removeGaussianStimuli();
		runSimulation(100);
		const double f3 = closenessToRestingLevel("nf 1");
		const double f4_1 = noBumps("nf 2");
		const double f4_2 = preShapednessAtPosition("nf 2", left);
		const double f4 =  0.2f * f4_1 + 0.8f * f4_2;
		parameters.partialFitness.push_back(f3);
		parameters.partialFitness.push_back(f4);

		// the same cue is given within a short delay (facilitation)
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, left,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
				{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(100);
		const double f5 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", left, 1.0, 4.0);
		parameters.partialFitness.push_back(f5);

		// the same cue is given within a long delay (inhibition)
		removeGaussianStimuli();
		runSimulation(500);
		const double f6 = negativePreShapednessAtPosition("nf 2", left);
		parameters.partialFitness.push_back(f6);
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, left,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
				{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(200);
		const double f7_1 = noBumps("nf 2");
		const double f7_2 =	preShapednessAtPosition("nf 2", left);
		const double f7 = 0.2f * f7_1 + 0.8f * f7_2;
		parameters.partialFitness.push_back(f7);

		// another cue is given for the same amount of time
		removeGaussianStimuli();
		runSimulation(500);
		parameters.partialFitness.push_back(f6);
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, right,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
				{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(200);
		const double f8 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 2", {right}, 5, 12);
		parameters.partialFitness.push_back(f8);

		// 1/8.f = 0.125f
		static constexpr double wf1 = 0.1;//1/8.f;
		static constexpr double wf2 = 0.125;//1/8.f;
		static constexpr double wf3 = 0.05;//1/8.f;
		static constexpr double wf4 = 0.2;//1/8.f;
		static constexpr double wf5 = 0.125;//1/8.f;
		static constexpr double wf6 = 0.15;//1/8.f;
		static constexpr double wf7 = 0.15;//1/8.f;
		static constexpr double wf8 = 0.1;//1/8.f;

		parameters.fitness = wf1 * f1 + wf2 * f2 + wf3 * f3 + wf4 * f4 + wf5 * f5 + wf6 * f6 + wf7 * f7 + wf8 * f8;
	}

	//void InhibitionOfReturn::testPhenotype()
	//{
	//	using namespace dnf_composer::element;
	//	parameters.fitness = 0.0f;
	//	static constexpr int iterations = SimulationConstants::maxSimulationSteps;
	//	parameters.partialFitness.clear();
//
	//	static constexpr double posA = 20.0;
	//	static constexpr double posB = 80.0;
//
	//	// cue activates spatial location
	//	initSimulation();
	//	addGaussianStimulus("nf 1",
	//		{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, posA,
	//			GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
	//			{ DimensionConstants::xSize, DimensionConstants::dx });
	//	runSimulation(500);
	//	const double f1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 1", posA, 15.0, 12.0);
	//	const double f2 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", posA, 6.0, 12.0);
	//	//const double f2_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", posA, 21.0, 14.0); // inhibitory
	//	//const double f2_2 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", posA, 9.0, 14.0); // excitatory
	//	parameters.partialFitness.push_back(f1);
	//	parameters.partialFitness.push_back(f2);
	//	//parameters.partialFitness.push_back(f2_1);
	//	//parameters.partialFitness.push_back(f2_2);
//
	//	//double guided_fitness = 1/4.f * (0.25f * f1 + 0.25f * f2 + 0.25f * f2_1 + 0.25f * f2_2);
	//	double non_guided_fitness = 1/4.f * (0.5f * f1 + 0.5f * f2);
//
	//	// attention disengages and residual inhibition dominates
	//	removeGaussianStimuli();
	//	runSimulation(500);
	//	const double f3 = closenessToRestingLevel("nf 1");
	//	const double f4 = negativePreShapednessAtPosition("nf 2", posA);
	//	//const double f4_1 = closenessToRestingLevel("nf 4"); // excitatory
	//	//const double f4_2 = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", posA, 5.0, 6.0); // inhibitory
	//	parameters.partialFitness.push_back(f3);
	//	parameters.partialFitness.push_back(f4);
	//	//parameters.partialFitness.push_back(f4_1);
	//	//parameters.partialFitness.push_back(f4_2);
//
	//	//guided_fitness += 1/4.f * (0.25f * f3 + 0.25f * f4 + 0.25f * f4_1 + 0.25f * f4_2);
	//	non_guided_fitness += 1/4.f * (0.5f * f3 + 0.5f * f4);
//
	//	// slower response to the same cue
	//	addGaussianStimulus("nf 1",
	//		{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, posA,
	//			GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
	//			{ DimensionConstants::xSize, DimensionConstants::dx });
	//	runSimulation(50);
	//	const double f5 = oneBumpAtPositionWithAmplitudeAndWidth("nf 1", posA, 15.0, 12.0);
	//	const double f6 = preShapednessAtPosition("nf 2", posA);
	//	//const double f6_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", posA, 10.0, 8.0); // inhibitory
	//	//const double f6_2 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", posA, 8.0, 12.0); // excitatory
	//	parameters.partialFitness.push_back(f5);
	//	parameters.partialFitness.push_back(f6);
	//	//parameters.partialFitness.push_back(f6_1);
	//	//parameters.partialFitness.push_back(f6_2);
//
	//	//guided_fitness += 1/4.f * (0.25f * f5 + 0.25f * f6 + 0.25f * f6_1 + 0.25f * f6_2);
	//	non_guided_fitness += 1/4.f * (0.5f * f5 + 0.5f * f6);
//
	//	// faster response to a different cue
	//	removeGaussianStimuli();
	//	addGaussianStimulus("nf 1",
	//		{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, posB,
	//			GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
	//			{ DimensionConstants::xSize, DimensionConstants::dx });
	//	runSimulation(50);
	//	const double f7 = oneBumpAtPositionWithAmplitudeAndWidth("nf 1", posB, 15.0, 12.0);
	//	const double f8 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", posB, 7.0, 8.0);
	//	//const double f8_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", posA, 10.0, 8.0); // inhibitory
	//	//const double f8_2 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", posB, 8.0, 12.0); // excitatory
	//	parameters.partialFitness.push_back(f7);
	//	parameters.partialFitness.push_back(f8);
	//	//parameters.partialFitness.push_back(f8_1);
	//	//parameters.partialFitness.push_back(f8_2);
//
	//	//guided_fitness += 1/4.f * (0.25f * f7 + 0.25f * f8 + 0.25f * f8_1 + 0.25f * f8_2);
	//	non_guided_fitness += 1/4.f * (0.5f * f7 + 0.5f * f8);
//
	//	parameters.fitness = non_guided_fitness;
	//}

	void InhibitionOfReturn::createPhenotypeEnvironment()
	{
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
				{ DimensionConstants::xSize, DimensionConstants::dx });
	}
}