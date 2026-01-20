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

	// void InhibitionOfReturn::testPhenotype()
	// {
	// 	using namespace dnf_composer::element;
	// 	parameters.fitness = 0.0f;
	// 	static constexpr int iterations = SimulationConstants::maxSimulationSteps;
	// 	parameters.partialFitness.clear();
	//
	// 	static constexpr double left = 20.0;
	// 	static constexpr double right = 80.0;
	//
	// 	// cue activates spatial location
	// 	initSimulation();
	// 	addGaussianStimulus("nf 1",
	// 		{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, left,
	// 			GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
	// 			{ DimensionConstants::xSize, DimensionConstants::dx });
	// 	runSimulation(200);
	// 	const double f1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 1", left, 15.0, 12.0);
	// 	const double f2 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", left, 3.0, 10.0);
	// 	parameters.partialFitness.push_back(f1);
	// 	parameters.partialFitness.push_back(f2);
	//
	// 	// cue is removed
	// 	removeGaussianStimuli();
	// 	runSimulation(100);
	// 	const double f3 = closenessToRestingLevel("nf 1");
	// 	const double f4_1 = noBumps("nf 2");
	// 	const double f4_2 = preShapednessAtPosition("nf 2", left);
	// 	const double f4 =  0.2f * f4_1 + 0.8f * f4_2;
	// 	parameters.partialFitness.push_back(f3);
	// 	parameters.partialFitness.push_back(f4);
	//
	// 	// the same cue is given within a short delay (facilitation)
	// 	addGaussianStimulus("nf 1",
	// 		{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, left,
	// 			GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
	// 			{ DimensionConstants::xSize, DimensionConstants::dx });
	// 	runSimulation(100);
	// 	const double f5 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", left, 3.0, 6.0); // 1, 4
	// 	parameters.partialFitness.push_back(f5);
	//
	// 	// the same cue is given within a long delay (inhibition)
	// 	removeGaussianStimuli();
	// 	runSimulation(500);
	// 	const double f6 = negativePreShapednessAtPosition("nf 2", left);
	// 	parameters.partialFitness.push_back(f6);
	// 	addGaussianStimulus("nf 1",
	// 		{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, left,
	// 			GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
	// 			{ DimensionConstants::xSize, DimensionConstants::dx });
	// 	runSimulation(200);
	// 	const double f7_1 = noBumps("nf 2");
	// 	const double f7_2 =	preShapednessAtPosition("nf 2", left);
	// 	const double f7 = 0.2f * f7_1 + 0.8f * f7_2;
	// 	parameters.partialFitness.push_back(f7);
	//
	// 	// another cue is given for the same amount of time
	// 	removeGaussianStimuli();
	// 	runSimulation(500);
	// 	parameters.partialFitness.push_back(f6);
	// 	addGaussianStimulus("nf 1",
	// 		{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, right,
	// 			GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
	// 			{ DimensionConstants::xSize, DimensionConstants::dx });
	// 	runSimulation(200);
	// 	const double f8 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 2", {right}, 5, 12);
	// 	parameters.partialFitness.push_back(f8);
	//
	// 	// 1/8.f = 0.125f
	// 	static constexpr double wf1 = 0.1;//1/8.f;
	// 	static constexpr double wf2 = 0.125;//1/8.f;
	// 	static constexpr double wf3 = 0.05;//1/8.f;
	// 	static constexpr double wf4 = 0.2;//1/8.f;
	// 	static constexpr double wf5 = 0.125;//1/8.f;
	// 	static constexpr double wf6 = 0.15;//1/8.f;
	// 	static constexpr double wf7 = 0.15;//1/8.f;
	// 	static constexpr double wf8 = 0.1;//1/8.f;
	//
	// 	parameters.fitness = wf1 * f1 + wf2 * f2 + wf3 * f3 + wf4 * f4 + wf5 * f5 + wf6 * f6 + wf7 * f7 + wf8 * f8;
	// }

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
		runSimulation(500);
		const double f1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 1", left, 15.0, 12.0);
		const double f2 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", left, 8.0, 12.0);
		parameters.partialFitness.push_back(f1);
		parameters.partialFitness.push_back(f2);

		// cue is removed
		removeGaussianStimuli();
		runSimulation(1000); //1000
		const double f3 = closenessToRestingLevel("nf 1");
		const double f4_1 = noBumps("nf 2");
		const double f4_2 = negativePreShapednessAtPosition("nf 2", left);
		const double f4 =  0.2f * f4_1 + 0.8f * f4_2;
		parameters.partialFitness.push_back(f3);
		parameters.partialFitness.push_back(f4);

		// the same cue is given
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, left,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
				{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(500);
		const double f5 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", left, 6.0, 10.0);
		parameters.partialFitness.push_back(f5);

		static constexpr double wf1 = 0.15;
		static constexpr double wf2 = 0.20;
		static constexpr double wf3 = 0.15;
		static constexpr double wf4 = 0.30;
		static constexpr double wf5 = 0.20;

		parameters.fitness = wf1 * f1 + wf2 * f2 + wf3 * f3 + wf4 * f4 + wf5 * f5;
	}

	void InhibitionOfReturn::createPhenotypeEnvironment()
	{
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
				{ DimensionConstants::xSize, DimensionConstants::dx });
	}
}