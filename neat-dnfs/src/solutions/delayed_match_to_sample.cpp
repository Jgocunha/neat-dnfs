#include "solutions/delayed_match_to_sample.h"

namespace neat_dnfs
{
	DelayedMatchToSample::DelayedMatchToSample(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "Delayed Match to Sample";
		loadFitnessWeights("dmts", 6);
	}

	DelayedMatchToSample::DelayedMatchToSample(const SolutionTopology& initialTopology,
		const dnf_composer::Simulation& phenotype)
		: Solution(initialTopology, phenotype)
	{
		name = "Delayed Match to Sample";
		loadFitnessWeights("dmts", 6);
	}

	SolutionPtr DelayedMatchToSample::clone() const
	{
		DelayedMatchToSample solution(initialTopology); //, phenotype
		auto clonedSolution = std::make_shared<DelayedMatchToSample>(solution);

		return clonedSolution;
	}

	SolutionPtr DelayedMatchToSample::copy() const
	{
		DelayedMatchToSample solution(initialTopology, phenotype);
		auto copy = std::make_shared<DelayedMatchToSample>(solution);

		return copy;
	}

	void DelayedMatchToSample::testPhenotype()
	{
		using namespace dnf_composer::element;
		parameters.fitness = 0.0;
		const int iterations = SimulationConstants::maxSimulationSteps;
		parameters.partialFitness.clear();

		// sample representation
		initSimulation();
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations); // enough to encode the memory of the sample
		const double f1 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 1", {50.0}, 15.0, 12.0);
		const double f2 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 2", {50.0}, 9.0, 12.0);
		parameters.partialFitness.push_back(f1);
		parameters.partialFitness.push_back(f2);

		// delay
		removeGaussianStimuli();
		runSimulation(iterations);
		const double f3 = closenessToRestingLevel("nf 1");
		parameters.partialFitness.push_back(f3);
		// make sure some kind of self-sustained activation exists within
		double f4_1 = 0.5 * noBumps("nf 2");
		double f4_2 = 0.5 * preShapednessAtPosition("nf 2", 50.0); //u_tar: -4.2
		runSimulation(iterations*4);
		f4_1 += 0.5 * noBumps("nf 2");
		f4_2 += 0.5 * preShapednessAtPosition("nf 2", 50.0); //u_tar: -4.2
		const double f4 = 0.2f * f4_1 + 0.8f * f4_2;
		parameters.partialFitness.push_back(f4);

		// test with two samples
		addGaussianStimulus("nf 1",
					dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0,
						GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
					dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 1",
					dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 100.0,
						GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
					dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f5 = twoBumpsAtPositionWithAmplitudeAndWidth("nf 1", 50, 15, 12,
																					100, 15, 12);
		const double f6 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 2", {50}, 9, 12);
		parameters.partialFitness.emplace_back(f5);
		parameters.partialFitness.emplace_back(f6);

		const auto& w = fitnessWeights;

		parameters.fitness = w[0] * f1 + w[1] * f2 + w[2] * f3 + w[3] * f4 + w[4] * f5 + w[5] * f6;
	}

	void DelayedMatchToSample::createPhenotypeEnvironment()
	{
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
				dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, 0.0, 100,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
				dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
	}
}
