#include "solutions/delayed_match_to_sample.h"

namespace neat_dnfs
{
	DelayedMatchToSample::DelayedMatchToSample(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "Delayed Match to Sample";
	}

	DelayedMatchToSample::DelayedMatchToSample(const SolutionTopology& initialTopology,
		const dnf_composer::Simulation& phenotype)
		: Solution(initialTopology, phenotype)
	{
		name = "Delayed Match to Sample";
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
		parameters.fitness = 0.0f;
		static constexpr int iterations = SimulationConstants::maxSimulationSteps;
		parameters.partialFitness.clear();

		// sample representation
		initSimulation();
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(300); // enough to encode the memory of the sample
		const double f1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 1", 50.0, 15.0, 12.0);
		const double f2 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", 50.0, 9.0, 12.0);
		parameters.partialFitness.push_back(f1);
		parameters.partialFitness.push_back(f2);

		// delay
		removeGaussianStimuli();
		runSimulation(1000); // make sure some kind of self-sustained activation exists within
		const double f3 = closenessToRestingLevel("nf 1");
		const double f4 = preShapednessAtPosition("nf 2", 50.0); //u_tar: -4.2
		// if ((f4 > 0.1) && (f4 < 0.7))
		// 	f4 += 0.15;
		//const auto hidden = phenotype.getElement("nf 3");
		//double f4_1 = 0.0;
		//if (hidden == nullptr)
		//	f4_1 = 0.0;
		//else {
		//	f4_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", 50.0, 13.0, 20.0);
		//		// if (f4_1 < 0.2)
		//		// 	f4_1 = 0.5;
		//}
		parameters.partialFitness.push_back(f3);
		parameters.partialFitness.push_back(f4);
		//parameters.partialFitness.push_back(f4_1);

		// test (no-match)
		addGaussianStimulus("nf 1",
					{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 100.0,
						GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
					{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(200);  // no response
		//const double f6 = noBumps("nf 2");
		// pre-shape at posA and pre-shape at posB
		const double f6_1 = preShapednessAtPosition("nf 2", 50.0);
		const double f6_2 = noBumps("nf 2");//preShapednessAtPosition("nf 2", 100.0);
		const double f6 = (f6_1 + f6_2) / 2.0;
		parameters.partialFitness.push_back(f6);

		// test (match)
		removeGaussianStimuli();
		runSimulation(200);
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(200);  // should be a fast response
		const double f5 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", 50.0, 8.0, 12.0);
		parameters.partialFitness.push_back(f5);

		static constexpr double wf1 = 1 / 6.f;
		static constexpr double wf2 = 1 / 6.f;
		static constexpr double wf3 = 1 / 6.f;
		static constexpr double wf4 = 1 / 6.f;
		static constexpr double wf5 = 1 / 6.f;
		static constexpr double wf6 = 1 / 6.f;
		//static constexpr double wf4_1 = 1 / 7.f;

		parameters.fitness = wf1 * f1 + wf2 * f2 + wf3 * f3 + wf4 * f4 + wf5 * f5 + wf6 * f6; //+ wf4_1 * f4_1;
	}

	void DelayedMatchToSample::createPhenotypeEnvironment()
	{
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
				{ DimensionConstants::xSize, DimensionConstants::dx });
	}
}