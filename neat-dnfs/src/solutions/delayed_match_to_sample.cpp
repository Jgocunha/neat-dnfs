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
		DelayedMatchToSample solution(initialTopology);
		auto clonedSolution = std::make_shared<DelayedMatchToSample>(solution);

		return clonedSolution;
	}

	void DelayedMatchToSample::testPhenotype()
	{
		using namespace dnf_composer::element;
		parameters.fitness = 0.0f;
		static constexpr int iterations = SimulationConstants::maxSimulationSteps;
		parameters.partialFitness.clear();
	}

	void DelayedMatchToSample::createPhenotypeEnvironment()
	{
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
				{ DimensionConstants::xSize, DimensionConstants::dx });
	}
}