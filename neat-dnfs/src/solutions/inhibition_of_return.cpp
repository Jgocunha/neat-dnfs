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
	}

	void InhibitionOfReturn::createPhenotypeEnvironment()
	{
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
				dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
	}
}