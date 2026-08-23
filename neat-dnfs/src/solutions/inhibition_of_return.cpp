#include "solutions/inhibition_of_return.h"

namespace neat_dnfs
{
	InhibitionOfReturn::InhibitionOfReturn(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "Inhibition of Return";
		loadFitnessWeights("ior", 5);
	}

	InhibitionOfReturn::InhibitionOfReturn(const SolutionTopology& initialTopology,
		const dnf_composer::Simulation& phenotype)
		: Solution(initialTopology, phenotype)
	{
		name = "Inhibition of Return";
		loadFitnessWeights("ior", 5);
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
		parameters.fitness = 0.0;
		const int iterations = SimulationConstants::maxSimulationSteps;
		parameters.partialFitness.clear();

		static constexpr double left = 20.0;

		// cue activates spatial location
		initSimulation();
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, left,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
				dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
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
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, left,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
				dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(500);
		const double f5 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", left, 6.0, 10.0);
		parameters.partialFitness.push_back(f5);

		const auto& w = fitnessWeights;

		parameters.fitness = w[0] * f1 + w[1] * f2 + w[2] * f3 + w[3] * f4 + w[4] * f5;
	}

	void InhibitionOfReturn::createPhenotypeEnvironment()
	{
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
				dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
	}
}
