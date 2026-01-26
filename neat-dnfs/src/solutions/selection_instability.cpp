#include "solutions/selection_instability.h"

namespace neat_dnfs
{
	SelectionInstability::SelectionInstability(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "Selection Instability";
	}

	SelectionInstability::SelectionInstability(const SolutionTopology& initialTopology,
		const dnf_composer::Simulation& phenotype)
		:Solution(initialTopology, phenotype)
	{
		name = "Selection Instability";
	}

	SolutionPtr SelectionInstability::clone() const
	{
		SelectionInstability solution(initialTopology);
		auto clonedSolution = std::make_shared<SelectionInstability>(solution);

		return clonedSolution;
	}

	SolutionPtr SelectionInstability::copy() const
	{
		SelectionInstability solution(initialTopology, phenotype);
		auto copy = std::make_shared<SelectionInstability>(solution);

		return copy;
	}

	void SelectionInstability::testPhenotype()
	{
		using namespace dnf_composer::element;
		parameters.fitness = 0.0;
		parameters.partialFitness.clear();
		static constexpr int iterations = SimulationConstants::maxSimulationSteps;

		static constexpr double in_amp = 8.0;
		static constexpr double in_width = 10.0;
		static constexpr double out_amp = 6.0;
		static constexpr double out_width = 5.0;

		initSimulation();
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);

		const double f1 = twoBumpsAtPositionWithAmplitudeAndWidth("nf 1",
			20.0, in_amp, in_width,
			80.0, in_amp, in_width);
		parameters.partialFitness.emplace_back(f1);
		const double f2 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 2",
			{ 20.0, 80.0 }, out_amp, out_width);
		parameters.partialFitness.emplace_back(f2);

		removeGaussianStimuli();
		runSimulation(iterations);

		const double f3 = closenessToRestingLevel("nf 1");
		const double f4 = closenessToRestingLevel("nf 2");
		parameters.partialFitness.emplace_back(f3);
		parameters.partialFitness.emplace_back(f4);

		static constexpr double wf1 = 1 / 4.f;
		static constexpr double wf2 = 1 / 4.f;
		static constexpr double wf3 = 1 / 4.f;
		static constexpr double wf4 = 1 / 4.f;

		parameters.fitness = wf1*f1 + wf2*f2 + wf3*f3 + wf4*f4;
	}

	void SelectionInstability::createPhenotypeEnvironment()
	{
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0,
		GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
		{ DimensionConstants::xSize, DimensionConstants::dx });

		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0,
	GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
{ DimensionConstants::xSize, DimensionConstants::dx });
	}
}
