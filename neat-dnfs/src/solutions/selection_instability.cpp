#include "solutions/selection_instability.h"

namespace neat_dnfs
{
	SelectionInstability::SelectionInstability(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "Selection Instability";
		loadFitnessWeights("selection-instability", 4);
	}

	SelectionInstability::SelectionInstability(const SolutionTopology& initialTopology,
		const dnf_composer::Simulation& phenotype)
		:Solution(initialTopology, phenotype)
	{
		name = "Selection Instability";
		loadFitnessWeights("selection-instability", 4);
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
		const int iterations = SimulationConstants::maxSimulationSteps;

		static constexpr double left = 20.0;
		static constexpr double right = 80.0;
		static constexpr double in_amp = 8.0;
		static constexpr double in_width = 10.0;
		static constexpr double out_amp = 6.0;
		static constexpr double out_width = 5.0;

		initSimulation();
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, left, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, right, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);

		const double f1 = twoBumpsAtPositionWithAmplitudeAndWidth("nf 1",
			left, in_amp, in_width,
			right, in_amp, in_width);
		parameters.partialFitness.emplace_back(f1);
		const double f2 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 2",
			{ left, right }, out_amp, out_width);
		parameters.partialFitness.emplace_back(f2);

		removeGaussianStimuli();
		runSimulation(iterations);

		const double f3 = closenessToRestingLevel("nf 1");
		const double f4 = closenessToRestingLevel("nf 2");
		parameters.partialFitness.emplace_back(f3);
		parameters.partialFitness.emplace_back(f4);

		const auto& w = fitnessWeights;

		parameters.fitness = w[0]*f1 + w[1]*f2 + w[2]*f3 + w[3]*f4;
	}

	void SelectionInstability::createPhenotypeEnvironment()
	{
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0,
		GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
		dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });

		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0,
	GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
	}
}
