#include "solutions/and.h"

namespace neat_dnfs
{
	AND::AND(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "AND";
		loadFitnessWeights("and", 8);
	}

	AND::AND(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype)
		:Solution(initialTopology, phenotype)
	{
		name = "AND";
		loadFitnessWeights("and", 8);
	}

	SolutionPtr AND::clone() const
	{
		AND solution(initialTopology);
		auto clonedSolution = std::make_shared<AND>(solution);

		return clonedSolution;
	}

	SolutionPtr AND::copy() const
	{
		AND solution(initialTopology, phenotype);
		auto copy = std::make_shared<AND>(solution);

		return copy;
	}

	void AND::testPhenotype()
	{
		using namespace dnf_composer::element;
		parameters.fitness = 0.0;
		parameters.partialFitness.clear();

		const int iterations = SimulationConstants::maxSimulationSteps;

		static constexpr double position = 50.0;
		static constexpr double in_amp = 15.0;
		static constexpr double in_width = 10.0;
		static constexpr double out_amp = 10.0;
		static constexpr double out_width = 10.0;

		initSimulation();
		addStandardStimulus("nf 1", position);

		runSimulation(iterations);

		const double f1_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 1", position, in_amp, in_width);
		const double f1_2 = noBumps("nf 3");
		parameters.partialFitness.emplace_back(f1_1);
		parameters.partialFitness.emplace_back(f1_2);

		removeGaussianStimuli();
		addStandardStimulus("nf 2", position);

		runSimulation(iterations);

		const double f2_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", position, in_amp, in_width);
		const double f2_2 = noBumps("nf 3");
		parameters.partialFitness.emplace_back(f2_1);
		parameters.partialFitness.emplace_back(f2_2);

		addStandardStimulus("nf 1", position);

		runSimulation(iterations);

		const double f3 = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", position, out_amp, out_width);
		parameters.partialFitness.emplace_back(f3);

		removeGaussianStimuli();
		runSimulation(iterations);

		const double f4_1 = closenessToRestingLevel("nf 1");
		const double f4_2 = closenessToRestingLevel("nf 2");
		const double f4_3 = closenessToRestingLevel("nf 3");
		parameters.partialFitness.emplace_back(f4_1);
		parameters.partialFitness.emplace_back(f4_2);
		parameters.partialFitness.emplace_back(f4_3);

		const auto& w = fitnessWeights;

		parameters.fitness = w[0] * f1_1 + w[1] * f1_2 +
			w[2] * f2_1 + w[3] * f2_2 +
			w[4] * f3 +
			w[5] * f4_1 + w[6] * f4_2 + w[7] * f4_3;
	}

	void AND::createPhenotypeEnvironment()
	{
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ 5.0, 15.0, 50.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });

		addGaussianStimulus("nf 2",
			dnf_composer::element::GaussStimulusParameters{ 5.0, 0.0, 50.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
	}
}
