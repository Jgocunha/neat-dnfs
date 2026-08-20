#include "solutions/and.h"

namespace neat_dnfs
{
	AND::AND(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "AND";
	}

	AND::AND(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype)
		:Solution(initialTopology, phenotype)
	{
		name = "AND";
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

		static constexpr int iterations = SimulationConstants::maxSimulationSteps;

		initSimulation();
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });

		runSimulation(iterations);

		const double f1_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 1", 50.0, 15, 10);
		const double f1_2 = noBumps("nf 3");
		parameters.partialFitness.emplace_back(f1_1);
		parameters.partialFitness.emplace_back(f1_2);

		removeGaussianStimuli();
		addGaussianStimulus("nf 2",
dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0,
	GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });

		runSimulation(iterations);

		const double f2_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", 50.0, 15, 10);
		const double f2_2 = noBumps("nf 3");
		parameters.partialFitness.emplace_back(f2_1);
		parameters.partialFitness.emplace_back(f2_2);

		addGaussianStimulus("nf 1",
dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0,
	GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });

		runSimulation(iterations);

		const double f3 = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", 50.0, 10, 10);
		parameters.partialFitness.emplace_back(f3);

		removeGaussianStimuli();
		runSimulation(iterations);

		const double f4_1 = closenessToRestingLevel("nf 1");
		const double f4_2 = closenessToRestingLevel("nf 2");
		const double f4_3 = closenessToRestingLevel("nf 3");
		parameters.partialFitness.emplace_back(f4_1);
		parameters.partialFitness.emplace_back(f4_2);
		parameters.partialFitness.emplace_back(f4_3);


		static constexpr double wf1_1 = 0.10;
		static constexpr double wf1_2 = 0.20; 

		static constexpr double wf2_1 = 0.10;
		static constexpr double wf2_2 = 0.20; 

		static constexpr double wf3 = 0.25;

		static constexpr double wf4_1 = 0.05;
		static constexpr double wf4_2 = 0.05;
		static constexpr double wf4_3 = 0.05;

		parameters.fitness = wf1_1 * f1_1 + wf1_2 * f1_2 +
			wf2_1 * f2_1 + wf2_2 * f2_2 +
			wf3 * f3 +
			wf4_1 * f4_1 + wf4_2 * f4_2 + wf4_3 * f4_3;
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
