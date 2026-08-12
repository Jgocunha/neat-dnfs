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
		const double f1 = preShapednessAtPosition("nf 3", 50.0);
		parameters.partialFitness.emplace_back(f1);

		removeGaussianStimuli();
		addGaussianStimulus("nf 2",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0,
	GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f2 = preShapednessAtPosition("nf 3", 50.0);
		parameters.partialFitness.emplace_back(f2);

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
		const double f4 = 1/3.f*f4_1 + 1/3.f*f4_2 +  1/3.f*f4_3;
		parameters.partialFitness.emplace_back(f4);

		static constexpr double wf1 = 1 / 4.f;
		static constexpr double wf2 = 1 / 4.f;
		static constexpr double wf3 = 1 / 4.f;
		static constexpr double wf4 = 1 / 4.f;

		parameters.fitness = wf1*f1 + wf2*f2 + wf3*f3 + wf4*f4;
	}

	void AND::createPhenotypeEnvironment()
	{
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude,
				50.0, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });

		addGaussianStimulus("nf 2",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude,
				50.0, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
	}
}
