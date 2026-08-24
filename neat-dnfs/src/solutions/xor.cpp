#include "solutions/xor.h"

namespace neat_dnfs
{
	XOR::XOR(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "XOR";
		loadFitnessWeights("xor", 4);
	}

	XOR::XOR(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype)
		: Solution(initialTopology, phenotype)
	{
		name = "XOR";
		loadFitnessWeights("xor", 4);
	}

	SolutionPtr XOR::clone() const
	{
		XOR solution(initialTopology, phenotype);
		auto clonedSolution = std::make_shared<XOR>(solution);

		return clonedSolution;
	}

	SolutionPtr XOR::copy() const
	{
		XOR solution(initialTopology, phenotype);
		auto copy = std::make_shared<XOR>(solution);

		return copy;
	}

	void XOR::testPhenotype()
	{
		using namespace dnf_composer::element;
		parameters.fitness = 0.0;
		parameters.partialFitness.clear();

		const int iterations = SimulationConstants::maxSimulationSteps;

		static constexpr double position = 50.0;
		static constexpr double out_amp = 5.0;
		static constexpr double out_width = 9.0;

		initSimulation();
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, position,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f1 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 3", { position }, out_amp, out_width);
		parameters.partialFitness.push_back(f1);
		removeGaussianStimuli();

		removeGaussianStimuli();
		initSimulation();
		addGaussianStimulus("nf 2",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, position,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f2 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 3", { position }, out_amp, out_width);
		parameters.partialFitness.push_back(f2);

		removeGaussianStimuli();
		initSimulation();
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, position,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 2",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, position,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f3 = noBumps("nf 3");
		parameters.partialFitness.push_back(f3);

		removeGaussianStimuli();
		runSimulation(iterations);
		const double f4 = closenessToRestingLevel("nf 3");
		parameters.partialFitness.push_back(f4);

		const auto& w = fitnessWeights;

		parameters.fitness = w[0] * f1 + w[1] * f2 + w[2] * f3 + w[3] * f4;
	}

	void XOR::createPhenotypeEnvironment()
	{
		static constexpr double position = 50.0;

		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, position, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });

		addGaussianStimulus("nf 2",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, position, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
	}
}
