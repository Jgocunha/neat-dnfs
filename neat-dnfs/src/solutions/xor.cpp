#include "solutions/xor.h"

namespace neat_dnfs
{
	XOR::XOR(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "XOR";
	}

	XOR::XOR(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype)
		: Solution(initialTopology, phenotype)
	{
		name = "XOR";
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
		static constexpr int iterations = SimulationConstants::maxSimulationSteps;

		// 1 | 0 = 1
		initSimulation();
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0,
				true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f1 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 3",
			{ 50.0 }, 5.0f, 10.0f);
		parameters.partialFitness.push_back(f1);
		removeGaussianStimuli();
		runSimulation(iterations/2);

		// 0 | 1 = 1
		addGaussianStimulus("nf 2",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0,
				true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f2 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 3",
			{ 50.0 }, 5.0f, 10.0f);
		parameters.partialFitness.push_back(f2);
		removeGaussianStimuli();
		runSimulation(iterations/2);

		// 1 | 1 = 0 (A)
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0,
				true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations/2);
		addGaussianStimulus("nf 2",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0,
				true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations/2);
		const double f3_1 = preShapednessAtPosition("nf 3", 50);
		removeGaussianStimuli();
		runSimulation(iterations/2);

		// 1 | 1 = 0 (B)
		addGaussianStimulus("nf 2",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0,
				true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations/2);
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0,
				true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations/2);
		const double f3_2 = preShapednessAtPosition("nf 3", 50);
		const double f3 = 1 / 2.f * f3_1 + 1 / 2.f * f3_2;
		parameters.partialFitness.push_back(f3);

		// 0 | 0 = 0
		removeGaussianStimuli();
		runSimulation(iterations/2);
		const double f4_1 = closenessToRestingLevel("nf 1");
		const double f4_2 = closenessToRestingLevel("nf 2");
		const double f4_3 = closenessToRestingLevel("nf 3");
		const double f4 = 1 / 3.f * f4_1 + 1 / 3.f * f4_2 + 1 / 3.f * f4_3;
		parameters.partialFitness.push_back(f4);

		static constexpr double wf1 = 0.25f;
		static constexpr double wf2 = 0.25f;
		static constexpr double wf3 = 0.30f;
		static constexpr double wf4 = 0.20f;

		parameters.fitness = wf1 * f1 + wf2 * f2 + wf3 * f3 + wf4 * f4;
	}

	void XOR::createPhenotypeEnvironment()
	{
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0,
				true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });

		addGaussianStimulus("nf 2",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0,
				true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
	}
}