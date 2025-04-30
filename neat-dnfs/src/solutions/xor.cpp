#include "solutions/xor.h"

namespace neat_dnfs
{
	XOR::XOR(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "Two robot team";
		// target fitness is 0.95
	}

	XOR::XOR(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype)
		: Solution(initialTopology, phenotype)
	{
		name = "XOR";
		// target fitness is 0.95
	}

	SolutionPtr XOR::clone() const
	{
		XOR solution(initialTopology, phenotype);
		auto clonedSolution = std::make_shared<XOR>(solution);

		return clonedSolution;
	}

	void XOR::testPhenotype()
	{
		using namespace dnf_composer::element;
		parameters.fitness = 0.0;
		parameters.partialFitness.clear();

		static constexpr int iterations = SimulationConstants::maxSimulationSteps;

		static constexpr double in_amp = 10.0;
		static constexpr double in_width = 12.0;
		static constexpr double out_amp = 5.0;
		static constexpr double out_width = 9.0;

		initSimulation();
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f1 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 3", { 50.0 }, out_amp, out_width);
		parameters.partialFitness.push_back(f1);
		removeGaussianStimuli();

		removeGaussianStimuli();
		initSimulation();
		addGaussianStimulus("nf 2",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f2 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 3", { 50.0 }, out_amp, out_width);
		parameters.partialFitness.push_back(f2);

		removeGaussianStimuli();
		initSimulation();
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 2",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f3 = preShapedness("nf 3");
		parameters.partialFitness.push_back(f3);

		removeGaussianStimuli();

		static constexpr double wf1 = 1 / 3.f;
		static constexpr double wf2 = 1 / 3.f;
		static constexpr double wf3 = 1 / 3.f;

		parameters.fitness = wf1 * f1 + wf2 * f2 + wf3 * f3;
	}

	void XOR::createPhenotypeEnvironment()
	{
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });

		addGaussianStimulus("nf 2",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
	}
}