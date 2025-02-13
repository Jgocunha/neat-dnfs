#include "solutions/timing_response.h"

namespace neat_dnfs
{
	TimingResponse::TimingResponse(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "Timing response";
	}

	SolutionPtr TimingResponse::clone() const
	{
		TimingResponse solution(initialTopology);
		auto clonedSolution = std::make_shared<TimingResponse>(solution);

		return clonedSolution;
	}

	void TimingResponse::testPhenotype()
	{
		using namespace dnf_composer::element;
		parameters.fitness = 0.0;

		static constexpr double wf1 = 0.15;
		static constexpr double wf2 = 0.15;
		static constexpr double wf3 = 0.35;
		static constexpr double wf4 = 0.35;


		initSimulation();
		addGaussianStimulus("nf 1", 
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(SimulationConstants::maxSimulationSteps);
		const double f1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 1", 50.0, 9.0, 10.0);
		const double f2 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", 50.0, 9.0, 10.0);
		parameters.fitness = wf1 * f1 + wf2 * f2;

		removeGaussianStimuli();
		initSimulation();
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		const double f3 = iterationsUntilBumpWithAmplitude("nf 1", 100, 9.0);
		parameters.fitness += wf3 * f3;

		removeGaussianStimuli();
		initSimulation();
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		const double f4 = iterationsUntilBumpWithAmplitude("nf 2", 1000, 9.0);
		parameters.fitness += wf4 * f4;

		removeGaussianStimuli();
	}

	void TimingResponse::createPhenotypeEnvironment()
	{
		addGaussianStimulus("nf 1",
			{ 5.0, 0.0, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
	}
}