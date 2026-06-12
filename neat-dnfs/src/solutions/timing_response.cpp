#include "solutions/timing_response.h"


namespace neat_dnfs
{
	TimingResponse::TimingResponse(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "Timing response";
	}

	TimingResponse::TimingResponse(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype)
		: Solution(initialTopology, phenotype)
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
		parameters.partialFitness.clear();

		static constexpr double wf1 = 1 / 4.f;
		static constexpr double wf2 = 1 / 4.f;
		static constexpr double wf3 = 1 / 4.f;
		static constexpr double wf4 = 1 / 4.f;

		static constexpr double in_amp		= 5.0;
		static constexpr double in_width	= 8.0;


		initSimulation();
		addGaussianStimulus("nf 1", 
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(200);
		const double f1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 1", 50.0, in_amp, in_width);
		const double f2 = closenessToRestingLevel("nf 2");

		runSimulation(200);
		const double f3 = preShapedness("nf 2");

		runSimulation(200);
		const double f4 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", 50.0, in_amp, in_width);
		removeGaussianStimuli();

		parameters.partialFitness.push_back(f1);
		parameters.partialFitness.push_back(f2);
		parameters.partialFitness.push_back(f3);
		parameters.partialFitness.push_back(f4);

		parameters.fitness = wf1 * f1 + wf2 * f2 + wf3 * f3 + wf4 * f4;
	}

	void TimingResponse::createPhenotypeEnvironment()
	{
		addGaussianStimulus("nf 1",
			{ 5.0, 0.0, 50.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
	}
}