#include "solutions/two_robot_team.h"
#include <format>

namespace neat_dnfs
{
	TwoRobotTeam::TwoRobotTeam(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "Two robot team";
		// target fitness is 0.95
	}

	TwoRobotTeam::TwoRobotTeam(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype)
		: Solution(initialTopology, phenotype)
	{
		name = "Two robot team";
		// target fitness is 0.95
	}

	SolutionPtr TwoRobotTeam::clone() const
	{
		TwoRobotTeam solution(initialTopology, phenotype);
		auto clonedSolution = std::make_shared<TwoRobotTeam>(solution);

		return clonedSolution;
	}

	void TwoRobotTeam::testPhenotype()
	{
		using namespace dnf_composer::element;
		parameters.fitness = 0.0;
		parameters.partialFitness.clear();

		static constexpr int iterations = SimulationConstants::maxSimulationSteps;

		static constexpr double in_amp = 10.0;
		static constexpr double in_width = 12.0;
		static constexpr double out_amp = 5.0;
		static constexpr double out_width = 9.0;

		// nf 1 - working memory small object field		(INPUT)
		// nf 2 - working memory large object field		(INPUT)
		// nf 3 - other robot movement field 			(INPUT)	
		// nf 4 - action execution small object field	(OUTPUT)
		// nf 5 - action execution large object field	(OUTPUT)

		initSimulation();
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 2",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 3",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f1 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 5", { 50.0 }, out_amp, out_width);
		const double f2 = preShapedness("nf 4", { 20.0, 80.0 });
		parameters.partialFitness.emplace_back(f1);
		parameters.partialFitness.emplace_back(f2);

		moveGaussianStimulusContinously(std::format("gs nf 3 {}", 50.0), 20.0, -0.5);
		const double f3 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 4", { 80.0 }, out_amp, out_width);
		const double f4 = preShapedness("nf 5", { 20.0, 50.0 });
		parameters.partialFitness.emplace_back(f3);
		parameters.partialFitness.emplace_back(f4);

		moveGaussianStimulusContinously(std::format("gs nf 3 {}", 50.0), 80.0, +0.5); 
		const double f5 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 4", { 20.0 }, out_amp, out_width);
		const double f6 = preShapedness("nf 5", { 80.0, 50.0 });
		parameters.partialFitness.emplace_back(f5);
		parameters.partialFitness.emplace_back(f6);

		removeGaussianStimuliFromField("nf 3");
		initSimulation();
		runSimulation(iterations);
		const double f7 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 4", { 20.0, 80.0 }, out_amp, out_width);
		const double f8 = preShapedness("nf 5", { 50.0 });
		parameters.partialFitness.emplace_back(f7);
		parameters.partialFitness.emplace_back(f8);

		removeGaussianStimuli();

		static constexpr double wf1 = 1 / 8.f;
		static constexpr double wf2 = 1 / 8.f;
		static constexpr double wf3 = 1 / 8.f;
		static constexpr double wf4 = 1 / 8.f;
		static constexpr double wf5 = 1 / 8.f;
		static constexpr double wf6 = 1 / 8.f;
		static constexpr double wf7 = 1 / 8.f;
		static constexpr double wf8 = 1 / 8.f;

		parameters.fitness = wf1 * f1 + wf2 * f2 + wf3 * f3 + wf4 * f4 + wf5 * f5 + wf6 * f6 + wf7 * f7 + wf8 * f8;
	}

	void TwoRobotTeam::createPhenotypeEnvironment()
	{
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });

		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });

		addGaussianStimulus("nf 2",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });

		addGaussianStimulus("nf 3",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
	}
}