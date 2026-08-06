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

		moveGaussianStimulusContinuously(std::format("gs nf 3 {}", 50.0), 20.0, -0.5);
		const double f3 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 4", { 80.0 }, out_amp, out_width);
		const double f4 = preShapedness("nf 5", { 20.0, 50.0 });
		parameters.partialFitness.emplace_back(f3);
		parameters.partialFitness.emplace_back(f4);

		moveGaussianStimulusContinuously(std::format("gs nf 3 {}", 50.0), 80.0, +0.5);
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

  //      // stage 1
  //      initSimulation();
  //      addGaussianStimulus("nf 1",
  //          dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
  //          dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
  //      addGaussianStimulus("nf 1",
  //          dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
  //          dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
  //      runSimulation(iterations);
  //      const double f1 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 4", { 20.0, 80.0 }, out_amp, out_width);
  //      parameters.partialFitness.push_back(f1);

  //      const double f2 = closenessToRestingLevel("nf 5");
		//parameters.partialFitness.push_back(f2);

		//
		//// stage 2
  //      removeGaussianStimuli();
  //      initSimulation();
  //      addGaussianStimulus("nf 2",
  //          dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
  //          dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
  //      runSimulation(iterations);
  //      const double f3 = preShapedness("nf 5");
  //      parameters.partialFitness.push_back(f3);

  //      removeGaussianStimuli();
  //      //initSimulation();
  //      addGaussianStimulus("nf 3",
  //          dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
  //          dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
  //      runSimulation(iterations);
  //      const double f4 = preShapedness("nf 5");
  //      parameters.partialFitness.push_back(f4);

  //      addGaussianStimulus("nf 2",
  //          dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
  //          dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
  //      runSimulation(iterations);
  //      const double f5 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 5", { 50.0 }, out_amp, out_width);
  //      parameters.partialFitness.push_back(f5);

  //      const double f6 = negativePreShapednessAtPosition("nf 4", 50.0);
		////const double f6 = negativeBaseline("nf 4");
  //      parameters.partialFitness.push_back(f6);

		//removeGaussianStimuli();
		//runSimulation(iterations);
		//const double f14 = closenessToRestingLevel("nf 4");
		//const double f15 = closenessToRestingLevel("nf 5");
		//parameters.partialFitness.push_back(f14);
		//parameters.partialFitness.push_back(f15);

  //      // stage 3
  //      removeGaussianStimuli();
  //      initSimulation();
		//addGaussianStimulus("nf 3",
		//	dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
		//	dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		//runSimulation(iterations);
		//const double f7 = negativePreShapednessAtPosition("nf 4", 20.0);
		//parameters.partialFitness.push_back(f7);

  //      addGaussianStimulus("nf 1",
  //          dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
  //          dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
  //      addGaussianStimulus("nf 1",
  //          dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
  //          dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		//runSimulation(iterations);
		//const double f8_1 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 4", { 80.0 }, out_amp, out_width);

		//moveGaussianStimulusContinuously("gs nf 3 " + std::to_string(20.0), 80.0, 0.5);
		//const double f8_2 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 4", { 20.0 }, out_amp, out_width);
		//const double f8 = 0.5f * f8_1 + 0.5f * f8_2;
		//parameters.partialFitness.push_back(f8);

		//// stage 4
		//removeGaussianStimuli();
		//initSimulation();
		//addGaussianStimulus("nf 1",
		//	dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
		//	dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		//addGaussianStimulus("nf 2",
		//	dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
		//	dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		//addGaussianStimulus("nf 3",
		//	dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
		//	dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		//runSimulation(iterations);
		//const double f10 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 5", { 50.0 }, out_amp, out_width);
		//parameters.partialFitness.push_back(f10);
		////const double f11 = negativeBaseline("nf 4");
		////const double f11 = preShapedness("nf 4");
		//const double f11 = noBumps("nf 4");
		//parameters.partialFitness.push_back(f11);

		//moveGaussianStimulusContinuously("gs nf 3 " + std::to_string(50.0), 80.0, 0.5);
		//const double f12 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 4", { 20.0 }, out_amp, out_width);
		//parameters.partialFitness.push_back(f12);

		//const double f13 = noBumps("nf 5");
		//parameters.partialFitness.push_back(f13);
		//
		//removeGaussianStimuli();

		//static constexpr double wf1		= 1 / 2.f;
		//static constexpr double wf2		= 1 / 2.f;

		//static constexpr double wf3		= 1 / 6.f;
		//static constexpr double wf4		= 1 / 6.f;
		//static constexpr double wf5		= 1 / 6.f;
		//static constexpr double wf6		= 1 / 6.f;
		//static constexpr double wf14	= 1 / 6.f;
		//static constexpr double wf15	= 1 / 6.f;
		//									  
		//static constexpr double wf7		= 1 / 2.f;
		//static constexpr double wf8		= 1 / 2.f;
		//									  
		//static constexpr double wf10	= 1 / 4.f;
		//static constexpr double wf11	= 1 / 4.f;
		//static constexpr double wf12	= 1 / 4.f;
		//static constexpr double wf13	= 1 / 4.f;

		//static constexpr double wf1_2				= 1 / 4.f;
		//static constexpr double wf3_4_5_6_14_15		= 1 / 4.f;
		//static constexpr double wf7_8				= 1 / 4.f;
		//static constexpr double wf10_11_12_13		= 1 / 4.f;

		//parameters.fitness = wf1_2 * (wf1 * f1 + wf2 * f2);
		//parameters.fitness += wf3_4_5_6_14_15 * (wf3 * f3 + wf4 * f4 + wf5 * f5 + wf6 * f6 + wf14 * f14 + wf15 * f15);
		//parameters.fitness += wf7_8 * (wf7 * f7 + wf8 * f8);
		//parameters.fitness += wf10_11_12_13 * (wf10 * f10 + wf11 * f11 + wf12 * f12 + wf13 * f13);
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