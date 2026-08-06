#include "solutions/select_the_object.h"
#include <format>

namespace neat_dnfs
{
	SelectTheObject::SelectTheObject(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "Select the object";
		// target fitness is 0.95
	}

	SelectTheObject::SelectTheObject(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype)
		: Solution(initialTopology, phenotype)
	{
		name = "Select the object";
		// target fitness is 0.95
	}

	SolutionPtr SelectTheObject::clone() const
	{
		SelectTheObject solution(initialTopology, phenotype);
		auto clonedSolution = std::make_shared<SelectTheObject>(solution);

		return clonedSolution;
	}

	void SelectTheObject::testPhenotype()
	{
		using namespace dnf_composer::element;
		parameters.fitness = 0.0;
		parameters.partialFitness.clear();

		static constexpr int iterations = SimulationConstants::maxSimulationSteps;

		static constexpr double in_amp = 10.0;
		static constexpr double in_width = 12.0;
		static constexpr double out_amp = 5.0;
		static constexpr double out_width = 10.0;

		//initSimulation();
		//addGaussianStimulus("nf 1",
		//	dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
		//	dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		//addGaussianStimulus("nf 2",
		//	dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
		//	dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		//addGaussianStimulus("nf 3",
		//	dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
		//	dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		//runSimulation(iterations);
		//addGaussianStimulus("nf 1",
		//	dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
		//	dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		//runSimulation(iterations);
		//const double f1 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 4", { 50.0 }, out_amp, out_width);
		//parameters.partialFitness.emplace_back(f1);

		//std::string gs = "gs nf 3";
		//gs += " " + std::to_string(50.0);
		//moveGaussianStimulusContinuously(gs, 80.0, 0.2);
		//const double f2 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 4", { 20.0 }, out_amp, out_width);
		//parameters.partialFitness.emplace_back(f2);

		//moveGaussianStimulusContinuously(gs, 20.0, -0.2);
		//const double f3 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 4", { 80.0 }, out_amp, out_width);
		//parameters.partialFitness.emplace_back(f3);

		//removeGaussianStimuli();

		//// adaptive weights for the fitness function
		//double inverseSumTotal = 0.0;
		//std::vector<double> inverseValues(3);
		//std::vector<double> adaptiveWeights(3);

		//// Calculate inverse values
		//for (int i = 0; i < 3; ++i) {
		//	constexpr double epsilon = 1e-10;  // Prevent division by zero
		//	inverseValues[i] = 1.0 / (parameters.partialFitness[i] + epsilon);
		//	inverseSumTotal += inverseValues[i];
		//}

		//// Normalize weights to sum to 1.0
		//for (int i = 0; i < 3; ++i) {
		//	adaptiveWeights[i] = inverseValues[i] / inverseSumTotal;
		//}

		//// Calculate final weighted fitness
		//parameters.fitness =
		//	adaptiveWeights[0] * f1 +
		//	adaptiveWeights[1] * f2 +
		//	adaptiveWeights[2] * f3;

		//parameters.partialFitness.emplace_back(adaptiveWeights[0]);
		//parameters.partialFitness.emplace_back(adaptiveWeights[1]);
		//parameters.partialFitness.emplace_back(adaptiveWeights[2]);

		//parameters.fitness = 1 / 3.f * f1 + 1 / 3.f * f2 + 1 / 3.f * f3;

		// nf 1 - input - small objects field (sof)
		// nf 2 - input - large object field (lof)
		// nf 3 - input - hand position field (hpf)
		// nf 4 - output - target action field (taf)

		// stage 1 - intra-field interactions
		static constexpr double weightSofMultiBump = 1 / 4.f; // sof multi bump
		static constexpr double weightLofSingleBump = 1 / 4.f; // lof single bump
		static constexpr double weightHpfSingleBump = 1 / 4.f; // hpf single bump
		static constexpr double weightTafSelectiveBump = 1 / 4.f; // taf selective bump
		static constexpr double weightStage1 = 1 / 4.f;
		// stage 2 - inter-field interactions (two-field interactions)
		static constexpr double weightSofToTafBump = 1 / 3.f; // sof (20, 80), -> taf (20 || 80)
		static constexpr double weightLofPreShapesTaf = 1 / 3.f; // lof (50) pre-shapes taf
		static constexpr double weightHpfNoBumpInTaf = 1 / 3.f; // hpf (50) -> no bump in taf
		static constexpr double weightStage2 = 1 / 4.f;
		// stage 3 - inter-field interactions (three-field interactions)
		static constexpr double weightSofHpf80ToTaf20 = 1 / 2.f; // sof (20, 80) + hpf(80) -> taf (20)
		static constexpr double weightSofHpf20ToTaf80 = 1 / 2.f; // sof (20, 80) + hpf(20) -> taf (80)
		static constexpr double weightSofHpfToTaf = 0.35f; // sofHpf80ToTaf20Fitness and sofHpf20ToTaf80Fitness
		static constexpr double weightLofHpfToTaf = 0.45f; // lof (50) + hpf(50) -> taf (50)
		static constexpr double weightSofHpf20NoBump = 1 / 2.f; // sof(20) + hpf (20) -> taf no bump
		static constexpr double weightSofHpf80NoBump = 1 / 2.f; // sof(80) + hpf (80) -> taf no bump
		static constexpr double weightSofHpfNoBump = 0.20f; // sofHpf20NoBumpFitness and sofHpf80NoBumpFitness
		static constexpr double weightStage3 = 1 / 4.f;
		// stage 4 - inter-field interactions (four-field interactions)
		static constexpr double weightLofHpfSofToTaf50 = 1 / 2.f; // lof(50) + hpf(50) + sof(20, 80) -> taf(50)
		static constexpr double weightLofHpfSofToTaf80 = 1 / 2.f; // lof(50) + hpf(20) + sof(20, 80) -> taf(80)
		static constexpr double weightLofHpfSofToTaf20 = 1 / 2.f; // lof(50) + hpf(80) + sof(20, 80) -> taf(20)
		static constexpr double weightLofHpfSofToTaf = 1 / 2.f; // lofHpfSofToTaf80Fitness and lofHpfSofToTaf20Fitness
		static constexpr double weightStage4 = 1 / 4.f;


		// stage 1 - intra-field interactions
		initSimulation();
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double sofMultiBumpFitness = twoBumpsAtPositionWithAmplitudeAndWidth("nf 1",
			20.0, in_amp, in_width,
			80.0, in_amp, in_width);
		parameters.partialFitness.emplace_back(sofMultiBumpFitness);

		removeGaussianStimuli();
		initSimulation();
		addGaussianStimulus("nf 2",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double lofSingleBumpFitness = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", 50.0, in_amp, in_width);
		parameters.partialFitness.emplace_back(lofSingleBumpFitness);

		removeGaussianStimuli();
		initSimulation();
		addGaussianStimulus("nf 3",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double hpfSingleBumpFitness = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", 50.0, in_amp, in_width);
		parameters.partialFitness.emplace_back(hpfSingleBumpFitness);

		removeGaussianStimuli();
		addGaussianStimulus("nf 4",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		addGaussianStimulus("nf 4",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double tafSelectiveBumpFitness = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 4", { 20.0, 80.0 }, out_amp, out_width);
		parameters.partialFitness.emplace_back(tafSelectiveBumpFitness);

		parameters.fitness = weightStage1 * (weightSofMultiBump * sofMultiBumpFitness + weightLofSingleBump * lofSingleBumpFitness
			+ weightHpfSingleBump * hpfSingleBumpFitness + weightTafSelectiveBump * tafSelectiveBumpFitness);

		
		// stage 2 - inter-field interactions (two-field interactions)
		removeGaussianStimuli();
		initSimulation();
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double sofToTafBumpFitness = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 4", { 20.0, 80.0 }, out_amp, out_width);
		parameters.partialFitness.emplace_back(sofToTafBumpFitness);

		removeGaussianStimuli();
		initSimulation();
		addGaussianStimulus("nf 2",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double lofPreShapesTafFitness = preShapedness("nf 4");
		parameters.partialFitness.emplace_back(lofPreShapesTafFitness);

		removeGaussianStimuli();
		initSimulation();
		addGaussianStimulus("nf 3",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double hpfNoBumpInTafFitness = closenessToRestingLevel("nf 4");
		parameters.partialFitness.emplace_back(hpfNoBumpInTafFitness);

		parameters.fitness += weightStage2 * (weightSofToTafBump * sofToTafBumpFitness + weightLofPreShapesTaf * lofPreShapesTafFitness
			+ weightHpfNoBumpInTaf * hpfNoBumpInTafFitness);

		// stage 3 - inter-field interactions (three-field interactions)
		removeGaussianStimuli();
		initSimulation();
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		addGaussianStimulus("nf 3",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double sofHpf80ToTaf20Fitness = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", 20.0, out_amp, out_width);
		parameters.partialFitness.emplace_back(sofHpf80ToTaf20Fitness);

		/*removeGaussianStimuli();
		initSimulation();
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);*/
		//addGaussianStimulus("nf 3",
		//	dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
		//	dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		std::string gs = "gs nf 3";
		gs += std::format(" {}", 80.0);
		moveGaussianStimulusContinuously(gs, 20.0, -0.2);
		//runSimulation(iterations);
		const double sofHpf20ToTaf80Fitness = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", 80.0, out_amp, out_width);
		parameters.partialFitness.emplace_back(sofHpf20ToTaf80Fitness);

		const double sofHpfToTafFitness = sofHpf80ToTaf20Fitness * weightSofHpf80ToTaf20 + sofHpf20ToTaf80Fitness * weightSofHpf20ToTaf80;

		removeGaussianStimuli();
		initSimulation();
		addGaussianStimulus("nf 2",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		addGaussianStimulus("nf 3",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double lofHpfToTafFitness = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", 50.0, out_amp, out_width);
		parameters.partialFitness.emplace_back(lofHpfToTafFitness);

		removeGaussianStimuli();
		initSimulation();
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		addGaussianStimulus("nf 3",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double sofHpf20NoBumpFitness = noBumps("nf 4");
		parameters.partialFitness.emplace_back(sofHpf20NoBumpFitness);

		//removeGaussianStimuli();
		//initSimulation();
		//addGaussianStimulus("nf 1",
		//	dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
		//	dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		//runSimulation(iterations);
		std::string gs2 = "gs nf 1";
		gs2 += std::format(" {}", 20.0);
		moveGaussianStimulusContinuously(gs2, 80.0, 0.2);
		//addGaussianStimulus("nf 3",
		//	dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
		//	dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		//runSimulation(iterations);
		std::string gs3 = "gs nf 3";
		gs3 += std::format(" {}", 20.0);
		moveGaussianStimulusContinuously(gs3, 80.0, 0.2);
		const double sofHpf80NoBumpFitness = noBumps("nf 4");
		parameters.partialFitness.emplace_back(sofHpf80NoBumpFitness);

		const double sofHpfNoBumpFitness = sofHpf20NoBumpFitness * weightSofHpf20NoBump + sofHpf80NoBumpFitness * weightSofHpf80NoBump;

		parameters.fitness += weightStage3 * (weightSofHpfToTaf * sofHpfToTafFitness + weightLofHpfToTaf * lofHpfToTafFitness
			+ weightSofHpfNoBump * sofHpfNoBumpFitness);

		// stage 4 - inter-field interactions (four-field interactions)
		removeGaussianStimuli();
		initSimulation();
		addGaussianStimulus("nf 2",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		addGaussianStimulus("nf 3",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double lofHpfSofToTaf50Fitness = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", 50.0, out_amp, in_width);
		parameters.partialFitness.emplace_back(lofHpfSofToTaf50Fitness);

		//removeGaussianStimuli();
		//initSimulation();
		/*addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		addGaussianStimulus("nf 2",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);*/
		/*ddGaussianStimulus("nf 3",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);*/
		std::string gs4 = "gs nf 3";
		gs4 += std::format(" {}", 50.0);
		moveGaussianStimulusContinuously(gs4, 20.0, -0.2);
		const double lofHpfSofToTaf80Fitness = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", 80.0, out_amp, in_width);
		parameters.partialFitness.emplace_back(lofHpfSofToTaf80Fitness);

		/*removeGaussianStimuli();
		initSimulation();
		addGaussianStimulus("nf 2",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		addGaussianStimulus("nf 3",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);*/
		moveGaussianStimulusContinuously(gs4, 80.0, 0.2);
		const double lofHpfSofToTaf20Fitness = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", 20.0, out_amp, in_width);
		parameters.partialFitness.emplace_back(lofHpfSofToTaf20Fitness);

		const double lofHpfSofToTafFitness = lofHpfSofToTaf80Fitness * weightLofHpfSofToTaf80 + lofHpfSofToTaf20Fitness * weightLofHpfSofToTaf20;

		parameters.fitness += weightStage4 * (weightLofHpfSofToTaf50 * lofHpfSofToTaf50Fitness + weightLofHpfSofToTaf * lofHpfSofToTafFitness);

		static constexpr double totalProbability = weightStage1 + weightStage2 + weightStage3 + weightStage4;

		constexpr double epsilon = 1e-6;
		if (std::abs(totalProbability - 1.0) > epsilon)
			throw std::runtime_error("Weight distribution in fitness function must sum up to 1.");
	}

	void SelectTheObject::createPhenotypeEnvironment()
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