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

		// nf 1 - input - small objects field (sof)
		// nf 2 - input - large object field (lof)
		// nf 3 - input - hand position field (hpf)
		// nf 4 - output - target action field (taf)

		// stage 1 - intra-field interactions
		static constexpr double wf1 = 1 / 4.f; // sof multi bump
		static constexpr double wf2 = 1 / 4.f; // lof single bump
		static constexpr double wf3 = 1 / 4.f; // hpf single bump
		static constexpr double wf4 = 1 / 4.f; // taf selective bump
		static constexpr double wf_1_2_3_4 = 1 / 4.f;
		// stage 2 - inter-field interactions (two-field interactions)
		static constexpr double wf5 = 1 / 3.f; // sof (20, 80), -> taf (20 || 80)
		static constexpr double wf6 = 1 / 3.f; // lof (50) pre-shapes taf
		static constexpr double wf7 = 1 / 3.f; // hpf (50) -> no bump in taf
		static constexpr double wf_5_6_7 = 1 / 4.f;
		// stage 3 - inter-field interactions (three-field interactions)
		static constexpr double wf8_1 = 1 / 2.f; // sof (20, 80) + hpf(80) -> taf (20)
		static constexpr double wf8_2 = 1 / 2.f; // sof (20, 80) + hpf(20) -> taf (80)
		static constexpr double wf8 = 0.35f; // f8_1 and f8_2
		static constexpr double wf9 = 0.45f; // lof (50) + hpf(50) -> taf (50)
		static constexpr double wf10_1 = 1 / 2.f; // sof(20) + hpf (20) -> taf no bump
		static constexpr double wf10_2 = 1 / 2.f; // sof(80) + hpf (80) -> taf no bump
		static constexpr double wf10 = 0.20f; // f10_1 and f10_2
		static constexpr double wf_8_9_10 = 1 / 4.f;
		// stage 4 - inter-field interactions (four-field interactions)
		static constexpr double wf11 = 1 / 2.f; // lof(50) + hpf(50) + sof(20, 80) -> taf(50)
		static constexpr double wf12_1 = 1 / 2.f; // lof(50) + hpf(20) + sof(20, 80) -> taf(80)
		static constexpr double wf12_2 = 1 / 2.f; // lof(50) + hpf(80) + sof(20, 80) -> taf(20)
		static constexpr double wf12 = 1 / 2.f; // f12_1 and f12_2
		static constexpr double wf_11_12 = 1 / 4.f;


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
		const double f1 = twoBumpsAtPositionWithAmplitudeAndWidth("nf 1",
			20.0, in_amp, in_width,
			80.0, in_amp, in_width);
		parameters.partialFitness.emplace_back(f1);

		removeGaussianStimuli();
		initSimulation();
		addGaussianStimulus("nf 2",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f2 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", 50.0, in_amp, in_width);
		parameters.partialFitness.emplace_back(f2);

		removeGaussianStimuli();
		initSimulation();
		addGaussianStimulus("nf 3",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f3 = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", 50.0, in_amp, in_width);
		parameters.partialFitness.emplace_back(f3);

		removeGaussianStimuli();
		addGaussianStimulus("nf 4",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		addGaussianStimulus("nf 4",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f4 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 4", { 20.0, 80.0 }, out_amp, out_width);
		parameters.partialFitness.emplace_back(f4);

		parameters.fitness = wf_1_2_3_4 * (wf1 * f1 + wf2 * f2 + wf3 * f3 + wf4 * f4);

		
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
		const double f5 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 4", { 20.0, 80.0 }, out_amp, out_width);
		parameters.partialFitness.emplace_back(f5);

		removeGaussianStimuli();
		initSimulation();
		addGaussianStimulus("nf 2",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f6 = preShapedness("nf 4");
		parameters.partialFitness.emplace_back(f6);

		removeGaussianStimuli();
		initSimulation();
		addGaussianStimulus("nf 3",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f7 = closenessToRestingLevel("nf 4");
		parameters.partialFitness.emplace_back(f7);

		parameters.fitness += wf_5_6_7 * (wf5 * f5 + wf6 * f6 + wf7 * f7);

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
		const double f8_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", 20.0, out_amp, out_width);
		parameters.partialFitness.emplace_back(f8_1);

		std::string gs = "gs nf 3";
		gs += std::format(" {}", 80.0);
		moveGaussianStimulusContinously(gs, 20.0, -0.2);
		const double f8_2 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", 80.0, out_amp, out_width);
		parameters.partialFitness.emplace_back(f8_2);

		const double f8 = f8_1 * wf8_1 + f8_2 * wf8_2;

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
		const double f9 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", 50.0, out_amp, out_width);
		parameters.partialFitness.emplace_back(f9);

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
		const double f10_1 = noBumps("nf 4");
		parameters.partialFitness.emplace_back(f10_1);

		std::string gs2 = "gs nf 1";
		gs2 += std::format(" {}", 20.0);
		moveGaussianStimulusContinously(gs2, 80.0, 0.2);
		std::string gs3 = "gs nf 3";
		gs3 += std::format(" {}", 20.0); 
		moveGaussianStimulusContinously(gs3, 80.0, 0.2);
		const double f10_2 = noBumps("nf 4");
		parameters.partialFitness.emplace_back(f10_2);

		const double f10 = f10_1 * wf10_1 + f10_2 * wf10_2;

		parameters.fitness += wf_8_9_10 * (wf8 * f8 + wf9 * f9 + wf10 * f10);

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
		const double f11 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", 50.0, out_amp, in_width);
		parameters.partialFitness.emplace_back(f11);

		std::string gs4 = "gs nf 3";
		gs4 += std::format(" {}", 50.0); 
		moveGaussianStimulusContinously(gs4, 20.0, -0.2);
		const double f12_1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", 80.0, out_amp, in_width);
		parameters.partialFitness.emplace_back(f12_1);

		moveGaussianStimulusContinously(gs4, 80.0, 0.2);
		const double f12_2 = oneBumpAtPositionWithAmplitudeAndWidth("nf 4", 20.0, out_amp, in_width);
		parameters.partialFitness.emplace_back(f12_2);

		const double f12 = f12_1 * wf12_1 + f12_2 * wf12_2;

		parameters.fitness += wf_11_12 * (wf11 * f11 + wf12 * f12);

		static constexpr double totalProbability = wf_1_2_3_4 + wf_5_6_7 + wf_8_9_10 + wf_11_12;

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