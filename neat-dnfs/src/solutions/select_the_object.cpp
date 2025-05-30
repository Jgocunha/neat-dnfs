#include "solutions/select_the_object.h"

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
		static constexpr double out_width = 9.0;

		// obj1_s -> r_rgp_obj1
		initSimulation();
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f1 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 4", { 20.0 }, out_amp, out_width);
		parameters.partialFitness.push_back(f1);

		// obj1_s + hand(obj1_s) -> null
		addGaussianStimulus("nf 3",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f2 = closenessToRestingLevel("nf 4");
		parameters.partialFitness.push_back(f2);

		// obj1_s + obj2_s -> r_rgp_obj1 OR r_rgp_obj2
		removeGaussianStimuli();
		initSimulation();
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f3 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 4", { 20.0, 80.0 }, out_amp, out_width);
		parameters.partialFitness.push_back(f3);

		// obj1_s + obj2_s + hand(obj1_s) -> r_rgp_obj2
		addGaussianStimulus("nf 3",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f4 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 4", { 80.0 }, out_amp, out_width);
		parameters.partialFitness.push_back(f4);

		// obj1_s + obj2_s + hand(obj2_s) -> r_rgp_obj1
		moveGaussianStimulusContinously("gs nf 3 " + std::to_string(20.0), 80.0, +0.5);
		const double f5 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 4", { 20.0 }, out_amp, out_width);
		parameters.partialFitness.push_back(f5);

		// obj1_s + obj2_s + obj_l -> r_rgp_obj1 OR r_rgp_obj2
		removeGaussianStimuliFromField("nf 3");
		addGaussianStimulus("nf 2",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f6 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 4", { 20.0, 80.0 }, out_amp, out_width);
		parameters.partialFitness.push_back(f6);

		// obj1_s + obj2_s + obj_l + hand(obj_l) -> r_rgp_obj_l
		addGaussianStimulus("nf 3",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
		runSimulation(iterations);
		const double f7 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 4", { 50.0 }, out_amp, out_width);
		parameters.partialFitness.push_back(f7);

		// obj1_s + obj2_s + obj_l + hand(obj1_s) -> r_rgp_obj2
		moveGaussianStimulusContinously("gs nf 3 " + std::to_string(50.0), 20.0, -0.5);
		const double f8 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 4", { 80.0 }, out_amp, out_width);
		parameters.partialFitness.push_back(f8);

		// obj1_s + obj2_s + obj_l + hand(obj2_s) -> r_rgp_obj1
		moveGaussianStimulusContinously("gs nf 3 " + std::to_string(50.0), 80.0, +0.5);
		const double f9 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 4", { 20.0 }, out_amp, out_width);
		parameters.partialFitness.push_back(f9);

		removeGaussianStimuli();
		runSimulation(iterations);
		const double f10 = closenessToRestingLevel("nf 4");
		parameters.partialFitness.push_back(f10);

		static constexpr double wf1 = 1 / 10.f;
		static constexpr double wf2 = 1 / 10.f;
		static constexpr double wf3 = 1 / 10.f;
		static constexpr double wf4 = 1 / 10.f;
		static constexpr double wf5 = 1 / 10.f;
		static constexpr double wf6 = 1 / 10.f;
		static constexpr double wf7 = 1 / 10.f;
		static constexpr double wf8 = 1 / 10.f;
		static constexpr double wf9 = 1 / 10.f;
		static constexpr double wf10 = 1 / 10.f;

		parameters.fitness = wf1 * f1 + wf2 * f2 + wf3 * f3
		+ wf4 * f4 + wf5 * f5 + wf6 * f6
		+ wf7 * f7 + wf8 * f8 + wf9 * f9
		+ wf10 * f10;
	}

	void SelectTheObject::createPhenotypeEnvironment()
	{
		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });

		addGaussianStimulus("nf 1",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });

		addGaussianStimulus("nf 2",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });

		addGaussianStimulus("nf 3",
			{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
			{ DimensionConstants::xSize, DimensionConstants::dx });
	}
}