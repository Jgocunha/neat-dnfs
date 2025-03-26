#include "solutions/two_robot_team.h"

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
		static constexpr double out_amp = 10.0;
		static constexpr double out_width = 12.0;

		// nf 1 - working memory small object field		(INPUT)
		// nf 2 - working memory large object field		(INPUT)
		// nf 3 - other robot movement field 			(INPUT)	
		// nf 4 - action execution small object field	(OUTPUT)
		// nf 5 - action execution large object field	(OUTPUT)


        initSimulation();
        addGaussianStimulus("nf 1",
            { GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 20.0, true, false },
            { DimensionConstants::xSize, DimensionConstants::dx });
        runSimulation(iterations);
        const double f1 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 4", { 20.0 }, out_amp, out_width);
        parameters.partialFitness.push_back(f1);

        addGaussianStimulus("nf 1",
            { GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 80.0, true, false },
            { DimensionConstants::xSize, DimensionConstants::dx });
        runSimulation(iterations);
        const double f2 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 4", { 20.0, 80.0 }, out_amp, out_width);
        parameters.partialFitness.push_back(f2);

        const double f3 = closenessToRestingLevel("nf 5");
        parameters.partialFitness.push_back(f3);

        removeGaussianStimuli();
        addGaussianStimulus("nf 2",
            { GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
            { DimensionConstants::xSize, DimensionConstants::dx });
        runSimulation(iterations);
        const double f4 = preShapedness("nf 5");
        parameters.partialFitness.push_back(f4);

        addGaussianStimulus("nf 3",
            { GaussStimulusConstants::width, GaussStimulusConstants::amplitude, 50.0, true, false },
            { DimensionConstants::xSize, DimensionConstants::dx });
        runSimulation(iterations);
        const double f5 = justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth("nf 5", { 50.0 }, out_amp, out_width);
        parameters.partialFitness.push_back(f5);

        const double f6 = closenessToRestingLevel("nf 4");
        parameters.partialFitness.push_back(f6);

        removeGaussianStimuliFromField("nf 2");
        runSimulation(iterations);

        const double f7 = preShapedness("nf 5");
        parameters.partialFitness.push_back(f7);

        removeGaussianStimuli();

        // Calculate adaptive weights based on inverse fitness
        // Add a small epsilon to prevent division by zero
        static constexpr double epsilon = 1e-10;

        static constexpr int numWeights = 7;

        // Calculate inverse fitness values
        std::vector<double> inverseValues(numWeights);
        double inverseSumTotal = 0.0;

        for (int i = 0; i < numWeights; ++i) {
            inverseValues[i] = 1.0 / (parameters.partialFitness[i] + epsilon);
            inverseSumTotal += inverseValues[i];
        }

        // Normalize weights to sum to 1.0
        std::vector<double> adaptiveWeights(numWeights);
        for (int i = 0; i < numWeights; ++i) {
            adaptiveWeights[i] = inverseValues[i] / inverseSumTotal;
        }

        // Store the weights for debugging/analysis
        for (const auto& weight : adaptiveWeights) {
            parameters.partialFitness.push_back(weight);
        }

        // Calculate final weighted fitness
        parameters.fitness =
            adaptiveWeights[0] * f1 +
            adaptiveWeights[1] * f2 +
            adaptiveWeights[2] * f3 +
            adaptiveWeights[3] * f4 +
            adaptiveWeights[4] * f5 +
            adaptiveWeights[5] * f6 +
            adaptiveWeights[6] * f7;
	}

	void TwoRobotTeam::createPhenotypeEnvironment()
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