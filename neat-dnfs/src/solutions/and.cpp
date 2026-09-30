#include "solutions/and.h"

namespace neat_dnfs
{
	AND::AND(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "AND";
		loadFitnessWeights("and", 8);
	}

	AND::AND(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype)
		:Solution(initialTopology, phenotype)
	{
		name = "AND";
		loadFitnessWeights("and", 8);
	}

	SolutionPtr AND::clone() const
	{
		AND solution(initialTopology);
		auto clonedSolution = std::make_shared<AND>(solution);

		return clonedSolution;
	}

	SolutionPtr AND::copy() const
	{
		AND solution(initialTopology, phenotype);
		auto copy = std::make_shared<AND>(solution);

		return copy;
	}

	void AND::testPhenotype()
	{
		using namespace dnf_composer::element;
		parameters.fitness = 0.0;
		parameters.partialFitness.clear();

		const int iterations = SimulationConstants::maxSimulationSteps;

		static constexpr double position = 50.0;
		static constexpr double in_amp = 11.0;
		static constexpr double in_width = 12.0;
		static constexpr double out_amp = 7.0;
		static constexpr double out_width = 16.0;

		initSimulation();
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, 
				GaussStimulusConstants::amplitude, position,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });

		runSimulation(10*iterations);

		const double f1 = oneBumpAtPositionWithAmplitudeAndWidth("nf 1", position, in_amp, in_width, 
			BumpFitnessWeights{ 0.40, 0.40, 0.10, 0.10 });
		//const std::string& fieldName, double position, const double sigma, const double epsilon
		const double f2 = preShapednessAtPosition("nf 3", position, 10.0, 0.1f);
		//noBumps("nf 3", BumpFitnessDefaults::noBumpsDecayRate);
		parameters.partialFitness.emplace_back(f1);
		parameters.partialFitness.emplace_back(f2);

		removeGaussianStimuli();
		addGaussianStimulus("nf 2",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, 
				GaussStimulusConstants::amplitude, position,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, 
				DimensionConstants::dx });

		runSimulation(5*iterations);

		const double f3 = oneBumpAtPositionWithAmplitudeAndWidth("nf 2", position, in_amp, in_width, 
			BumpFitnessWeights{ 0.40, 0.40, 0.10, 0.10 });
		const double f4 = preShapednessAtPosition("nf 3", position, 10.0, 0.1f);
		parameters.partialFitness.emplace_back(f3);
		parameters.partialFitness.emplace_back(f4);

		addGaussianStimulus("nf 1",
		dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, 
			GaussStimulusConstants::amplitude, position,
			GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
					dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, 
						DimensionConstants::dx });

		runSimulation(5*iterations);

		const double f5 = oneBumpAtPositionWithAmplitudeAndWidth("nf 3", position, out_amp, out_width, 
			BumpFitnessWeights{ 0.40, 0.40, 0.10, 0.10 });
		parameters.partialFitness.emplace_back(f5);

		removeGaussianStimuli();
		runSimulation(5*iterations);

		const double f6_1 = closenessOfMeanActivationToRestingLevel("nf 1");
		const double f6_2 = closenessOfMeanActivationToRestingLevel("nf 2");
		const double f6_3 = closenessOfMeanActivationToRestingLevel("nf 3");
		parameters.partialFitness.emplace_back(f6_1);
		parameters.partialFitness.emplace_back(f6_2);
		parameters.partialFitness.emplace_back(f6_3);

		const auto& w = fitnessWeights;

		parameters.fitness = w[0] * f1 + w[1] * f2 +
			w[2] * f3 + w[3] * f4 +
			w[4] * f5 +
			w[5] * f6_1 + w[6] * f6_2 + w[7] * f6_3;
	}

	void AND::createPhenotypeEnvironment()
	{
		addGaussianStimulus("nf 1",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, 
				GaussStimulusConstants::amplitude, 50.0,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
				dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });

		addGaussianStimulus("nf 2",
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, 
				GaussStimulusConstants::amplitude, 50.0,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
				dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
	}
}
