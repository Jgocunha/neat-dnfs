#include "solutions/single_bump.h"

namespace neat_dnfs
{
	SingleBumpSolution::SingleBumpSolution(const SolutionTopology& topology)
		: Solution(topology)
	{}

	SolutionPtr SingleBumpSolution::clone() const
	{
		SingleBumpSolution solution(initialTopology);
		auto clonedSolution = std::make_shared<SingleBumpSolution>(solution);

		return clonedSolution;
	}

	void SingleBumpSolution::testPhenotype()
	{
		const dnf_composer::element::GaussStimulusParameters stimParams = {
			5.0,
			15.0,
			50.0,
			false,
			false
		};
		addGaussianStimulus("nf 1", stimParams);
		initSimulation();
		//runSimulationUntilFieldStable("nf 1");
		runSimulation(1000);
		updateFitness();
		removeGaussianStimuli();
		//runSimulationUntilFieldStable("nf 1");
		runSimulation(1000);

		stopSimulation();
	}

	void SingleBumpSolution::updateFitness()
	{
		using namespace dnf_composer::element;

		static constexpr double expectedBumpPosition = 50.0;
		static constexpr double expectedBumpWidth = 10.0;
		static constexpr double expectedBumpAmplitude = 10.0;

		const auto field =
			std::dynamic_pointer_cast<NeuralField>(phenotype.getElement("nf 2"));
		const auto fieldBumps = field->getBumps();

		if (fieldBumps.empty())
		{
			parameters.fitness = 0.0;
			return;
		}

		const auto bump = fieldBumps.front();
		const double centroidDifference = std::abs(bump.centroid - expectedBumpPosition);
		const double widthDifference = std::abs(bump.width - expectedBumpWidth);
		const double amplitudeDifference = std::abs(bump.amplitude - expectedBumpAmplitude);

		const double positionFitness = 0.8 / (1.0 + centroidDifference);
		const double widthFitness = 0.1 / (1.0 + widthDifference);
		const double amplitudeFitness = 0.1 / (1.0 + amplitudeDifference);

		parameters.fitness = positionFitness + widthFitness + amplitudeFitness;

		{
		//	// For debugging purposes.
		//if (parameters.fitness > 0.95)
		//{
		//	log(tools::logger::LogLevel::INFO, "Fitness is: " + std::to_string(parameters.fitness) + " centroidDifference: " + std::to_string(centroidDifference) + " widthDifference: " + std::to_string(widthDifference) + " amplitudeDifference: " + std::to_string(amplitudeDifference));
		//	log(tools::logger::LogLevel::INFO, "Position contribution: " + std::to_string(positionFitness) + " Width contribution: " + std::to_string(widthFitness) + " Amplitude contribution: " + std::to_string(amplitudeFitness));
		//	log(tools::logger::LogLevel::INFO, "Centroid: " + std::to_string(bump.centroid) + " width: " + std::to_string(bump.width) + " amplitude: " + std::to_string(bump.amplitude));
		//}

		// This evaluation is faulty (I think because of the precarious values of amplitude and width).
		// Fitness decreases but solution is present in the population.
		// bsAddress: 0000021F3D3F1A20 bsFitness: 0.5
		// opbsAddress: 0000021F3D3F1A21 opbsFitness: 0.66
		// npbsAddress: 0000021F3D3F1A21 npbsFitness: 0.43
		// pbsf > bsf > npbsf
		// pbsAddr = npbsAddr != bsAddr
		// Or:
		// Fitness decreased but previous best solution is in the population.
		// bsAddress: 0000021F3D3F1A20 bsFitness: 0.1667
		// opbsAddress: 0000021F3D3F1A20 opbsFitness: 0.1667
		// npbsAddress: 0000021F3D3F1A20 npbsFitness: 0.333
		// pbsf > npbsf = bsf
		// pbsAddr = npbsAddr = bsAddr
		// And:
		// Fitness decreased and previous best solution is not in the population.
		// Should just probably mean the previous best solution was so poorly evaluated
		// that it was not included as an elite in the population.
		// Meaning, the same solutions are being evaluated differently in evolutionary runs.
		// Part of the problem was the generation of non-stable solutions due to the lik parameters.
		// This still occurs tho.
		}
		
	}
}
