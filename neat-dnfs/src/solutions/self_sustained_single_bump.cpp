#include "solutions/self_sustained_single_bump.h"

namespace neat_dnfs
{
	SelfSustainedSingleBumpSolution::SelfSustainedSingleBumpSolution(const SolutionTopology& topology)
		: Solution(topology)
	{}

	SolutionPtr SelfSustainedSingleBumpSolution::clone() const
	{
		SelfSustainedSingleBumpSolution solution(initialTopology);
		auto clonedSolution = std::make_shared<SelfSustainedSingleBumpSolution>(solution);

		return clonedSolution;
	}

	void SelfSustainedSingleBumpSolution::testPhenotype()
	{
		parameters.fitness = 0.0;
		updateFitness();
	}

	void SelfSustainedSingleBumpSolution::updateFitness()
	{
		const dnf_composer::element::GaussStimulusParameters stimParams = {
			5.0,
			15.0,
			50.0,
			false,
			false
		};
		//parameters.fitness = 0.0;


		for (const auto& element : phenotype.getElements())
			if (element->getUniqueName() == "nf 3")
				parameters.fitness += 0.1;

		addGaussianStimulus("nf 1", stimParams);
		initSimulation();
		runSimulationUntilFieldStable("nf 1");
		parameters.fitness += 0.4 * selfStabilityFitness();
		removeGaussianStimuli();
		runSimulationUntilFieldStable("nf 1");
		parameters.fitness += 0.4 * selfSustainabilityFitness();
		runSimulationUntilFieldStable("nf 1");
		stopSimulation();
		if (std::isnan(parameters.fitness))
		{
			log(tools::logger::LogLevel::ERROR, "Fitness is NaN.");
			log(tools::logger::LogLevel::ERROR, "Stability fitness: " + std::to_string(selfStabilityFitness()) +
							" Sustainability fitness: " + std::to_string(selfSustainabilityFitness()));
		}
	}

	double SelfSustainedSingleBumpSolution::selfStabilityFitness() const
	{
		using namespace dnf_composer::element;

		static constexpr double expectedBumpPosition = 50.0;
		static constexpr double expectedBumpWidth = 10.0;
		static constexpr double expectedBumpAmplitude = 10.0;

		const auto field =
			std::dynamic_pointer_cast<NeuralField>(phenotype.getElement("nf 2"));
		const auto fieldBumps = field->getBumps();

		if (fieldBumps.empty())
			return 0.0;

		const auto bump = fieldBumps.front();
		const double centroidDifference = std::abs(bump.centroid - expectedBumpPosition);
		const double widthDifference = std::abs(bump.width - expectedBumpWidth);
		const double amplitudeDifference = std::abs(bump.amplitude - expectedBumpAmplitude);

		const double positionFitness = 1 / (1.0 + centroidDifference);
		//const double widthFitness = 0.1 / (1.0 + widthDifference);
		//const double amplitudeFitness = 0.1 / (1.0 + amplitudeDifference);

		return positionFitness/* + widthFitness + amplitudeFitness*/;
	}

	double SelfSustainedSingleBumpSolution::selfSustainabilityFitness() const
	{
		using namespace dnf_composer::element;

		const auto field =
			std::dynamic_pointer_cast<NeuralField>(phenotype.getElement("nf 2"));

		static constexpr double maxActivation = 20.0;
		static const double minActivation = field->getParameters().startingRestingLevel;

		 double currentMaxActivation = field->getHighestActivation();

		if (std::isnan(currentMaxActivation))
		{
			log(tools::logger::LogLevel::ERROR, "Current max activation is NaN.");
			currentMaxActivation = 0.0;
		}

		const double sustainabilityFitness = (currentMaxActivation - minActivation) / (maxActivation - minActivation);

		if (std::isnan(sustainabilityFitness))
		{
			log(tools::logger::LogLevel::ERROR, "Sustainability fitness is NaN.");
			log(tools::logger::LogLevel::ERROR, "Current max activation: " + std::to_string(currentMaxActivation) +
				" min activation: " + std::to_string(minActivation) + " max activation: " + std::to_string(maxActivation));
		}
		return sustainabilityFitness;
	}


}
