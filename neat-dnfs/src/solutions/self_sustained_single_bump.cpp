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

		addGaussianStimulus("nf 1", stimParams);
		initSimulation();
		runSimulation(100);
		//runSimulationUntilFieldStable("nf 1");
		const double bumpFormationScore = selfStabilityFitness();
		removeGaussianStimuli();
		runSimulation(100);
		//runSimulationUntilFieldStable("nf 1");
		const double bumpPersistenceScore = selfSustainabilityFitness();
		//runSimulationUntilFieldStable("nf 1");
		runSimulation(100);
		stopSimulation();
		parameters.fitness = 0.5*bumpFormationScore + 0.5*bumpPersistenceScore;
	}

	double SelfSustainedSingleBumpSolution::selfStabilityFitness() const
	{
		using namespace dnf_composer::element;

		const auto field = std::dynamic_pointer_cast<NeuralField>(phenotype.getElement("nf 2"));
		const auto fieldBumps = field->getBumps();

		if (fieldBumps.empty())
			return 0.0;

		const auto bump = fieldBumps.front();

		constexpr double w = 1.0 / 3.0;
		const double score = w * 1 + 
			w * tools::utils::normalize(bump.amplitude,5,50) + 
			w * tools::utils::normalize(bump.width, 5, 50);
		return score;
	}

	double SelfSustainedSingleBumpSolution::selfSustainabilityFitness() const
	{
		using namespace dnf_composer::element;

		const auto field = std::dynamic_pointer_cast<NeuralField>(phenotype.getElement("nf 2"));
		const auto fieldBumps = field->getBumps();

		if (fieldBumps.empty())
			return 0.0;

		const auto bump = fieldBumps.front();

		constexpr double w = 1.0 / 3.0;
		const double score = w * 1 +
			w * tools::utils::normalize(bump.amplitude, 0.1, 50) +
			w * tools::utils::normalize(bump.width, 2, 50);
		return score;
	}



	//double SelfSustainedSingleBumpSolution::selfStabilityFitness() const
	//{
	//	using namespace dnf_composer::element;

	//	static constexpr double expectedBumpPosition = 50.0;
	//	static constexpr double expectedBumpWidth = 10.0;
	//	static constexpr double expectedBumpAmplitude = 50.0;

	//	const auto field =
	//		std::dynamic_pointer_cast<NeuralField>(phenotype.getElement("nf 2"));
	//	const auto fieldBumps = field->getBumps();

	//	if (fieldBumps.empty())
	//		return 0.0;

	//	const auto bump = fieldBumps.front();
	//	const double centroidDifference = std::abs(bump.centroid - expectedBumpPosition);
	//	const double widthDifference = std::abs(bump.width - expectedBumpWidth);
	//	const double amplitudeDifference = std::abs(bump.amplitude - expectedBumpAmplitude);

	//	const double positionFitness = 0.7 / (1.0 + centroidDifference);
	//	//const double widthFitness = 0.1 / (1.0 + widthDifference);
	//	const double amplitudeFitness = 0.3 / (1.0 + amplitudeDifference);

	//	return positionFitness/* + widthFitness*/ + amplitudeFitness;
	//}

	//double SelfSustainedSingleBumpSolution::selfSustainabilityFitness() const
	//{
	//	using namespace dnf_composer::element;

	//	static constexpr double expectedBumpPosition = 50.0;
	//	static constexpr double expectedBumpWidth = 10.0;
	//	static constexpr double expectedBumpAmplitude = 10.0;

	//	const auto field =
	//		std::dynamic_pointer_cast<NeuralField>(phenotype.getElement("nf 2"));
	//	const auto fieldBumps = field->getBumps();

	//	if (fieldBumps.empty())
	//		return 0.0;

	//	const auto bump = fieldBumps.front();
	//	const double centroidDifference = std::abs(bump.centroid - expectedBumpPosition);
	//	const double widthDifference = std::abs(bump.width - expectedBumpWidth);
	//	const double amplitudeDifference = std::abs(bump.amplitude - expectedBumpAmplitude);

	//	const double positionFitness = 1 / (1.0 + centroidDifference);
	//	//const double widthFitness = 0.1 / (1.0 + widthDifference);
	//	//const double amplitudeFitness = 0.1 / (1.0 + amplitudeDifference);

	//	return positionFitness/* + widthFitness + amplitudeFitness*/;
	//}

	//double SelfSustainedSingleBumpSolution::selfSustainabilityFitness() const
	//{
	//	using namespace dnf_composer::element;

	//	const auto field =
	//		std::dynamic_pointer_cast<NeuralField>(phenotype.getElement("nf 2"));

	//	static constexpr double maxActivation = 20.0;
	//	static const double minActivation = field->getParameters().startingRestingLevel;

	//	 double currentMaxActivation = field->getHighestActivation();

	//	if (std::isnan(currentMaxActivation))
	//	{
	//		log(tools::logger::LogLevel::ERROR, "Current max activation is NaN.");
	//		currentMaxActivation = 0.0;
	//	}
	//	else
	//	{
	//		std::cout << "Current max activation: " << currentMaxActivation << std::endl;
	//	}

	//	const double sustainabilityFitness = (currentMaxActivation - minActivation) / (maxActivation - minActivation);

	//	if (std::isnan(sustainabilityFitness))
	//	{
	//		log(tools::logger::LogLevel::ERROR, "Sustainability fitness is NaN.");
	//		log(tools::logger::LogLevel::ERROR, "Current max activation: " + std::to_string(currentMaxActivation) +
	//			" min activation: " + std::to_string(minActivation) + " max activation: " + std::to_string(maxActivation));
	//	}
	//	return sustainabilityFitness;
	//}


}
