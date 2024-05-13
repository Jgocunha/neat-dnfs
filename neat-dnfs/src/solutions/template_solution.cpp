#include "solutions/template_solution.h"

namespace neat_dnfs
{
	TemplateSolution::TemplateSolution(const SolutionTopology& initialTopology)
		:Solution(initialTopology)
	{
	}

	SolutionPtr TemplateSolution::clone() const
	{
		TemplateSolution solution(initialTopology);
		auto clonedSolution = std::make_shared<TemplateSolution>(solution);

		return clonedSolution;
	}

	void TemplateSolution::evaluate()
	{
		buildPhenotype();
		evaluatePhenotype();
	}

	void TemplateSolution::createRandomInitialConnectionGenes()
	{
		static constexpr double connectionProbability = 0.0;
		for(int i = 0; i < initialTopology.numInputGenes * initialTopology.numOutputGenes; ++i)
			if(dnf_composer::tools::utils::generateRandomNumber(0.0, 1.0) < connectionProbability)
				genome.addRandomInitialConnectionGene();
	}

	void TemplateSolution::evaluatePhenotype()
	{
		using namespace dnf_composer;
		using namespace dnf_composer::element;

		std::map<std::string, std::vector<std::vector<double>>> inputFieldsBumpPositions;
		inputFieldsBumpPositions["nf 1"] = { {25.0}, {50.0}, {75.0} };
		inputFieldsBumpPositions["nf 2"] = { {25.0, 50.0, 75}, {25.0, 50.0}, {25.0} };

		static const ElementSpatialDimensionParameters dimension = { 100, 1 };
		static constexpr double width = 5.0;
		static constexpr double amplitude = 25.0;
		static constexpr bool circular = false;
		static constexpr bool normalize = false;

		// Vector to store iterators for each vector in the map
		std::vector<std::vector<std::vector<double>>::const_iterator> iterators;
		// Initialize iterators for each vector in the map
		for (const auto& entry : inputFieldsBumpPositions)
			iterators.push_back(entry.second.begin());

		// Loop through the vectors
		bool moreData = true;
		while (moreData)
		{
			moreData = false; // Assume this is the last row unless proven otherwise

			// Iterate over each iterator and output its current value
			auto it = iterators.begin();
			for (auto& entry : inputFieldsBumpPositions)
			{
				if (*it != entry.second.end())
				{
					for (const double val : **it)
						addStimulus(entry.first, val);

					// Check if there is more data
					if (*it != entry.second.end()) moreData = true;
					++(*it);  // Move to next element
				}
				++it;  // Move to next iterator
			}
			runSimulation();
			recordSimulationResults();
			removeStimulus();
		}
		updateFitness();
	}

	void TemplateSolution::runSimulation()
	{
		static constexpr int numIterations = 100;

		phenotype.init();
		for (int i = 0; i < numIterations; ++i)
			phenotype.step();
	}

	void TemplateSolution::addStimulus(const std::string & name, const double& position)
	{
		using namespace dnf_composer;
		using namespace dnf_composer::element;

		static const ElementSpatialDimensionParameters dimension = { 100, 1 };
		static constexpr double width = 5.0;
		static constexpr double amplitude = 25.0;
		static constexpr bool circular = false;
		static constexpr bool normalize = false;

		const GaussStimulusParameters gsp = {
							width, amplitude, position, circular, normalize
		};
		const std::string gsId = "gs " + name + " " + std::to_string(position);
		const auto gaussStimulus = GaussStimulus{
			{gsId, dimension}, gsp };
		phenotype.addElement(std::make_shared<GaussStimulus>(gaussStimulus));
		phenotype.createInteraction(gsId, "output", name);
	}

	void TemplateSolution::removeStimulus()
	{
		using namespace dnf_composer::element;

		for (const auto& element : phenotype.getElements())
		{
			if (element->getLabel() == GAUSS_STIMULUS)
				phenotype.removeElement(element->getUniqueName());
		}
	}

	void TemplateSolution::recordSimulationResults()
	{
		using namespace dnf_composer::element;

		const std::vector<std::string> outputFieldsNames = { "nf 3" };

		for (const auto& outputFieldName : outputFieldsNames)
		{
			const auto field =
				std::dynamic_pointer_cast<NeuralField>(phenotype.getElement(outputFieldName));
			const auto fieldBumps = field->getBumps();
			std::map<std::string, std::vector<NeuralFieldBump>> correspondingFieldBumps;
			correspondingFieldBumps[outputFieldName] = fieldBumps;
			outputFieldsBumps.push_back(correspondingFieldBumps);
			//// log the bump centroids
			//if (!fieldBumps.empty())
			//{
			//	std::string logMessage = "Bumps for field " + outputFieldName + ":";
			//	for (const auto& bump : fieldBumps)
			//		logMessage += " " + std::to_string(bump.centroid);
			//	log(tools::logger::LogLevel::INFO, logMessage);
			//}
		}
		phenotype.close();
	}

	//void TemplateSolution::updateFitness()
	//{
	//	using namespace dnf_composer::element;
	//	std::map<std::string, std::vector<double>> expectedOutputFieldBumps;
	//	expectedOutputFieldBumps["nf 3"] = { 50.0 }; // nf1 {25} * nf2 {25.0, 50.0, 75} = nf3 {50.0}
	//	expectedOutputFieldBumps["nf 3"] = { 25.0 }; // nf1 {50} * nf2 {25.0, 50.0} = nf3 {25.0}
	//	expectedOutputFieldBumps["nf 3"] = { }; // nf1 {75} * nf2 {25.0} = nf3 {}

	//	// fitness function has to reward for:
	//	// only one bump in the output field for each round of stimulus (or no bumps in the last scenario)
	//	// the centroids of the bumps in the output field are close to the target centroid (+/- 5)
	//	// the presence of bumps in the output field should be rewarded with the same weight (1/3)
	//	// the bumps in the output field are not too wide (width < 10)
	//	// the bumps in the output field are not too narrow (width > 5)
	//	// the bumps in the output field are not too high (amplitude < 20)
	//	// the bumps in the output field are not too low (amplitude > 5)
	//	// parameters.fitness = ??;
	//}

	void TemplateSolution::updateFitness()
	{
		using namespace dnf_composer::element;
		// Expected output mapping for different scenarios
		std::map<std::string, std::vector<double>> expectedOutputFieldBumps;
		//expectedOutputFieldBumps["nf 3"] = { {50.0}, {25.0}, {} }; // Correct expectations for three scenarios
		expectedOutputFieldBumps["nf 3 1"] = { 50.0 }; // nf1 {25} * nf2 {25.0, 50.0, 75} = nf3 {50.0}
		expectedOutputFieldBumps["nf 3 2"] = { 25.0 }; // nf1 {50} * nf2 {25.0, 50.0} = nf3 {25.0}
		expectedOutputFieldBumps["nf 3 3"] = { }; // nf1 {75} * nf2 {25.0} = nf3 {}

		double totalFitness = 0.0;
		const double bumpPresenceWeight = 1.0 / 3.0; // Each part of the scenario contributes equally
		const double centroidAccuracyThreshold = 5.0; // Within 5 units is considered accurate
		const double widthMin = 5.0, widthMax = 10.0;
		const double amplitudeMin = 5.0, amplitudeMax = 20.0;

		for (size_t i = 0; i < outputFieldsBumps.size(); ++i) {
			const auto& observedBumps = outputFieldsBumps[i]["nf 3"];
			const auto& expectedBumps = expectedOutputFieldBumps["nf 3 " + std::to_string(i)];

			// Evaluate bump presence
			if (expectedBumps.empty()) {
				if (observedBumps.empty()) {
					totalFitness += bumpPresenceWeight;
				}
				else {
					totalFitness -= bumpPresenceWeight;
				}
			}
			else {
				if (observedBumps.size() == 1) {
					totalFitness += bumpPresenceWeight;
					const auto& bump = observedBumps.front();

					// Evaluate centroid accuracy
					if (std::abs(bump.centroid - expectedBumps[0]) <= centroidAccuracyThreshold) {
						totalFitness += bumpPresenceWeight;
					}
					else {
						totalFitness -= bumpPresenceWeight;
					}

					// Evaluate bump width
					if (bump.width >= widthMin && bump.width <= widthMax) {
						totalFitness += bumpPresenceWeight;
					}
					else {
						totalFitness -= bumpPresenceWeight;
					}

					// Evaluate bump amplitude
					if (bump.amplitude >= amplitudeMin && bump.amplitude <= amplitudeMax) {
						totalFitness += bumpPresenceWeight;
					}
					else {
						totalFitness -= bumpPresenceWeight;
					}
				}
				else {
					totalFitness -= 4 * bumpPresenceWeight; // Heavy penalty for wrong number of bumps
				}
			}
		}

		// Normalize total fitness score
		totalFitness = std::max(0.1, totalFitness); // Ensure fitness is not negative
		parameters.fitness = totalFitness; // Store the computed fitness in a member variable
	}
}


