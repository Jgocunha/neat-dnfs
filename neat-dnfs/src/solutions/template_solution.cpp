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
		static constexpr int numIterations = 200;

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

	void TemplateSolution::updateFitness()
	{
		// remove last element from outputFieldsBumps
		outputFieldsBumps.pop_back();

		static const std::vector<double> expectedCentroidPositions = { 50.0, 25.0, 0.0 };
		static constexpr int targetNumBumps = 1;
		static constexpr double targetBumpWidth = 5.0;
		static constexpr double targetBumpAmplitude = 10.0;

		std::vector<double> fitnessValues = {0.0, 0.0, 0.0};
		parameters.fitness = 0.0;

		for (size_t i = 0; i < outputFieldsBumps.size(); ++i) 
		{
			const auto& observedBumps = outputFieldsBumps[i]["nf 3"];
			if(observedBumps.empty())
			{
				if (expectedCentroidPositions[i] == 0.0)
					fitnessValues[i] = 1;
				else
					fitnessValues[i] = 0;
			}
			else
			{
				const int numBumps = static_cast<int>(observedBumps.size());
				//fitnessValues[i] += 1 - std::abs(numBumps - targetNumBumps);
			//}
			//if(!observedBumps.empty())
			//{
				const auto& bump = observedBumps.front();
				const double centroidDifference = std::abs(bump.centroid - expectedCentroidPositions[i]);
				const double widthDifference = std::abs(bump.width - targetBumpWidth);
				const double amplitudeDifference = std::abs(bump.amplitude - targetBumpAmplitude);

				fitnessValues[i] += 1 / (1 + centroidDifference);
				//fitnessValues[i] += 0.5 / (1 + widthDifference);
				//fitnessValues[i] += 0.5 / (1 + amplitudeDifference);
			}
		}

		parameters.fitness = std::accumulate(fitnessValues.begin(), fitnessValues.end(), 0.0);

		outputFieldsBumps.clear();
		//parameters.fitness = 0.4;
	}

}


