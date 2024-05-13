#include "simulate_behaviour_wizard.h"

namespace neat_dnfs
{
	SimulateBehaviourWizard::SimulateBehaviourWizard(const Phenotype& phenotype)
		: phenotype(std::make_shared<Phenotype>(phenotype))
	{}

	void SimulateBehaviourWizard::setOutputFieldsNames(const std::vector<std::string>& names)
	{
		for (const std::string& name : names)
			outputFieldsNames.push_back(name);
	}


	void SimulateBehaviourWizard::addInputFieldBumps(const std::string& name,
		const std::vector<std::vector<double>>& bump)
	{
		inputFieldsBumpPositions[name] = bump;
	}

	std::vector<std::map<std::string, std::vector<dnf_composer::element::NeuralFieldBump>>>
		SimulateBehaviourWizard::getResultingBumps()
	{
		return outputFieldsBumps;
	}

	void SimulateBehaviourWizard::simulate()
	{
		using namespace dnf_composer;

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
 			recordOutputFieldBumps();
			removeStimulus();
		}
	}

	void SimulateBehaviourWizard::addStimulus(const std::string& name, const double& position)
	{
		using namespace dnf_composer;

		static const element::ElementSpatialDimensionParameters dimension = { 100, 1 };
		static constexpr double width = 5.0;
		static constexpr double amplitude = 25.0;
		static constexpr bool circular = false;
		static constexpr bool normalize = false;

		const element::GaussStimulusParameters gsp = {
							width, amplitude, position, circular, normalize
		};
		const std::string gsId = "gs " + name + " " + std::to_string(position);
		const auto gaussStimulus = element::GaussStimulus{
			{gsId, dimension}, gsp };
		phenotype->addElement(std::make_shared<element::GaussStimulus>(gaussStimulus));
		phenotype->createInteraction(gsId, "output", name);
	}


	void SimulateBehaviourWizard::removeStimulus()
	{
		using namespace dnf_composer;

		for (const auto& element : phenotype->getElements())
		{
			if (element->getLabel() == element::GAUSS_STIMULUS)
				phenotype->removeElement(element->getUniqueName());
		}
	}

	void SimulateBehaviourWizard::runSimulation()
	{
		static constexpr int numIterations = 1000;

		phenotype->init();
		for (int i = 0; i < numIterations; ++i)
			phenotype->step();
		auto field_1 =
			std::dynamic_pointer_cast<dnf_composer::element::NeuralField>(phenotype->getElement("nf 1"));
		auto field_2 =
			std::dynamic_pointer_cast<dnf_composer::element::NeuralField>(phenotype->getElement("nf 2"));
		auto field_3 =
			std::dynamic_pointer_cast<dnf_composer::element::NeuralField>(phenotype->getElement("nf 3"));
		double centroid = field_1->getCentroid();
		if (centroid > 0)
			std::cout << "Centroid: " << centroid << std::endl;
		const double centroid_2 = field_2->getCentroid();
		if (centroid_2 > 0)
			std::cout << "Centroid: " << centroid << std::endl;
		double centroid_3 = field_3->getCentroid();
		if(centroid_3 > 0)
			std::cout << "Centroid: " << centroid << std::endl;
	}

	void SimulateBehaviourWizard::recordOutputFieldBumps()
	{
		using namespace dnf_composer;

		for (const auto& outputFieldName : outputFieldsNames)
		{
			const auto field =
				std::dynamic_pointer_cast<element::NeuralField>(phenotype->getElement(outputFieldName));
			const auto fieldBumps = field->getBumps();
			std::map<std::string, std::vector<element::NeuralFieldBump>> correspondingFieldBumps;
			correspondingFieldBumps[outputFieldName] = fieldBumps;
			outputFieldsBumps.push_back(correspondingFieldBumps);
		}

		phenotype->close();
	}

}

