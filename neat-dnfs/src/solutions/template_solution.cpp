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

	void TemplateSolution::evaluatePhenotype()
	{
		using namespace dnf_composer;

		// add inputs to the neural fields
		const element::GaussStimulusParameters gcp_a = { 5, 25, 25, false, false };
		const std::shared_ptr<element::GaussStimulus> gauss_stimulus
		(new element::GaussStimulus({ "gauss stimulus", {100, 1.0 }}, gcp_a));
		phenotype.addElement(gauss_stimulus);
		phenotype.createInteraction(gauss_stimulus->getUniqueName(), "output", "nf 1");

		// add inputs to the neural fields
		const element::GaussStimulusParameters gcp_b = { 5, 25, 75, false, false };
		const std::shared_ptr<element::GaussStimulus> gauss_stimulus_b
		(new element::GaussStimulus({ "gauss stimulus b", {100, 1.0 } }, gcp_b));
		phenotype.addElement(gauss_stimulus_b);
		phenotype.createInteraction(gauss_stimulus_b->getUniqueName(), "output", "nf 2");

		// simulate behavior
		static constexpr int numSteps = 1000;
		phenotype.init();
		for(int i = 0; i < numSteps; i++)
			phenotype.step();
		const auto ael = std::dynamic_pointer_cast<element::NeuralField>(phenotype.getElement("nf 4"));
		const double centroid = ael->getCentroid();
		std::cout << "Centroid: " << centroid << std::endl;
		phenotype.close();

		// Calculate fitness
		static constexpr double targetCentroid = 50;
		parameters.fitness = 1 / (1 + std::abs(centroid - targetCentroid));
	}

	void TemplateSolution::createRandomInitialConnectionGenes()
	{
		static constexpr double connectionProbability = 0.0;
		for(int i = 0; i < initialTopology.numInputGenes * initialTopology.numOutputGenes; ++i)
			if(dnf_composer::tools::utils::generateRandomNumber(0.0, 1.0) < connectionProbability)
				genome.addRandomInitialConnectionGene();
	}

}