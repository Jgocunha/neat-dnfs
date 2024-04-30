#include "template_solution.h"

namespace neat_dnfs
{
	TemplateSolution::TemplateSolution(const SolutionParameters& parameters)
		:Solution(parameters)
	{
	}

	SolutionPtr TemplateSolution::clone() const
	{
		TemplateSolution solution(parameters);
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
		const element::GaussStimulusParameters gcp_a = { 5, 25, 50, false, false };
		const std::shared_ptr<element::GaussStimulus> gauss_stimulus
		(new element::GaussStimulus({ "gauss stimulus", {100, 1.0 }}, gcp_a));
		phenotype.addElement(gauss_stimulus);
		phenotype.createInteraction(gauss_stimulus->getUniqueName(), "output", "nf 1");

		// simulate behavior
		static constexpr int numSteps = 1000;
		phenotype.init();
		for(int i = 0; i < numSteps; i++)
			phenotype.step();
		const auto ael = std::dynamic_pointer_cast<element::NeuralField>(phenotype.getElement("nf 11"));
		const double centroid = ael->getCentroid();
		std::cout << "Centroid: " << centroid << std::endl;
		phenotype.close();

		// Calculate fitness
		static constexpr double targetCentroid = 50;
		parameters.fitness = 1 / (1 + std::abs(centroid - targetCentroid));

		std::cout << "Fitness: " << parameters.fitness << std::endl;
		if( parameters.fitness > 0.8)
		{
			std::cout << "Solution found!" << std::endl;
			//std::cout << "Genome: " << genome << std::endl;
			//std::cout << "Phenotype: " << phenotype << std::endl;
		}
	}

	void TemplateSolution::createRandomInitialConnectionGenes()
	{
		static constexpr double connectionProbability = 0.0;
		for(int i = 0; i < parameters.numInputGenes * parameters.numOutputGenes; ++i)
			if(dnf_composer::tools::utils::generateRandomNumber(0.0, 1.0) < connectionProbability)
				genome.addRandomInitialConnectionGene();
	}
}