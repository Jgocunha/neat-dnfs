#include "solutions/template_solution.h"

namespace neat_dnfs
{
	TemplateSolution::TemplateSolution(const SolutionTopology& topology)
		:Solution(topology)
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
		parameters.fitness = 0.0;
		addStimulus("nf 1", 50.0);
		runSimulation();
		updateFitness(50.0);
		removeStimulus();
		runSimulation();
	}

	//void TemplateSolution::evaluatePhenotype()
	//{
	//	using namespace dnf_composer;
	//	using namespace dnf_composer::element;

	//	// XOR
	//	// 0 0 -> 0
	//	// 0 1 -> 1
	//	// 1 0 -> 1
	//	// 1 1 -> 0

	//	constexpr uint8_t numBehaviors = 4;
	//	const std::vector<double> stimulusFieldA { 0.0, 0.0, 50.0, 50.0 };
	//	const std::vector<double> stimulusFieldB { 0.0, 50.0, 0.0, 50.0 };
	//	const std::vector<double> expectedOutput { 0.0, 50.0, 50.0, 0.0 };
	//	std::vector<double> output(numBehaviors);

	//	parameters.fitness = 0.0;
	//	for( uint8_t i = 0; i < numBehaviors; i++)
	//	{
	//		addStimulus("nf 1", stimulusFieldA[i]);
	//		addStimulus("nf 2", stimulusFieldB[i]);
	//		runSimulation();
	//		updateFitness(expectedOutput[i]);
	//		removeStimulus();
	//	}
	//}

	void TemplateSolution::addStimulus(const std::string& name, const double& position)
	{
		if (position < 0.1)
			return;
		using namespace dnf_composer;
		using namespace dnf_composer::element;

		static const ElementSpatialDimensionParameters dimension = { 100, 1.0 };
		static constexpr double width = 5.0;
		static constexpr double amplitude = 25.0;
		static constexpr bool circular = true;
		static constexpr bool normalize = true;

		GaussStimulusParameters gsp;// = {}
		gsp.position = position;

		/*const GaussStimulusParameters gsp = {
							width, amplitude, position, circular, normalize
		};*/
		const std::string gsId = "gs " + name + " " + std::to_string(position);
		const auto gaussStimulus = GaussStimulus{
			{gsId, dimension}, gsp };
		phenotype.addElement(std::make_shared<GaussStimulus>(gaussStimulus));
		phenotype.createInteraction(gsId, "output", name);
	}

	void TemplateSolution::runSimulation()
	{
		static constexpr int numIterations = 1000;

		phenotype.init();
		for (int i = 0; i < numIterations; ++i)
			phenotype.step();
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

	void TemplateSolution::updateFitness(double expectedOutput)
	{
		using namespace dnf_composer;
		using namespace dnf_composer::element;

		constexpr double targetBumpWidth = 5.0;
		constexpr double targetBumpAmplitude = 10.0;

		const auto field = std::dynamic_pointer_cast<NeuralField>(phenotype.getElement("nf 2"));
		const auto fieldBumps = field->getBumps();

		if(fieldBumps.empty())
		{
			if (expectedOutput == 0.0)
				parameters.fitness += 1.0;
			else
				parameters.fitness += 0.0;
			return;
		}
		if (fieldBumps.size() > 1)
		{
			parameters.fitness += 0.0;
			return;
		}
		const auto& bump = fieldBumps.front();
		const double centroidDifference = std::abs(bump.centroid - expectedOutput);
		//const double widthDifference = std::abs(bump.width - targetBumpWidth);
		//const double amplitudeDifference = std::abs(bump.amplitude - targetBumpAmplitude);

		parameters.fitness += 1 / (1 + centroidDifference);
		//parameters.fitness += 1 / (1 + widthDifference);
		//parameters.fitness += 1 / (1 + amplitudeDifference);
	}
}


