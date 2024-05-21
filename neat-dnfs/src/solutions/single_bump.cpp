#include "solutions/single_bump.h"

namespace neat_dnfs
{
	SingleBumpSolution::SingleBumpSolution(const SolutionTopology& topology)
		: Solution(topology)
	{}

	void SingleBumpSolution::evaluate()
	{
		buildPhenotype();
		testPhenotype();
	}

	SolutionPtr SingleBumpSolution::clone() const
	{
		SingleBumpSolution solution(initialTopology);
		auto clonedSolution = std::make_shared<SingleBumpSolution>(solution);

		return clonedSolution;
	}

	SolutionPtr SingleBumpSolution::crossover(const SolutionPtr& other)
	{
		const SolutionPtr self = std::make_shared<SingleBumpSolution>(*this);

		const double fitnessDifference = std::abs(this->getFitness() - other->getFitness());

		const SolutionPtr moreFitParent = this->getFitness() > other->getFitness() ? self : other;
		const SolutionPtr lessFitParent = this->getFitness() > other->getFitness() ? other : self;

		SolutionPtr offspring = std::make_shared<SingleBumpSolution>(initialTopology);

		for (const auto& gene : moreFitParent->getGenome().getFieldGenes())
			offspring->addFieldGene(gene);

		const auto& parentConnectionGenes = moreFitParent->getGenome().getConnectionGenes();
		for (const auto& gene : parentConnectionGenes)
		{
			// Matching genes are inherited randomly from either parent
			if (lessFitParent->containsConnectionGene(gene))
			{
				const auto lessFitGene = lessFitParent->getGenome().getConnectionGeneByInnovationNumber(gene.getInnovationNumber());
				if (tools::utils::generateRandomInt(0, 1))
					offspring->addConnectionGene(gene);
				else
					offspring->addConnectionGene(lessFitGene);
			}
			else
			{
				// Disjoint and excess genes are inherited from the more fit parent
				// unless the fitness difference is 0, in which case the gene is inherited randomly
				// here we are only considering the most fit parent
				// later add randomly from the less fit parent
				if (fitnessDifference < 1e-6)
				{
					if (tools::utils::generateRandomInt(0, 1))
						offspring->addConnectionGene(gene);
				}
				else
					offspring->addConnectionGene(gene);
			}
		}

		// If the fitness is the same we still have to randomly inherit the excess and disjoint genes
		// from the less fit parent
		if (fitnessDifference < 1e-6)
		{
			const auto& lessFitParentConnectionGenes = lessFitParent->getGenome().getConnectionGenes();
			for (const auto& gene : lessFitParentConnectionGenes)
			{
				// Non-matching genes are inherited randomly from the less fit parent
				if (!moreFitParent->containsConnectionGene(gene))
				{
					if (tools::utils::generateRandomInt(0, 1))
					{
						offspring->addConnectionGene(gene);
						// make sure the field genes are also added
						const uint16_t inFieldGeneId = gene.getInFieldGeneId();
						const uint16_t outFieldGeneId = gene.getOutFieldGeneId();
						for (const auto& fieldGene : lessFitParent->getGenome().getFieldGenes())
						{
							if (fieldGene.getParameters().id == inFieldGeneId || fieldGene.getParameters().id == outFieldGeneId)
								offspring->addFieldGene(fieldGene);
						}

					}
				}
			}

		}
		

		return offspring;
	}

	void SingleBumpSolution::testPhenotype()
	{
		static constexpr double expectedBumpPosition = 50;
		addStimulus("nf 1", expectedBumpPosition);
		runSimulation();
		updateFitness();
		removeStimulus();
		runSimulation();
	}

	void SingleBumpSolution::updateFitness()
	{
		using namespace dnf_composer::element;

		static constexpr double expectedBumpPosition = 50;
		static constexpr double expectedBumpWidth = 5;
		static constexpr double expectedBumpAmplitude = 5;

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

		parameters.fitness = 0.8 / (1.0 + centroidDifference);
		parameters.fitness += 0.1 / (1.0 + widthDifference);
		parameters.fitness += 0.1 / (1.0 + amplitudeDifference);
	}

	void SingleBumpSolution::runSimulation()
	{
		static constexpr int numIterations = 300;

		phenotype.init();
		for (int i = 0; i < numIterations; ++i)
			phenotype.step();
	}

	void SingleBumpSolution::addStimulus(const std::string& targetElement, const double& position)
	{
		using namespace dnf_composer;
		using namespace dnf_composer::element;

		static const ElementSpatialDimensionParameters dimension = 
		{ DimensionConstants::xSize, DimensionConstants::dx };
		static constexpr double width = 5.0;
		static constexpr double amplitude = 15.0;
		static constexpr bool circular = false;
		static constexpr bool normalize = false;

		const GaussStimulusParameters gsp = {
							width, amplitude, position, circular, normalize
		};
		const std::string gsId = "gs " + targetElement + " " + std::to_string(position);
		const auto gaussStimulus = GaussStimulus{
			{gsId, dimension}, gsp };
		phenotype.addElement(std::make_shared<GaussStimulus>(gaussStimulus));
		phenotype.createInteraction(gsId, "output", targetElement);
	}

	void SingleBumpSolution::removeStimulus()
	{
		using namespace dnf_composer::element;

		for (const auto& element : phenotype.getElements())
		{
			if (element->getLabel() == GAUSS_STIMULUS)
				phenotype.removeElement(element->getUniqueName());
		}
	}
}
