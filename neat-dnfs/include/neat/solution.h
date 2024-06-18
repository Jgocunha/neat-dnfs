#pragma once

#include "genome.h"

namespace neat_dnfs
{
	class Solution;
	typedef dnf_composer::Simulation Phenotype;
	typedef std::shared_ptr<Solution> SolutionPtr;

	struct SolutionTopology
	{
		int numInputGenes;
		int numOutputGenes;
		int numHiddenGenes;
		int numConnections;

		SolutionTopology(int numInputGenes = 3, int numOutputGenes = 1,
			int numHiddenGenes = 0, int numConnections = 0)
			: numInputGenes(numInputGenes), numOutputGenes(numOutputGenes),
			numHiddenGenes(numHiddenGenes), numConnections(numConnections)
		{}
	};

	struct SolutionParameters
	{
		double fitness;
		double adjustedFitness;
		int age;
		std::vector<dnf_composer::element::NeuralFieldBump> bumps;

		SolutionParameters(double fitness = 0.0,
			double adjustedFitness = 0.0, int age = 0)
			: fitness(fitness), adjustedFitness(adjustedFitness), age(age)
			, bumps({})
		{}
	};

	class Solution : public std::enable_shared_from_this<Solution>
	{
	protected:
		SolutionTopology initialTopology;
		SolutionParameters parameters;
		Phenotype phenotype;
		Genome genome;
	public:
		Solution(const SolutionTopology& initialTopology);
		virtual SolutionPtr clone() const = 0;
		virtual SolutionPtr crossover(const SolutionPtr& other);
		void evaluate();
		void initialize();
		void mutate();
		Phenotype getPhenotype() const;
		Genome getGenome() const;
		SolutionParameters getParameters() const;
		double getFitness() const;
		size_t getGenomeSize() const;
		std::vector<uint16_t> getInnovationNumbers() const;
		void clearGenerationalInnovations() const;
		void incrementAge();
		void setAdjustedFitness(double adjustedFitness);
		void buildPhenotype();
		void clearPhenotype();
		void addFieldGene(const FieldGene& gene);
		void addConnectionGene(const ConnectionGene& gene);
		bool containsConnectionGene(const ConnectionGene& gene) const;
	private:
		void createInputGenes();
		void createOutputGenes();
		void createHiddenGenes();
		void createRandomInitialConnectionGenes();
		void translateGenesToPhenotype();
		void translateConnectionGenesToPhenotype();
	public:
	protected:
		virtual void updateFitness() = 0;
		virtual void testPhenotype() = 0;
		void initSimulation();
		void stopSimulation();
		void runSimulation(const uint16_t iterations);
		void runSimulationUntilFieldStable(const std::string& targetElement);
		void addGaussianStimulus(const std::string& targetElement, 
			const dnf_composer::element::GaussStimulusParameters& parameters);
		void removeGaussianStimuli();
	};
}
