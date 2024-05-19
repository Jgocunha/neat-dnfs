#pragma once

#include <tools/utils.h>

#include "genome.h"
#include "simulate_behaviour_wizard.h"

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

		SolutionParameters(double fitness = 0.0, 
			double adjustedFitness = 0.0, int age = 0)
			: fitness(fitness), adjustedFitness(adjustedFitness), age(age)
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

		virtual void evaluate() = 0;
		virtual SolutionPtr clone() const = 0;

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
	protected:
		void buildPhenotype();
	private:
		void createInputGenes();
		void createOutputGenes();
		void createRandomInitialConnectionGenes();
		void translateGenesToPhenotype();
		void translateConnectionGenesToPhenotype();
	public:
		virtual SolutionPtr crossover(const SolutionPtr& other) = 0;
		void addFieldGene(const FieldGene& gene);
		void addConnectionGene(const ConnectionGene& gene);
		bool containsConnectionGene(const ConnectionGene& gene) const;
	};
}

