#pragma once


#include "genome.h"
#include "phenotype.h"

namespace neat_dnfs
{
	class Solution;
	typedef dnf_composer::Simulation Phenotype;
	typedef std::shared_ptr<Solution> SolutionPtr;

	struct SolutionParameters
	{
		int numInputGenes;
		int numOutputGenes;

		double fitness;
		double adjustedFitness;
		int age;
		int species;

		SolutionParameters(int numInputGenes, int numOutputGenes, double fitness = 0.0,
			double adjustedFitness = 0.0,
			int age = 0, int species = 0)
			:  numInputGenes(numInputGenes), numOutputGenes(numOutputGenes),
			fitness(fitness),
			adjustedFitness(adjustedFitness),
			age(age),
			species(species)
		{
		}
	};

	class Solution
	{
	protected:
		SolutionParameters parameters;
		Phenotype phenotype;
		Genome genome;
	public:
		Solution(const SolutionParameters& parameters);
		virtual ~Solution() = default;

		void initialize();
		virtual void evaluate() = 0;
		void mutate();
		virtual SolutionPtr clone() const = 0;

		Phenotype getPhenotype() const;
		Genome getGenome() const;
		SolutionParameters getParameters() const;
		double getFitness() const;
		int getGenomeSize() const;
		void clearGenerationalInnovations() const;
		std::vector<uint16_t> getInnovationNumbers() const;

		void setSpecies(int species);
		void incrementAge();
	protected:
		void buildPhenotype();
	private:
		virtual void evaluatePhenotype() = 0;
		virtual void createRandomInitialConnectionGenes() = 0;
		void createInputGenes();
		void createOutputGenes();
		void translateGenesToPhenotype();
		void translateConnectionGenesToPhenotype();
	};
}

