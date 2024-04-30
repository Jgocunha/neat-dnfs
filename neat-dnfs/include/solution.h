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
		int speciesId;
		double adjustedFitness;
		int age;

		SolutionParameters(int numInputGenes, int numOutputGenes, double fitness = 0.0,
			int speciesId = 0,
			double adjustedFitness = 0.0,
			int age = 0)
			:  numInputGenes(numInputGenes), numOutputGenes(numOutputGenes),
			fitness(fitness),
			speciesId(speciesId),
			adjustedFitness(adjustedFitness),
			age(age)
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
		void clearGenerationalInnovations();
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

