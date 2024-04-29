#pragma once


#include "genome.h"
#include "phenotype.h"

namespace neat_dnfs
{
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
		virtual void mutate() = 0;
		virtual std::shared_ptr<Solution> clone() const = 0;

		Phenotype getPhenotype() const;
		Genome getGenome() const;
		SolutionParameters getParameters() const;
	private:
		virtual void buildPhenotype() = 0;
		virtual void evaluatePhenotype() = 0;
		virtual void createRandomInitialConnectionGenes() = 0;
		void createInputGenes();
		void createOutputGenes();
	};
}

