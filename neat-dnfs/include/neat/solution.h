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

		bool operator==(const SolutionTopology& other) const
		{
			return numInputGenes == other.numInputGenes &&
				numOutputGenes == other.numOutputGenes &&
				numHiddenGenes == other.numHiddenGenes &&
				numConnections == other.numConnections;
		}

		std::string toString() const
		{
			std::string result = "Initial solution topology \n"
				"InputGenes: " + std::to_string(numInputGenes) +
				", OutputGenes: " + std::to_string(numOutputGenes) +
				", HiddenGenes: " + std::to_string(numHiddenGenes) +
				", Connections: " + std::to_string(numConnections);
			return result;
		}

		void print() const
		{
			tools::logger::log(tools::logger::INFO, toString());
		}
	};

	struct SolutionParameters
	{
		double fitness;
		double adjustedFitness;
		int age;
		// bumps really should be a parameter?
		std::vector<dnf_composer::element::NeuralFieldBump> bumps;

		SolutionParameters(double fitness = 0.0,
			double adjustedFitness = 0.0, int age = 0)
			: fitness(fitness), adjustedFitness(adjustedFitness), age(age)
			, bumps({})
		{}

		bool operator==(const SolutionParameters& other) const
		{
			constexpr double epsilon = 1e-6;
			return std::abs(fitness - other.fitness) < epsilon &&
				std::abs(adjustedFitness - other.adjustedFitness) < epsilon &&
				age == other.age;
		}

		std::string toString() const
		{
			std::string result = "Solution parameters \n"
				"Fitness: " + std::to_string(fitness) +
				", AdjustedFitness: " + std::to_string(adjustedFitness) +
				", Age: " + std::to_string(age);
			return result;
		}

		void print() const
		{
			tools::logger::log(tools::logger::INFO, toString());
		}
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
		bool hasTheSameTopology(const SolutionPtr& other) const;
		bool hasTheSameParameters(const SolutionPtr& other) const;
		bool hasTheSameGenome(const SolutionPtr& other) const;
		bool hasTheSamePhenotype(const SolutionPtr& other) const;
		bool operator==(const SolutionPtr& other) const;
		std::string toString() const;
		void print() const;
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
		bool runSimulationUntilFieldStable(const std::string& targetElement);
		void addGaussianStimulus(const std::string& targetElement, 
			const dnf_composer::element::GaussStimulusParameters& parameters);
		void removeGaussianStimuli();
	};
}
