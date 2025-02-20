#pragma once

#include "genome.h"

namespace neat_dnfs
{
	class Solution;
	typedef std::shared_ptr<dnf_composer::Simulation> PhenotypePtr;
	typedef std::shared_ptr<Solution> SolutionPtr;

	struct SolutionTopology
	{
		std::vector<std::pair<FieldGeneType, dnf_composer::element::ElementDimensions>> geneTopology;

		SolutionTopology(const std::vector<std::pair<FieldGeneType, dnf_composer::element::ElementDimensions>>& geneTypeAndDimension)
			: geneTopology(geneTypeAndDimension)
		{}

		bool operator==(const SolutionTopology& other) const
		{
			return geneTopology == other.geneTopology;
		}

		/*std::string toString() const
		{
			std::string result = "Initial solution topology \n"
				"InputGenes: " + std::to_string(numInputGenes) +
				", OutputGenes: " + std::to_string(numOutputGenes) +
				", HiddenGenes: " + std::to_string(numHiddenGenes) +
				", Connections: " + std::to_string(numConnections);
			return result;
			return "Initial solution topology to do";
		}

		void print() const
		{
			tools::logger::log(tools::logger::INFO, toString());
		}*/
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
			std::string result = 
				"{ Fitness: " + std::to_string(fitness) +
				", AdjustedFitness: " + std::to_string(adjustedFitness) +
				", Age: " + std::to_string(age) + " }";
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
		static inline int uniqueIdentifierCounter = 0;
		int id;
		std::string name;
		SolutionTopology initialTopology;
		SolutionParameters parameters;
		PhenotypePtr phenotype;
		Genome genome;
	public:
		Solution(const SolutionTopology& initialTopology);
		virtual SolutionPtr clone() const = 0;
		SolutionPtr crossover(const SolutionPtr& other);
		void evaluate();
		void initialize();
		void mutate();
		PhenotypePtr getPhenotype() const;
		Genome getGenome() const;
		SolutionParameters getParameters() const;
		std::string getName() const { return name; }
		std::string getAddress() const;
		double getFitness() const;
		size_t getGenomeSize() const;
		size_t getNumFieldGenes() const { return genome.getFieldGenes().size(); }
		std::vector<int> getInnovationNumbers() const;
		int getId() const { return id; }
		void clearGenerationalInnovations() const;
		void incrementAge();
		void setAdjustedFitness(double adjustedFitness);
		void buildPhenotype();
		void clearPhenotype();
		void addFieldGene(const FieldGene& gene);
		void addConnectionGene(const ConnectionGene& gene);
		bool containsConnectionGene(const ConnectionGene& gene) const;
		bool containsConnectionGeneWithTheSameInputOutputPair(const ConnectionGene& gene) const;
		bool hasTheSameTopology(const SolutionPtr& other) const;
		bool hasTheSameParameters(const SolutionPtr& other) const;
		bool hasTheSameGenome(const SolutionPtr& other) const;
		bool hasTheSamePhenotype(const SolutionPtr& other) const;
		bool operator==(const SolutionPtr& other) const;
		std::string toString() const;
		void print() const;
		virtual void createPhenotypeEnvironment() = 0;
	private:
		void createInputGenes();
		void createOutputGenes();
		void createHiddenGenes();
		void translateGenesToPhenotype();
		void translateConnectionGenesToPhenotype();
	protected:
		virtual void testPhenotype() = 0;
		void initSimulation();
		void stopSimulation();
		void runSimulation(const int iterations);
		bool runSimulationUntilFieldStable(const std::string& targetElement);
		void addGaussianStimulus(const std::string& targetElement, 
			const dnf_composer::element::GaussStimulusParameters& stimulusParameters,
			const dnf_composer::element::ElementDimensions& dimensions
		);
		void removeGaussianStimuli();
		double oneBumpAtPositionWithAmplitudeAndWidth(const std::string& fieldName,
			const double& position, const double& amplitude, const double& width);
		double twoBumpsAtPositionWithAmplitudeAndWidth(const std::string& fieldName,
						const double& position1, const double& amplitude1, const double& width1,
						const double& position2, const double& amplitude2, const double& width2);
		double threeBumpsAtPositionWithAmplitudeAndWidth(const std::string& fieldName,
									const double& position1, const double& amplitude1, const double& width1,
									const double& position2, const double& amplitude2, const double& width2,
									const double& position3, const double& amplitude3, const double& width3);
		double closenessToRestingLevel(const std::string& fieldName);
		double preShapedness(const std::string& fieldName);
		double negativePreShapednessAtPosition(const std::string& fieldName, const double& position);
		double justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth(const std::string& fieldName,
			const std::vector<double>& positions, const double& amplitude, const double& width);
		void removeGaussianStimuliFromField(const std::string& fieldName);
		double noBumps(const std::string& fieldName);
		double iterationsUntilBump(const std::string& fieldName, double targetIterations);
		double iterationsUntilBumpWithAmplitude(const std::string& fieldName, double targetIterations, double targetAmplitude);
	};
}
