#pragma once

#include "genome.h"
#include "neat_tools/logger.h"

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
	};

	struct SolutionParameters
	{
		double fitness;
		double adjustedFitness;
		int age;
		int speciesId;
		std::vector<double> partialFitness;
		std::vector<dnf_composer::element::NeuralFieldBump> bumps;

		SolutionParameters(double fitness = 0.0,
			double adjustedFitness = 0.0, int age = 0)
			: fitness(fitness), adjustedFitness(adjustedFitness), age(age), speciesId(-1)
			, partialFitness({}), bumps({})
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
				" fit.: " + std::to_string(fitness) +
				", part.: (";

			for (const auto& partial : partialFitness)
			{
				result += std::to_string(partial) + ", ";
			}

			result += "), spec.: " + std::to_string(speciesId) +
				", adj.fit.: " + std::to_string(adjustedFitness) +
				", age: " + std::to_string(age);
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
		dnf_composer::Simulation phenotype;
		Genome genome;
		std::tuple <int, int> parents;
	public:
		virtual ~Solution() = default;

		explicit Solution(const SolutionTopology& initialTopology);
		Solution(SolutionTopology  initialTopology, dnf_composer::Simulation  phenotype);
		virtual SolutionPtr clone() const = 0;
		virtual SolutionPtr copy() const = 0;
		SolutionPtr crossover(const SolutionPtr& other);
		void evaluate();
		void initialize();
		void mutate();
		void setSpeciesId(int speciesId);
		void setParents(int parent1, int parent2);
		int getSpeciesId() const { return parameters.speciesId; }
		std::tuple<int, int> getParents() const { return parents; }
		dnf_composer::Simulation getPhenotype() const;
		Genome getGenome() const;
		SolutionParameters getParameters() const;
		std::string getName() const { return name; }
		std::string getAddress() const;
		double getFitness() const;
		size_t getGenomeSize() const;
		size_t getNumFieldGenes() const { return genome.getFieldGenes().size(); }
		size_t getNumConnectionGenes() const { return genome.getConnectionGenes().size(); }
		std::vector<int> getInnovationNumbers() const;
		int getId() const { return id; }
		static void clearGenerationalInnovations();
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
		std::string toString() const;
		void print() const;
		virtual void createPhenotypeEnvironment() = 0;
		static void resetUniqueIdentifier();
		void translatePhenotypeToGenome();
		void clearGenome();
		void clearLastMutations();
	private:
		void createInputGenes();
		void createOutputGenes();
		void createSeededHiddenGenes();
		void createSeededConnectionGenes();
		void translateGenesToPhenotype();
		void translateConnectionGenesToPhenotype();

	protected:
		virtual void testPhenotype() = 0;
		// validated
		void initSimulation();
		void stopSimulation();
		void runSimulation(int iterations);
		void addGaussianStimulus(const std::string& targetElement,
			const dnf_composer::element::GaussStimulusParameters& stimulusParameters,
			const dnf_composer::element::ElementDimensions& dimensions
		);
		void removeGaussianStimuli();
		void removeGaussianStimuliFromField(const std::string& fieldName);
		void setGaussianStimulusParameters(const std::string& stimulusName, const dnf_composer::element::GaussStimulusParameters& parameters) const;
		double closenessToRestingLevel(const std::string& fieldName) const;
		double noBumps(const std::string& fieldName) const;
		double iterationsUntilBump(const std::string& fieldName, double targetIterations, double maxIterations, double tolerance);
		double iterationsUntilNoBump(const std::string& fieldName, double targetIterations, double maxIterations, double tolerance);

		// validated but could be improved
		double oneBumpAtPositionWithAmplitudeAndWidth(const std::string& fieldName,
			const double& position, const double& amplitude, const double& width) const;
		double twoBumpsAtPositionWithAmplitudeAndWidth(const std::string& fieldName,
						const double& position1, const double& amplitude1, const double& width1,
						const double& position2, const double& amplitude2, const double& width2) const;
		double threeBumpsAtPositionWithAmplitudeAndWidth(const std::string& fieldName,
									const double& position1, const double& amplitude1, const double& width1,
									const double& position2, const double& amplitude2, const double& width2,
									const double& position3, const double& amplitude3, const double& width3) const;
		// not validated
		//double preShapedness(const std::string& fieldName) const;
		//double preShapedness(const std::string& fieldName, const std::vector<double>& positions);
		double preShapednessAtPosition(const std::string& fieldName, double position ) const;
		double negativePreShapednessAtPosition(const std::string& fieldName, const double& position);
		double justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth(const std::string& fieldName,
		                                                                    const std::vector<double>& positions, const double& amplitude, const double& width) const;


		void moveGaussianStimulusContinuously(const std::string& name, double targetPosition, double step);
		double negativeBaseline(const std::string& fieldName) const;
	};
}
