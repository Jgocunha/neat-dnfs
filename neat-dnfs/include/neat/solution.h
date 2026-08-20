#pragma once

#include "genome.h"
#include <format>
#include <optional>

namespace neat_dnfs
{
	class Solution;
	using PhenotypePtr = std::shared_ptr<dnf_composer::Simulation>;
	using SolutionPtr = std::shared_ptr<Solution>;

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
		int speciesId{-1};
		std::vector<double> partialFitness;
		std::vector<dnf_composer::element::NeuralFieldBump> bumps;

		SolutionParameters(double fitness = 0.0,
			double adjustedFitness = 0.0, int age = 0)
			: fitness(fitness), adjustedFitness(adjustedFitness), age(age),
			partialFitness({}), bumps({})
		{}

		bool operator==(const SolutionParameters& other) const
		{
			constexpr double epsilon = 1e-6;
			return std::abs(fitness - other.fitness) < epsilon &&
				std::abs(adjustedFitness - other.adjustedFitness) < epsilon &&
				age == other.age;
		}

		[[nodiscard]] std::string toString() const
    {
        std::string partials;
        for (const auto& partial : partialFitness)
        {
            partials += std::format("{}, ", partial);
        }

        return std::format(" fit.: {}, part.: ({}) spec.: {}, adj.fit.: {}, age: {}", 
            fitness, 
            partials, 
            speciesId, 
            adjustedFitness, 
            age);
    }

		void print() const
		{
			tools::logger::log(tools::logger::INFO, toString());
		}
	};

	/// @brief Abstract base class for all NEAT-evolved solutions.
	///
	/// Subclasses must implement:
	/// - @c clone() / @c copy() — for genome inheritance during crossover
	/// - @c createPhenotypeEnvironment() — build the DNF simulation elements
	/// - @c testPhenotype() — run the simulation and compute @c parameters.fitness
	///
	/// The fitness value set in @c testPhenotype() drives species assignment,
	/// adjusted-fitness calculation, and offspring allocation.
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
		[[nodiscard]] int getSpeciesId() const { return parameters.speciesId; }
		[[nodiscard]] std::tuple<int, int> getParents() const { return parents; }
		/// @return A const reference to this solution's phenotype simulation.
		[[nodiscard]] const dnf_composer::Simulation& getPhenotype() const;
		/// @return A const reference to this solution's genome.
		[[nodiscard]] const Genome& getGenome() const;
		[[nodiscard]] SolutionParameters getParameters() const;
		[[nodiscard]] std::string getName() const { return name; }
		[[nodiscard]] std::string getAddress() const;
		[[nodiscard]] double getFitness() const;
		[[nodiscard]] size_t getGenomeSize() const;
		[[nodiscard]] size_t getNumFieldGenes() const { return genome.getFieldGenes().size(); }
		[[nodiscard]] size_t getNumConnectionGenes() const { return genome.getConnectionGenes().size(); }
		[[nodiscard]] std::vector<int> getInnovationNumbers() const;
		[[nodiscard]] int getId() const { return id; }
		static void clearGenerationalInnovations();
		void incrementAge();
		void setAdjustedFitness(double adjustedFitness);
		void buildPhenotype();
		void clearPhenotype();
		void addFieldGene(const FieldGene& gene);
		void addConnectionGene(const ConnectionGene& gene);
		[[nodiscard]] bool containsConnectionGene(const ConnectionGene& gene) const;
		[[nodiscard]] bool containsConnectionGeneWithTheSameInputOutputPair(const ConnectionGene& gene) const;
		[[nodiscard]] bool hasTheSameTopology(const SolutionPtr& other) const;
		[[nodiscard]] bool hasTheSameParameters(const SolutionPtr& other) const;
		[[nodiscard]] bool hasTheSameGenome(const SolutionPtr& other) const;
		[[nodiscard]] std::string toString() const;
		void print() const;
		virtual void createPhenotypeEnvironment() = 0;
		static void resetUniqueIdentifier();
		void translatePhenotypeToGenome();
		void clearGenome();
		void clearLastMutations();
	private:
		void createInputGenes();
		void createOutputGenes();
		void translateGenesToPhenotype();
		void translateConnectionGenesToPhenotype();
		/// @brief Casts the named phenotype element to a NeuralField, or throws
		/// std::invalid_argument if it doesn't exist or isn't a NeuralField --
		/// fitness helpers are always called with field names from the
		/// solution's own topology, so a miss means a genuine configuration bug.
		/// @param fieldName   Unique name of the phenotype element to look up.
		/// @param callerName  Name of the calling fitness helper, included in the
		///                    exception message to identify which computation failed.
		/// @return The field, guaranteed non-null.
		/// @throws std::invalid_argument if no element named @p fieldName exists in
		///         the phenotype, or if it exists but isn't a NeuralField.
		std::shared_ptr<dnf_composer::element::NeuralField> getNeuralFieldOrThrow(const std::string& fieldName, const std::string& callerName) const;

		/// @brief Converts a spatial position into a valid index into @p neuralField's
		/// component vectors (e.g. "activation"), clamping to the field's sample range.
		/// @details Positions passed to fitness helpers come from genome-derived
		/// stimulus parameters and task definitions, and can legitimately fall on or
		/// beyond a field's upper spatial bound (e.g. position == field size * d_x,
		/// the field's own extent). Clamping keeps component lookups well-defined at
		/// that boundary instead of indexing past the end of the underlying buffer.
		/// @param neuralField  Field whose spatial resolution (d_x) and sample count bound the index.
		/// @param position     Spatial position to convert.
		/// @return An index in [0, neuralField->getSize() - 1].
		static int clampedIndexForPosition(const std::shared_ptr<dnf_composer::element::NeuralField>& neuralField, double position);

		/// @brief Finds the bump in @p candidates closest to @p targetPosition and
		/// removes it from @p candidates.
		/// @details Used by the multi-bump fitness helpers to match target positions
		/// to bumps injectively: once a bump is matched to one target slot, it is no
		/// longer available to be matched -- and credited -- against another target
		/// slot (see issue #53).
		/// @param candidates      Pool of not-yet-matched bumps; the matched bump, if
		///                        any, is erased from this vector.
		/// @param targetPosition  Spatial position to match against.
		/// @return The matched bump, or std::nullopt if @p candidates was empty.
		static std::optional<dnf_composer::element::NeuralFieldBump> matchClosestBump(
			std::vector<dnf_composer::element::NeuralFieldBump>& candidates, double targetPosition);

	protected:
		/// @brief Run the simulation and write the result into @c parameters.fitness. Called by @c evaluate().
		virtual void testPhenotype() = 0;

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
		/// @brief Fitness score in [0,1] reflecting how close the field activity is to its resting level (no bump).
		double closenessToRestingLevel(const std::string& fieldName) const;
		/// @brief Returns 1.0 if the field has no active bump, 0.0 otherwise.
		double noBumps(const std::string& fieldName) const;
		/// @brief Fitness score that rewards forming a bump within @p targetIterations; penalises exceeding @p maxIterations.
		double iterationsUntilBump(const std::string& fieldName, double targetIterations, double maxIterations, double tolerance);
		/// @brief Fitness score that rewards losing a bump within @p targetIterations; penalises exceeding @p maxIterations.
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
		double preShapednessAtPosition(const std::string& fieldName, double position ) const;
		double negativePreShapednessAtPosition(const std::string& fieldName, const double& position) const;
		double justOneBumpAtOneOfTheFollowingPositionsWithAmplitudeAndWidth(const std::string& fieldName,
		                                                                    const std::vector<double>& positions, const double& amplitude, const double& width) const;


		void moveGaussianStimulusContinuously(const std::string& name, double targetPosition, double step);
		double negativeBaseline(const std::string& fieldName) const;
	};
}
