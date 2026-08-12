#include <catch2/catch_test_macros.hpp>

#include "neat/population.h"

namespace
{
	using namespace neat_dnfs;

	// A Solution stand-in whose fitness/partial fitness are fixed by the test,
	// independent of genome/mutation, so Population::endConditionMet() can be
	// exercised deterministically through evolve() without running real DNF
	// simulations.
	class ControllableSolution final : public Solution
	{
	public:
		static std::vector<double> partialFitnessToReturn;

		explicit ControllableSolution(const SolutionTopology& topology)
			: Solution(topology)
		{
			name = "ControllableSolution";
		}

		ControllableSolution(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype)
			: Solution(initialTopology, phenotype)
		{
			name = "ControllableSolution";
		}

		SolutionPtr clone() const override
		{
			ControllableSolution solution(initialTopology);
			return std::make_shared<ControllableSolution>(solution);
		}

		SolutionPtr copy() const override
		{
			ControllableSolution solution(initialTopology, phenotype);
			return std::make_shared<ControllableSolution>(solution);
		}

	private:
		void testPhenotype() override
		{
			parameters.partialFitness = partialFitnessToReturn;
			double sum = 0.0;
			for (const double f : parameters.partialFitness)
				sum += f;
			parameters.fitness = parameters.partialFitness.empty()
				? 0.0 : sum / static_cast<double>(parameters.partialFitness.size());
		}

		void createPhenotypeEnvironment() override {}
	};

	std::vector<double> ControllableSolution::partialFitnessToReturn = {};

	SolutionTopology minimalTopology()
	{
		using namespace dnf_composer::element;
		const ElementDimensions dims{ DimensionConstants::xSize, DimensionConstants::dx };
		return SolutionTopology{ {
			{FieldGeneType::INPUT, dims},
			{FieldGeneType::OUTPUT, dims},
		} };
	}
}

TEST_CASE("Population end condition requires every partial fitness above target", "[Population][endCondition]")
{
	// Population size must exceed 5 (per species) for NEAT elitism
	// (species->size() > 5) to carry an already-evaluated champion into the
	// next generation's solution list; below that, every solution is a freshly
	// bred, not-yet-evaluated offspring when endConditionMet() runs, so the
	// fitness-based stop can never trigger regardless of target.
	SECTION("Overall fitness above target but one partial below target keeps evolving to the generation limit")
	{
		// average = 0.76 > target (0.7), but the third partial fitness (0.3) is not.
		ControllableSolution::partialFitnessToReturn = { 0.99, 0.99, 0.3 };
		const auto initialSolution = std::make_shared<ControllableSolution>(minimalTopology());
		const PopulationParameters parameters(10, 3, 0.7);
		Population population(parameters, initialSolution);
		population.initialize();

		REQUIRE_NOTHROW(population.evolve());

		REQUIRE(population.getCurrentGeneration() == parameters.numGenerations);
	}

	SECTION("All partial fitnesses above target stops evolution before the generation limit")
	{
		ControllableSolution::partialFitnessToReturn = { 0.95, 0.95, 0.95 };
		const auto initialSolution = std::make_shared<ControllableSolution>(minimalTopology());
		const PopulationParameters parameters(10, 50, 0.9);
		Population population(parameters, initialSolution);
		population.initialize();

		REQUIRE_NOTHROW(population.evolve());

		REQUIRE(population.getCurrentGeneration() < parameters.numGenerations);
		for (const double partial : population.getBestSolution()->getParameters().partialFitness)
			REQUIRE(partial > parameters.targetFitness);
	}
}
