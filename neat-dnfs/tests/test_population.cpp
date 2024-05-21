#include <catch2/catch_test_macros.hpp>

#include "neat/population.h"
#include "solutions/empty_solution.h"

TEST_CASE("Population Initialization", "[Population]")
{
    neat_dnfs::SolutionTopology topology(3, 1);
    const neat_dnfs::PopulationParameters params(10, 100, 0.95);
    const auto initialSolution = std::make_shared<neat_dnfs::EmptySolution>(topology);

    const neat_dnfs::Population population(params, initialSolution);
    population.initialize();

    REQUIRE(population.getSize() == 10);
    for (const auto& solution : population.getSolutions())
        REQUIRE(solution->getGenome().getFieldGenes().size() == 4); // 3 input + 1 output
}

TEST_CASE("Population Evaluation", "[Population]")
{
    neat_dnfs::SolutionTopology topology(3, 1);
    const neat_dnfs::PopulationParameters params(10, 100, 0.95);
    const auto initialSolution = std::make_shared<neat_dnfs::EmptySolution>(topology);

    const neat_dnfs::Population population(params, initialSolution);
    population.initialize();
    population.evaluate();

    for (const auto& solution : population.getSolutions()) 
    {
        REQUIRE(solution->getFitness() >= 0.0);
        REQUIRE(solution->getFitness() <= 1.0);
    }
}

TEST_CASE("Population Speciation", "[Population]")
{
    neat_dnfs::SolutionTopology topology(3, 1);
    const neat_dnfs::PopulationParameters params(10, 100, 0.95);
    const auto initialSolution = std::make_shared<neat_dnfs::EmptySolution>(topology);

    neat_dnfs::Population population(params, initialSolution);
    population.initialize();
    population.speciate();

    // Each solution should belong to some species
    for (const auto& solution : population.getSolutions()) 
    {
        bool found = false;
        for (const auto& species : population.getSpeciesList()) 
        {
            if (species.contains(solution)) 
            {
                found = true;
                break;
            }
        }
        REQUIRE(found);
    }
}

TEST_CASE("Population Selection", "[Population]")
{
    neat_dnfs::SolutionTopology topology(3, 1);
    const neat_dnfs::PopulationParameters params(10, 100, 0.95);
    const auto initialSolution = std::make_shared<neat_dnfs::EmptySolution>(topology);

    neat_dnfs::Population population(params, initialSolution);
    population.initialize();

    population.evaluate();
    population.speciate();
    population.select();
	REQUIRE(population.getSolutions().size() == params.size * neat_dnfs::PopulationConstants::killRatio);
}

TEST_CASE("Population Reproduction", "[Population]")
{
    constexpr static uint16_t numAttempts = 5;
    constexpr static uint16_t numGenerations = 100;

    for (uint16_t i = 0; i < numAttempts; ++i)
    {
        int generations = 0;
        const auto size = 
            static_cast<uint16_t>(neat_dnfs::tools::utils::generateRandomInt(3, 531));
        do
        {
            neat_dnfs::SolutionTopology topology(3, 1);
            const neat_dnfs::PopulationParameters params(size, numGenerations, 0.95);
            const auto initialSolution = std::make_shared<neat_dnfs::EmptySolution>(topology);

            neat_dnfs::Population population(params, initialSolution);
            population.initialize();
            population.evaluate();
            population.speciate();
            population.select();
            population.reproduce();
            population.upkeepBestSolution();

            // Check if the population size is correct after reproduction
            REQUIRE(population.getSolutions().size() == params.size);
            generations++;
        } while (generations < numGenerations);
    }
}

TEST_CASE("Population Best Solution Tracking", "[Population]")
{
    neat_dnfs::SolutionTopology topology(3, 1);
    const neat_dnfs::PopulationParameters params(10, 100, 0.95);
    const auto initialSolution = std::make_shared<neat_dnfs::EmptySolution>(topology);

    neat_dnfs::Population population(params, initialSolution);
    population.initialize();

    constexpr static uint16_t numAttempts = 100;

    double prevBestFitness = 0.0;
    for (uint16_t i = 0; i < numAttempts; ++i)
    {
        population.evaluate();
        population.speciate();
        population.select();
        population.reproduce();
        population.upkeepBestSolution();

		const auto bestSolution = population.getBestSolution();
		REQUIRE(bestSolution != nullptr);
        REQUIRE(bestSolution->getFitness() >= prevBestFitness);

		for (const auto& solution : population.getSolutions()) 
			REQUIRE(solution->getFitness() <= bestSolution->getFitness());
    	prevBestFitness = bestSolution->getFitness();
    }
}

TEST_CASE("Population End Condition", "[Population]")
{
    neat_dnfs::SolutionTopology topology(3, 1);
    const neat_dnfs::PopulationParameters params(10, 1, 0.5); // Shortened for testing
    const auto initialSolution = std::make_shared<neat_dnfs::EmptySolution>(topology);

    neat_dnfs::Population population(params, initialSolution);
    population.initialize();
    population.evolve();

    REQUIRE(population.getCurrentGeneration() <= params.numGenerations);
    const bool validEndCondition = population.getBestSolution()->getFitness() >= params.targetFitness ||
        		population.getCurrentGeneration() >= params.numGenerations;
    REQUIRE(validEndCondition);
}

TEST_CASE("Population Offspring Calculation", "[Population]")
{
    neat_dnfs::SolutionTopology topology(3, 1);
    const neat_dnfs::PopulationParameters params(10, 100, 0.95);
    const auto initialSolution = std::make_shared<neat_dnfs::EmptySolution>(topology);

    neat_dnfs::Population population(params, initialSolution);
    population.initialize();
    population.evaluate();
    population.speciate();
    population.calculateSpeciesOffspring();

    int totalOffspring = 0;
    for (const auto& species : population.getSpeciesList())
        totalOffspring += species.getOffspringCount();
    

    REQUIRE(totalOffspring == static_cast<int>(params.size * neat_dnfs::PopulationConstants::killRatio));
}

TEST_CASE("Population Kill Least Fit Solutions", "[Population]")
{
    neat_dnfs::SolutionTopology topology(3, 1);
    const neat_dnfs::PopulationParameters params(10, 100, 0.95);
    const auto initialSolution = std::make_shared<neat_dnfs::EmptySolution>(topology);

    neat_dnfs::Population population(params, initialSolution);
    population.initialize();
    population.evaluate();
    population.speciate();
    population.calculateSpeciesOffspring();
    population.killLeastFitSolutions();

    for (const auto& species : population.getSpeciesList())
        REQUIRE(species.size() <= species.getOffspringCount());
}
