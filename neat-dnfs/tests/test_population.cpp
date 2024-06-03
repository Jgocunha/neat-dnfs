#include <catch2/catch_test_macros.hpp>

#include "neat/population.h"
#include "solutions/empty_solution.h"

using namespace neat_dnfs;

//TEST_CASE("Population Initialization", "[Population]")
//{
//    neat_dnfs::SolutionTopology topology(3, 1);
//    const neat_dnfs::PopulationParameters params(10, 100, 0.95);
//    const auto initialSolution = std::make_shared<neat_dnfs::EmptySolution>(topology);
//
//    const neat_dnfs::Population population(params, initialSolution);
//    population.initialize();
//
//    REQUIRE(population.getSize() == 10);
//    for (const auto& solution : population.getSolutions())
//        REQUIRE(solution->getGenome().getFieldGenes().size() == 4); // 3 input + 1 output
//}
//
//TEST_CASE("Population Evaluation", "[Population]")
//{
//    neat_dnfs::SolutionTopology topology(3, 1);
//    const neat_dnfs::PopulationParameters params(10, 100, 0.95);
//    const auto initialSolution = std::make_shared<neat_dnfs::EmptySolution>(topology);
//
//    const neat_dnfs::Population population(params, initialSolution);
//    population.initialize();
//    population.evaluate();
//
//    for (const auto& solution : population.getSolutions()) 
//    {
//        REQUIRE(solution->getFitness() >= 0.0);
//        REQUIRE(solution->getFitness() <= 1.0);
//    }
//}
//
//TEST_CASE("Population Speciation", "[Population]")
//{
//    neat_dnfs::SolutionTopology topology(3, 1);
//    const neat_dnfs::PopulationParameters params(10, 100, 0.95);
//    const auto initialSolution = std::make_shared<neat_dnfs::EmptySolution>(topology);
//
//    neat_dnfs::Population population(params, initialSolution);
//    population.initialize();
//    population.speciate();
//
//    // Each solution should belong to some species
//    for (const auto& solution : population.getSolutions()) 
//    {
//        bool found = false;
//        for (const auto& species : population.getSpeciesList()) 
//        {
//            if (species.contains(solution)) 
//            {
//                found = true;
//                break;
//            }
//        }
//        REQUIRE(found);
//    }
//}
//
//TEST_CASE("Population Selection", "[Population]")
//{
//    neat_dnfs::SolutionTopology topology(3, 1);
//    const neat_dnfs::PopulationParameters params(10, 100, 0.95);
//    const auto initialSolution = std::make_shared<neat_dnfs::EmptySolution>(topology);
//
//    neat_dnfs::Population population(params, initialSolution);
//    population.initialize();
//
//    population.evaluate();
//    population.speciate();
//    population.select();
//	REQUIRE(population.getSolutions().size() == params.size * neat_dnfs::PopulationConstants::killRatio);
//}
//
//TEST_CASE("Population Reproduction", "[Population]")
//{
//    constexpr static uint16_t numAttempts = 5;
//    constexpr static uint16_t numGenerations = 10;
//
//    for (uint16_t i = 0; i < numAttempts; ++i)
//    {
//        int generations = 0;
//        const auto size = 
//            static_cast<uint16_t>(neat_dnfs::tools::utils::generateRandomInt(3, 55));
//        do
//        {
//            neat_dnfs::SolutionTopology topology(3, 1);
//            const neat_dnfs::PopulationParameters params(size, numGenerations, 0.95);
//            const auto initialSolution = std::make_shared<neat_dnfs::EmptySolution>(topology);
//
//            neat_dnfs::Population population(params, initialSolution);
//            population.initialize();
//            population.evaluate();
//            population.speciate();
//            population.select();
//            population.reproduce();
//            population.upkeepBestSolution();
//
//            // Check if the population size is correct after reproduction
//            REQUIRE(population.getSolutions().size() == params.size);
//            generations++;
//        } while (generations < numGenerations);
//    }
//}
//
//TEST_CASE("Population Best Solution Tracking", "[Population]")
//{
//    neat_dnfs::SolutionTopology topology(3, 1);
//    const neat_dnfs::PopulationParameters params(10, 100, 0.95);
//    const auto initialSolution = std::make_shared<neat_dnfs::EmptySolution>(topology);
//
//    neat_dnfs::Population population(params, initialSolution);
//    population.initialize();
//
//    constexpr static uint16_t numAttempts = 10;
//
//    double prevBestFitness = 0.0;
//    for (uint16_t i = 0; i < numAttempts; ++i)
//    {
//        population.evaluate();
//        population.speciate();
//        population.select();
//        population.reproduce();
//        population.upkeepBestSolution();
//
//		const auto bestSolution = population.getBestSolution();
//		REQUIRE(bestSolution != nullptr);
//        REQUIRE(bestSolution->getFitness() >= prevBestFitness);
//
//		for (const auto& solution : population.getSolutions()) 
//			REQUIRE(solution->getFitness() <= bestSolution->getFitness());
//    	prevBestFitness = bestSolution->getFitness();
//    }
//}
//
//TEST_CASE("Population End Condition", "[Population]")
//{
//    neat_dnfs::SolutionTopology topology(3, 1);
//    const neat_dnfs::PopulationParameters params(10, 1, 0.5); // Shortened for testing
//    const auto initialSolution = std::make_shared<neat_dnfs::EmptySolution>(topology);
//
//    neat_dnfs::Population population(params, initialSolution);
//    population.initialize();
//    population.evolve();
//
//    REQUIRE(population.getCurrentGeneration() <= params.numGenerations);
//    const bool validEndCondition = population.getBestSolution()->getFitness() >= params.targetFitness ||
//        		population.getCurrentGeneration() >= params.numGenerations;
//    REQUIRE(validEndCondition);
//}
//
//TEST_CASE("Population Offspring Calculation", "[Population]")
//{
//    neat_dnfs::SolutionTopology topology(3, 1);
//    const neat_dnfs::PopulationParameters params(10, 100, 0.95);
//    const auto initialSolution = std::make_shared<neat_dnfs::EmptySolution>(topology);
//
//    neat_dnfs::Population population(params, initialSolution);
//    population.initialize();
//    population.evaluate();
//    population.speciate();
//    population.select();
//    population.reproduce();
//
//    int totalOffspring = 0;
//    for (const auto& species : population.getSpeciesList())
//        totalOffspring += species.getOffspringCount();
//
//    REQUIRE(totalOffspring == static_cast<int>(params.size * neat_dnfs::PopulationConstants::killRatio));
//}
//
//TEST_CASE("Population Kill Least Fit Solutions", "[Population]")
//{
//    neat_dnfs::SolutionTopology topology(3, 1);
//    const neat_dnfs::PopulationParameters params(10, 100, 0.95);
//    const auto initialSolution = std::make_shared<neat_dnfs::EmptySolution>(topology);
//
//    neat_dnfs::Population population(params, initialSolution);
//    population.initialize();
//    population.evaluate();
//    population.speciate();
//    population.select();
//    population.reproduce();
//
//    for (const auto& species : population.getSpeciesList())
//        REQUIRE(species.size() <= species.getOffspringCount());
//}


//TEST_CASE("Population Initialization", "[Population]")
//{
//    neat_dnfs::SolutionTopology topology(3, 1);
//    const neat_dnfs::PopulationParameters params(10, 100, 0.95);
//    const auto initialSolution = std::make_shared<neat_dnfs::EmptySolution>(topology);
//
//    const neat_dnfs::Population population(params, initialSolution);
//    population.initialize();
//
//    REQUIRE(population.getSize() == 10);
//    for (const auto& solution : population.getSolutions())
//        REQUIRE(solution->getGenome().getFieldGenes().size() == 4); // 3 input + 1 output
//    // All solutions should be unique
//	for (size_t i = 0; i < population.getSolutions().size(); i++)
//	    for (size_t j = i + 1; j < population.getSolutions().size(); j++)
//	    	REQUIRE(population.getSolutions()[i] != population.getSolutions()[j]);
//}
//
//TEST_CASE("Population Evaluation", "[Population]")
//{
//    neat_dnfs::SolutionTopology topology(3, 1);
//    const neat_dnfs::PopulationParameters params(10, 100, 0.95);
//    const auto initialSolution = std::make_shared<neat_dnfs::EmptySolution>(topology);
//
//    const neat_dnfs::Population population(params, initialSolution);
//    population.initialize();
//    population.evaluate();
//
//    for (const auto& solution : population.getSolutions()) 
//    {
//        REQUIRE(solution->getFitness() >= 0.0);
//        REQUIRE(solution->getFitness() <= 1.0);
//    }
//    REQUIRE(population.getSize() == 10);
//    for (size_t i = 0; i < population.getSolutions().size(); i++)
//        for (size_t j = i + 1; j < population.getSolutions().size(); j++)
//            REQUIRE(population.getSolutions()[i] != population.getSolutions()[j]);
//}
//
//TEST_CASE("Population Speciation", "[Population]")
//{
//    neat_dnfs::SolutionTopology topology(3, 1);
//    const neat_dnfs::PopulationParameters params(10, 100, 0.95);
//    const auto initialSolution = std::make_shared<neat_dnfs::EmptySolution>(topology);
//
//    neat_dnfs::Population population(params, initialSolution);
//    population.initialize();
//    population.evaluate();
//    population.speciate();
//
//    // Each solution should belong to some species
//    for (const auto& solution : population.getSolutions()) 
//    {
//        bool found = false;
//        for (const auto& species : population.getSpeciesList()) 
//        {
//            if (species.contains(solution)) 
//            {
//                found = true;
//                break;
//            }
//        }
//        REQUIRE(found);
//    }
//    REQUIRE(population.getSize() == 10);
//    for (size_t i = 0; i < population.getSolutions().size(); i++)
//        for (size_t j = i + 1; j < population.getSolutions().size(); j++)
//            REQUIRE(population.getSolutions()[i] != population.getSolutions()[j]);
//}
//
//TEST_CASE("Population Evolutionary Run", "[Population]")
//{
//    neat_dnfs::SolutionTopology topology(1, 1);
//    const neat_dnfs::PopulationParameters params(10, 1000, 1.0);
//    const auto initialSolution = std::make_shared<neat_dnfs::EmptySolution>(topology);
//
//    neat_dnfs::Population population(params, initialSolution);
//    population.initialize();
//
//    constexpr static uint16_t numAttempts = 100;
//
//    double prevBestFitness = 0.0;
//    for (uint16_t i = 0; i < numAttempts; ++i)
//    {
//        population.evaluate();
//        population.speciate();
//        population.reproduceAndSelect();
//        population.upkeep();
//
//        const auto bestSolution = population.getBestSolution();
//        REQUIRE(bestSolution != nullptr);
//        REQUIRE(bestSolution->getFitness() >= prevBestFitness);
//
//        for (const auto& solution : population.getSolutions())
//            REQUIRE(solution->getFitness() <= bestSolution->getFitness());
//        prevBestFitness = bestSolution->getFitness();
//    }
//}


TEST_CASE("Population::initialize", "[Population]")
{
    const PopulationParameters parameters(31, 50, 0.9);
    const auto initialSolution = std::make_shared<EmptySolution>(SolutionTopology(3, 1));

    const Population population(parameters, initialSolution);

    SECTION("Initialize population")
	{
        REQUIRE_NOTHROW(population.initialize());
        const auto solutions = population.getSolutions();
        REQUIRE(solutions.size() == parameters.size);
        for (const auto& solution : solutions)
            REQUIRE(!solution->getGenome().getFieldGenes().empty());
    }
}

TEST_CASE("Population::evolve", "[Population]")
{
    const PopulationParameters parameters(65, 50, 0.9);
    const auto initialSolution = std::make_shared<EmptySolution>(SolutionTopology(3, 1));

    Population population(parameters, initialSolution);
    population.initialize();

    SECTION("Evolve population")
	{
        REQUIRE_NOTHROW(population.evolve());
        REQUIRE(population.getCurrentGeneration() <= parameters.numGenerations);
        const bool validEndCondition = population.getBestSolution()->getFitness() >= parameters.targetFitness ||
			population.getCurrentGeneration() >= parameters.numGenerations;
        REQUIRE(validEndCondition);
    }
}

TEST_CASE("Population::evaluate", "[Population]")
{
    const PopulationParameters parameters(11, 50, 0.9);
    const auto initialSolution = std::make_shared<EmptySolution>(SolutionTopology(3, 1));

    const Population population(parameters, initialSolution);
    population.initialize();

    SECTION("Evaluate population")
	{
        REQUIRE_NOTHROW(population.evaluate());
        const auto solutions = population.getSolutions();
        for (const auto& solution : solutions)
            REQUIRE(solution->getFitness() > 0.0);
    }
}

TEST_CASE("Population::speciate", "[Population]")
{
    const PopulationParameters parameters(81, 50, 0.9);
    const auto initialSolution = std::make_shared<EmptySolution>(SolutionTopology(3, 1));

    Population population(parameters, initialSolution);
    population.initialize();
    population.evaluate();

    SECTION("Speciate population")
	{
        REQUIRE_NOTHROW(population.speciate());
        const auto speciesList = population.getSpeciesList();
        REQUIRE(!speciesList.empty());
        for (const auto& species : speciesList)
            REQUIRE(species.getMembers().size() == 81);
    }
}

TEST_CASE("Population::reproduceAndSelect", "[Population]")
{
    const PopulationParameters parameters(13, 50, 0.9);
    const auto initialSolution = std::make_shared<EmptySolution>(SolutionTopology(3, 1));

    Population population(parameters, initialSolution);
    population.initialize();
    population.evaluate();
    population.speciate();

    SECTION("Reproduce and select population")
	{
        REQUIRE_NOTHROW(population.reproduceAndSelect());
        const auto solutions = population.getSolutions();
        REQUIRE(solutions.size() == parameters.size);
    }
}

TEST_CASE("Population::upkeep", "[Population]")
{
    const PopulationParameters parameters(91, 50, 0.9);
    const auto initialSolution = std::make_shared<EmptySolution>(SolutionTopology(3, 1));

    Population population(parameters, initialSolution);
    population.initialize();
    population.evaluate();
    population.speciate();
    population.reproduceAndSelect();

    SECTION("Upkeep population")
	{
        REQUIRE_NOTHROW(population.upkeep());
        REQUIRE(population.getCurrentGeneration() == 1);
        REQUIRE(population.getBestSolution()->getFitness() > 0.0);
    }
}

TEST_CASE("Population::endConditionMet", "[Population]")
{
    const PopulationParameters parameters(10, 50, 0.9);
    const auto initialSolution = std::make_shared<EmptySolution>(SolutionTopology(3, 1));

    Population population(parameters, initialSolution);
    population.initialize();
    population.evaluate();

    SECTION("End condition not met")
	{
        REQUIRE_FALSE(population.endConditionMet());
    }

    SECTION("End condition met by fitness")
	{
        const auto bestSolution = population.getBestSolution();
        do
        {
	        bestSolution->evaluate();
        } while (bestSolution->getFitness() < 0.9);
        REQUIRE(population.endConditionMet());
    }

    SECTION("End condition met by generation")
	{
        for (uint16_t i = 0; i <= parameters.numGenerations; ++i)
            population.upkeep();
        REQUIRE(population.endConditionMet());
    }
}

//TEST_CASE("Population::selectElites", "[Population]")
//{
//    const PopulationParameters parameters(10, 50, 0.9);
//    const auto initialSolution = std::make_shared<EmptySolution>(SolutionTopology(3, 1));
//
//    Population population(parameters, initialSolution);
//    population.initialize();
//    population.evaluate();
//    population.speciate();
//
//    SECTION("Select elites")
//	{
//        const auto elites = population.selectElites();
//        REQUIRE(!elites.empty());
//        for (const auto& elite : elites) 
//        {
//            REQUIRE(elite->getFitness() > 0.0);
//        }
//    }
//}
//
//TEST_CASE("Population::selectLessFit", "[Population]")
//{
//    const PopulationParameters parameters(10, 50, 0.9);
//    const auto initialSolution = std::make_shared<EmptySolution>(SolutionTopology(3, 1));
//
//    Population population(parameters, initialSolution);
//    population.initialize();
//    population.evaluate();
//    population.speciate();
//
//    SECTION("Select less fit solutions")
//	{
//        const auto lessFit = population.selectLessFit();
//        REQUIRE(!lessFit.empty());
//        for (const auto& solution : lessFit) {
//            REQUIRE(solution->getFitness() > 0.0);
//        }
//    }
//}
//
//TEST_CASE("Population::reproduce", "[Population]")
//{
//    const PopulationParameters parameters(10, 50, 0.9);
//    const auto initialSolution = std::make_shared<EmptySolution>(SolutionTopology(3, 1));
//
//    Population population(parameters, initialSolution);
//    population.initialize();
//    population.evaluate();
//    population.speciate();
//    population.reproduceAndSelect();
//
//    SECTION("Reproduce population")
//	{
//        const auto offspring = population.reproduce();
//        REQUIRE(!offspring.empty());
//        for (const auto& solution : offspring)
//            REQUIRE(solution->getGenomeSize() > 0);
//    }
//}

//TEST_CASE("Population::validateElitism", "[Population]") {
//    PopulationParameters parameters(10, 50, 0.9);
//    auto initialSolution = std::make_shared<EmptySolution>(SolutionTopology(3, 1));
//
//    Population population(parameters, initialSolution);
//    population.initialize();
//    population.evaluate();
//
//    SECTION("Validate elitism without error") {
//        population.upkeep();
//        REQUIRE_NOTHROW(population.validateElitism());
//    }
//
//    SECTION("Validate elitism with error") {
//        auto bestSolution = population.getBestSolution();
//        bestSolution->setAdjustedFitness(bestSolution->getFitness() - 1.0);  // Force an elitism failure
//        REQUIRE_THROWS_AS(population.validateElitism(), std::runtime_error);
//    }
//}
//
//TEST_CASE("Population::validateUniqueSolutions", "[Population]") {
//    PopulationParameters parameters(10, 50, 0.9);
//    auto initialSolution = std::make_shared<EmptySolution>(SolutionTopology(3, 1));
//
//    Population population(parameters, initialSolution);
//    population.initialize();
//    population.evaluate();
//
//    SECTION("Validate unique solutions without error") {
//        REQUIRE_NOTHROW(population.validateUniqueSolutions());
//    }
//
//    SECTION("Validate unique solutions with duplicates") {
//        auto solutions = population.getSolutions();
//        solutions.push_back(solutions[0]);  // Force a duplicate
//        REQUIRE_THROWS_AS(population.validateUniqueSolutions(), std::runtime_error);
//    }
//}
//
//TEST_CASE("Population::validatePopulationSize", "[Population]") {
//    PopulationParameters parameters(10, 50, 0.9);
//    auto initialSolution = std::make_shared<EmptySolution>(SolutionTopology(3, 1));
//
//    Population population(parameters, initialSolution);
//    population.initialize();
//    population.evaluate();
//
//    SECTION("Validate population size without error") {
//        REQUIRE_NOTHROW(population.validatePopulationSize());
//    }
//
//    SECTION("Validate population size with error") {
//        auto solutions = population.getSolutions();
//        solutions.pop_back();  // Force an incorrect size
//        REQUIRE_THROWS_AS(population.validatePopulationSize(), std::runtime_error);
//    }
//}