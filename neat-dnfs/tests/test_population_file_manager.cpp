#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <ctime>
#include <array>
#include <unordered_set>

#include "neat/pareto.h"
#include "neat/population.h"
#include "neat_tools/resource_paths.h"
#include "test_helpers.h"
#include "test_stub_solution.h"

using namespace neat_dnfs;
using namespace neat_dnfs::test;

// PopulationFileManager writes to <data root>/data/<solutionName>/<timestamp>/ with no
// test-mode override (see population_file_manager.cpp setFileDirectory()). The helper
// below reads paths::dataRoot() itself rather than assuming PROJECT_DIR, so the test
// still tracks production if $NEAT_DNFS_DATA_DIR happens to be set in the environment.
// Every other test in this suite passes enableFileIO=false for exactly this reason. This
// is the one test that exercises real file IO, so it: (1) uses CountingSolution's
// distinctive name to keep the directory identifiable, (2) reproduces the same
// name+timestamp path the production code computes to locate what it wrote, and
// (3) deletes that directory afterward so the repo's data/ folder is not polluted.
namespace
{
    // FixedFitnessSolution hardcodes name = "FixedFitness", which the
    // run_metadata.json test above already claims. Both tests do real file IO
    // into <data root>/data/<name>/<timestamp>/, so under a parallel ctest run
    // two tests sharing a name can compute the same second-resolution timestamp
    // and race each other's writes and cleanup. Renaming this one keeps them
    // in separate directories.
    class JsonOverviewSolution final : public Solution
    {
    public:
        JsonOverviewSolution(const SolutionTopology& topology, const double fitness)
            : Solution(topology), fitnessToReport(fitness)
        {
            name = "FixedFitnessJsonOverview";
        }

        SolutionPtr clone() const override
        {
            return std::make_shared<JsonOverviewSolution>(initialTopology, fitnessToReport);
        }

        SolutionPtr copy() const override
        {
            return std::make_shared<JsonOverviewSolution>(initialTopology, fitnessToReport);
        }

    private:
        double fitnessToReport = 0.0;

        void testPhenotype() override
        {
            parameters.fitness = fitnessToReport;
        }

        void createPhenotypeEnvironment() override {}
    };

    // Reports four partials that differ between solutions, so a population
    // spreads over several Pareto fronts. clone() (initial population and
    // offspring) draws the next partials from a deterministic sequence; copy()
    // keeps them, so a preserved elite re-evaluates to the same fitness.
    class ObjectivesSolution final : public Solution
    {
    public:
        ObjectivesSolution(const SolutionTopology& topology, std::vector<double> partials, const std::string& solutionName)
            : Solution(topology), partialsToReport(std::move(partials))
        {
            name = solutionName;
            loadFitnessWeights("xor", 4);
        }

        SolutionPtr clone() const override
        {
            return std::make_shared<ObjectivesSolution>(initialTopology, nextPartials(), name);
        }

        SolutionPtr copy() const override
        {
            return std::make_shared<ObjectivesSolution>(initialTopology, partialsToReport, name);
        }

        static std::vector<double> nextPartials()
        {
            static int draw = 0;
            ++draw;
            std::vector<double> partials;
            for (int k = 0; k < 4; ++k)
            {
                partials.push_back(static_cast<double>((draw * (2 * k + 3) + k) % 11) / 10.0);
            }
            return partials;
        }

    private:
        std::vector<double> partialsToReport;

        void testPhenotype() override
        {
            parameters.partialFitness = partialsToReport;
            parameters.fitness = 0.25 * (partialsToReport[0] + partialsToReport[1] + partialsToReport[2] + partialsToReport[3]);
        }

        void createPhenotypeEnvironment() override {}
    };

    struct ScopedSelectionSettings
    {
        ScopedSelectionSettings(std::vector<std::vector<size_t>> groups, const double floor, const bool saveObjectives)
        {
            SelectionConstants::objectiveGroups = std::move(groups);
            SelectionConstants::feasibilityFloor = floor;
            PopulationConstants::saveObjectives = saveObjectives;
        }
        ~ScopedSelectionSettings()
        {
            SelectionConstants::reset();
            PopulationConstants::saveObjectives = true;
        }
        ScopedSelectionSettings(const ScopedSelectionSettings&) = delete;
        ScopedSelectionSettings& operator=(const ScopedSelectionSettings&) = delete;
        ScopedSelectionSettings(ScopedSelectionSettings&&) = delete;
        ScopedSelectionSettings& operator=(ScopedSelectionSettings&&) = delete;
    };

    std::vector<nlohmann::json> readJsonLines(const std::string& path)
    {
        std::ifstream file(path);
        std::vector<nlohmann::json> records;
        std::string line;
        while (std::getline(file, line))
        {
            if (!line.empty())
            {
                records.push_back(nlohmann::json::parse(line));
            }
        }
        return records;
    }

    RankedPoint rankedPointOf(const nlohmann::json& individual)
    {
        return { individual.at("objectives").get<std::vector<double>>(), individual.at("violation").get<double>() };
    }

    std::string expectedRunDirectory(const std::string& solutionName)
    {
        const auto now = std::time(nullptr);
        struct tm localTime{};
#ifdef _WIN32
        localtime_s(&localTime, &now);
#else
        localtime_r(&now, &localTime);
#endif
        std::array<char, 100> timeBuffer{};
        std::strftime(timeBuffer.data(), timeBuffer.size(), "%Y-%m-%d %Hh%Mm%Ss", &localTime);
        return (paths::dataRoot() / "data" / solutionName / timeBuffer.data()).generic_string() + "/";
    }

    // Directory-name collision avoidance for the run_metadata.json test below: rather than
    // reproducing setFileDirectory()'s own second-resolution timestamp (which could still
    // miss a directory created in the second between the pre- and post-evolve() snapshots),
    // this snapshots the solution's parent directory before evolve() and reports whichever
    // child directory is new afterward.
    std::string newlyCreatedRunDirectory(const std::string& solutionName,
        const std::unordered_set<std::string>& preExistingRunDirs)
    {
        const auto parentDirectory = paths::dataRoot() / "data" / solutionName;
        if (!std::filesystem::exists(parentDirectory))
        {
            return "";
        }
        for (const auto& entry : std::filesystem::directory_iterator(parentDirectory))
        {
            if (entry.is_directory() && !preExistingRunDirs.contains(entry.path().filename().string()))
            {
                return entry.path().generic_string() + "/";
            }
        }
        return "";
    }

    std::unordered_set<std::string> existingRunDirs(const std::string& solutionName)
    {
        std::unordered_set<std::string> dirs;
        const auto parentDirectory = paths::dataRoot() / "data" / solutionName;
        if (std::filesystem::exists(parentDirectory))
        {
            for (const auto& entry : std::filesystem::directory_iterator(parentDirectory))
            {
                if (entry.is_directory())
                {
                    dirs.insert(entry.path().filename().string());
                }
            }
        }
        return dirs;
    }
}

TEST_CASE("PopulationFileManager writes per-generation artifacts to disk", "[PopulationFileManager]")
{
    CountingSolution::live = 0;
    CountingSolution::peak = 0;

    const PopulationParameters parameters(5, 2, 1.1);
    const std::string solutionName = "Counting"; // CountingSolution sets this name itself
    const auto initialSolution = std::make_shared<CountingSolution>(makeTopology(1, 1));

    // File IO enabled (the 3rd ctor arg defaults to true) -- this is the point of the test.
    Population population(parameters, initialSolution);
    population.initialize();

    // setFileDirectory() (called inside evolve()) computes its own timestamp
    // independently of the test, so the directory is reconstructed on both sides
    // of the call to tolerate a wall-clock second boundary being crossed in between.
    const std::string directoryBeforeEvolve = expectedRunDirectory(solutionName);

    REQUIRE_NOTHROW(population.evolve());

    const std::string directoryAfterEvolve = expectedRunDirectory(solutionName);
    const std::string runDirectory =
        std::filesystem::exists(directoryBeforeEvolve) ? directoryBeforeEvolve : directoryAfterEvolve;

    REQUIRE(std::filesystem::exists(runDirectory));
    REQUIRE(std::filesystem::is_directory(runDirectory));

    bool wroteAtLeastOneFile = false;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(runDirectory))
    {
        if (entry.is_regular_file() && entry.file_size() > 0)
        {
            wroteAtLeastOneFile = true;
            break;
        }
    }
    REQUIRE(wroteAtLeastOneFile);

    // Remove only this run's own timestamped directory, not the whole shared
    // data/Counting/ root, so other/concurrent runs under that name are untouched.
    std::filesystem::remove_all(runDirectory);
}

TEST_CASE("PopulationFileManager writes run_metadata.json with build, dependency, machine and run-parameter facts", "[PopulationFileManager]")
{
    const PopulationParameters parameters(5, 2, 1.1);
    // A distinct solution name from the sibling test above (which uses CountingSolution /
    // "Counting"): both tests do real file IO into <data root>/data/<solutionName>/<timestamp>/,
    // and under a parallel ctest run two tests sharing a name could compute the same
    // second-resolution timestamp and race each other's writes/cleanup.
    const std::string solutionName = "FixedFitness";
    const auto initialSolution = std::make_shared<FixedFitnessSolution>(makeTopology(1, 1), 0.5);

    Population population(parameters, initialSolution);
    population.initialize();

    const auto preExisting = existingRunDirs(solutionName);

    REQUIRE_NOTHROW(population.evolve());

    const std::string runDirectory = newlyCreatedRunDirectory(solutionName, preExisting);
    REQUIRE(!runDirectory.empty());

    const std::string metadataPath = runDirectory + "run_metadata.json";
    REQUIRE(std::filesystem::exists(metadataPath));

    std::ifstream metadataFile(metadataPath);
    REQUIRE(metadataFile.is_open());
    nlohmann::json metadata;
    REQUIRE_NOTHROW(metadata = nlohmann::json::parse(metadataFile));
    metadataFile.close();

    REQUIRE(metadata.contains("build"));
    REQUIRE(metadata.contains("dependencies"));
    REQUIRE(metadata.contains("machine"));
    REQUIRE(metadata.contains("run_parameters"));
    REQUIRE(metadata["build"]["git_dirty"].is_boolean());

    std::filesystem::remove_all(runDirectory);
}

TEST_CASE("PopulationFileManager writes overview.jsonl as line-delimited JSON per generation", "[PopulationFileManager]")
{
    const PopulationParameters parameters(5, 2, 1.1);
    // A distinct solution name from the sibling tests above, for the same
    // parallel-test-directory-collision reason documented on the
    // run_metadata.json test.
    const std::string solutionName = "FixedFitnessJsonOverview";
    const auto initialSolution = std::make_shared<JsonOverviewSolution>(makeTopology(1, 1), 0.5);

    Population population(parameters, initialSolution);
    population.initialize();

    const auto preExisting = existingRunDirs(solutionName);

    REQUIRE_NOTHROW(population.evolve());

    const std::string runDirectory = newlyCreatedRunDirectory(solutionName, preExisting);
    REQUIRE(!runDirectory.empty());

    const std::string overviewPath = runDirectory + "overview.jsonl";
    REQUIRE(std::filesystem::exists(overviewPath));

    std::ifstream overviewFile(overviewPath);
    REQUIRE(overviewFile.is_open());

    std::vector<nlohmann::json> generationRecords;
    std::string line;
    while (std::getline(overviewFile, line))
    {
        if (line.empty())
        {
            continue;
        }
        nlohmann::json record;
        REQUIRE_NOTHROW(record = nlohmann::json::parse(line));
        generationRecords.push_back(record);
    }
    overviewFile.close();

    REQUIRE(!generationRecords.empty());

    for (const auto& record : generationRecords)
    {
        REQUIRE(record.contains("generation"));
        REQUIRE(record.contains("numberOfSolutions"));
        REQUIRE(record.contains("numberOfSpecies"));
        REQUIRE(record.contains("numberOfActiveSpecies"));
        REQUIRE(record.contains("hasFitnessImproved"));
        REQUIRE(record.contains("generationsWithoutImprovement"));
        REQUIRE(record.contains("averageFitness"));
        REQUIRE(record.contains("bestFitness"));
        REQUIRE(record.contains("innovationNumber"));
        REQUIRE(record.contains("averageGenomeSize"));
        REQUIRE(record.contains("averageConnectionGenes"));
        REQUIRE(record.contains("averageFieldGenes"));

        REQUIRE(record.contains("fitnessDistribution"));
        const auto& fitnessDistribution = record["fitnessDistribution"];
        REQUIRE(fitnessDistribution.contains("min"));
        REQUIRE(fitnessDistribution.contains("max"));
        REQUIRE(fitnessDistribution.contains("mean"));
        REQUIRE(fitnessDistribution.contains("median"));
        REQUIRE(fitnessDistribution.contains("stddev"));
        REQUIRE(fitnessDistribution.contains("q1"));
        REQUIRE(fitnessDistribution.contains("q3"));

        REQUIRE(record.contains("species"));
        REQUIRE(record["species"].is_array());
        for (const auto& species : record["species"])
        {
            REQUIRE(species.contains("id"));
            REQUIRE(species.contains("size"));
        }

        REQUIRE(record.contains("bestSolution"));
        const auto& bestSolution = record["bestSolution"];
        REQUIRE(bestSolution.contains("id"));
        REQUIRE(bestSolution.contains("fitness"));
        REQUIRE(bestSolution.contains("parentIds"));
        REQUIRE(bestSolution.contains("partialFitness"));
    }

    // per_generation_overview.txt must remain untouched by this feature.
    REQUIRE(std::filesystem::exists(runDirectory + "per_generation_overview.txt"));

    std::filesystem::remove_all(runDirectory);
}

TEST_CASE("PopulationFileManager writes objectives.jsonl with one ranked record per generation", "[PopulationFileManager]")
{
    const ScopedSelectionSettings settings({ { 0, 1 }, { 2, 3 } }, 0.2, true);
    constexpr size_t populationSize = 12;
    const PopulationParameters parameters(static_cast<int>(populationSize), 2, 1.1);
    const std::string solutionName = "FixedFitnessObjectives";
    const auto initialSolution = std::make_shared<ObjectivesSolution>(makeTopology(1, 1), ObjectivesSolution::nextPartials(), solutionName);

    Population population(parameters, initialSolution);
    population.initialize();
    const auto preExisting = existingRunDirs(solutionName);

    REQUIRE_NOTHROW(population.evolve());

    const std::string runDirectory = newlyCreatedRunDirectory(solutionName, preExisting);
    REQUIRE(!runDirectory.empty());
    const std::string objectivesPath = runDirectory + "objectives.jsonl";
    REQUIRE(std::filesystem::exists(objectivesPath));

    const auto records = readJsonLines(objectivesPath);
    REQUIRE(records.size() == 2);

    for (size_t generation = 0; generation < records.size(); ++generation)
    {
        const auto& record = records[generation];
        INFO("generation " << generation);
        REQUIRE(record.at("generation") == generation);
        REQUIRE(record.at("mode") == "scalar");
        REQUIRE(record.at("epsilon") == 0.0);
        REQUIRE(record.at("feasibilityFloor") == 0.2);
        REQUIRE(record.at("objectiveGroups") == nlohmann::json({ { 0, 1 }, { 2, 3 } }));
        REQUIRE(record.at("archive").at("size").is_number_unsigned());
        REQUIRE(record.at("archive").at("acceptedThisGeneration").is_array());

        const auto& individuals = record.at("individuals");
        REQUIRE(individuals.size() == populationSize);

        bool rankZeroPresent = false;
        for (const auto& individual : individuals)
        {
            REQUIRE(individual.contains("id"));
            REQUIRE(individual.contains("species"));
            REQUIRE(individual.contains("fitness"));
            REQUIRE(individual.at("partialFitness").size() == 4);
            REQUIRE(individual.at("objectives").size() == 2);
            REQUIRE(individual.at("rank").get<int>() >= 0);
            REQUIRE(individual.contains("crowding"));
            REQUIRE(individual.at("violation").get<double>() >= 0.0);
            rankZeroPresent = rankZeroPresent || individual.at("rank") == 0;
        }
        REQUIRE(rankZeroPresent);

        for (const auto& dominator : individuals)
        {
            for (const auto& dominated : individuals)
            {
                if (constrainedDominates(rankedPointOf(dominator), rankedPointOf(dominated), 0.0))
                {
                    REQUIRE(dominator.at("rank").get<int>() < dominated.at("rank").get<int>());
                }
            }
        }
    }

    std::filesystem::remove_all(runDirectory);
}

TEST_CASE("PopulationFileManager writes no objectives.jsonl when saveObjectives is off", "[PopulationFileManager]")
{
    const ScopedSelectionSettings settings({}, 0.0, false);
    const PopulationParameters parameters(5, 2, 1.1);
    // A name of its own, for the parallel-directory-collision reason
    // documented on the run_metadata.json test.
    const std::string solutionName = "FixedFitnessNoObjectives";
    const auto initialSolution = std::make_shared<ObjectivesSolution>(makeTopology(1, 1), ObjectivesSolution::nextPartials(), solutionName);

    Population population(parameters, initialSolution);
    population.initialize();
    const auto preExisting = existingRunDirs(solutionName);

    REQUIRE_NOTHROW(population.evolve());

    const std::string runDirectory = newlyCreatedRunDirectory(solutionName, preExisting);
    REQUIRE(!runDirectory.empty());
    REQUIRE(std::filesystem::exists(runDirectory + "overview.jsonl"));
    REQUIRE_FALSE(std::filesystem::exists(runDirectory + "objectives.jsonl"));

    std::filesystem::remove_all(runDirectory);
}
