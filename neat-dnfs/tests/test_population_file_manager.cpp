#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <ctime>
#include <array>
#include <unordered_set>

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
