#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <ctime>
#include <array>

#include "neat/population.h"
#include "test_helpers.h"
#include "test_stub_solution.h"

using namespace neat_dnfs;
using namespace neat_dnfs::test;

// PopulationFileManager writes to a hardcoded PROJECT_DIR/data/<solutionName>/<timestamp>/
// path with no test-mode override (see population_file_manager.cpp setFileDirectory()).
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
        return std::string(PROJECT_DIR) + "/data/" + solutionName + "/" + timeBuffer.data() + "/";
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
