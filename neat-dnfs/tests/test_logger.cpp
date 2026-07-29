#include <catch2/catch_test_macros.hpp>

#include <thread>
#include <vector>
#include <sstream>

#include "neat_tools/logger.h"

// The race this guards against (concurrent writers to a shared global Logger
// object) is two aligned enum stores and will not fail under plain execution
// on MSVC. Its real red-green signal is CI's ThreadSanitizer job (Linux/macOS
// only) reporting a data race if the shared-global pattern is reintroduced.
// Locally this is a smoke test: it must not crash or throw.
TEST_CASE("logger::log does not mutate shared logger state", "[Logger]")
{
    using namespace neat_dnfs::tools::logger;

    // A level strictly above every level logged below, so nothing is printed
    // and ctest output stays clean.
    Logger::setMinLogLevel(static_cast<LogLevel>(FATAL + 1));

    constexpr int threadCount = 8;
    constexpr int messagesPerThread = 50;

    std::vector<std::thread> threads;
    threads.reserve(threadCount);
    for (int t = 0; t < threadCount; ++t)
    {
        threads.emplace_back([t]()
            {
                const LogLevel levels[] = { DEBUG, INFO, WARNING, ERROR, FATAL };
                for (int i = 0; i < messagesPerThread; ++i)
                {
                    log(levels[i % 5], "message from thread " + std::to_string(t), LogOutputMode::CONSOLE);
                }
            });
    }

    REQUIRE_NOTHROW([&]()
        {
            for (auto& thread : threads)
                thread.join();
        }());

    Logger::setMinLogLevel(LogLevel::DEBUG); // restore default
}

TEST_CASE("logger::log honours minLogLevel per call", "[Logger]")
{
    using namespace neat_dnfs::tools::logger;

    Logger::setMinLogLevel(LogLevel::WARNING);

    std::ostringstream captured;
    std::streambuf* const originalCoutBuffer = std::cout.rdbuf(captured.rdbuf());

    try
    {
        log(LogLevel::DEBUG, "should be filtered", LogOutputMode::CONSOLE);
        log(LogLevel::INFO, "should be filtered", LogOutputMode::CONSOLE);
        log(LogLevel::WARNING, "should be printed", LogOutputMode::CONSOLE);
        log(LogLevel::ERROR, "should be printed", LogOutputMode::CONSOLE);
    }
    catch (...)
    {
        std::cout.rdbuf(originalCoutBuffer);
        throw;
    }

    std::cout.rdbuf(originalCoutBuffer);

    const std::string output = captured.str();
    REQUIRE(output.find("should be filtered") == std::string::npos);
    REQUIRE(output.find("should be printed") != std::string::npos);

    Logger::setMinLogLevel(LogLevel::DEBUG); // restore default
}
