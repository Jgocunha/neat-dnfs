#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "neat_tools/resource_paths.h"

using namespace neat_dnfs;

// resourceRoot() caches its answer in a function-local static and reads the
// environment, so it cannot be re-resolved from inside one test binary. The
// tests below therefore drive selectResourceRoot() -- the pure part of the
// resolution -- directly, and assert separately that the cached resolution this
// test binary itself got is the source-tree fallback (rule 4), which is what
// keeps a build tree behaving as it did before resource_paths.h existed.
namespace
{
    class TemporaryDirectory
    {
    public:
        explicit TemporaryDirectory(const std::string& name)
            : directory(std::filesystem::temp_directory_path() / ("neat-dnfs-" + name))
        {
            std::filesystem::remove_all(directory);
            std::filesystem::create_directories(directory);
        }
        ~TemporaryDirectory()
        {
            std::error_code ec;
            std::filesystem::remove_all(directory, ec);
        }
        TemporaryDirectory(const TemporaryDirectory&) = delete;
        TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

        const std::filesystem::path& path() const { return directory; }

        // The one file selectResourceRoot() looks for. Its contents are never
        // parsed here, only its presence.
        void writeReferenceConfig() const
        {
            std::filesystem::create_directories(directory / "config");
            std::ofstream(directory / "config" / "neat_dnfs.json") << "{}";
        }

    private:
        std::filesystem::path directory;
    };
}

TEST_CASE("selectResourceRoot returns the first candidate holding the reference config", "[ResourcePaths]")
{
    const TemporaryDirectory withoutConfig{ "select-without-config" };
    const TemporaryDirectory withConfig{ "select-with-config" };
    withConfig.writeReferenceConfig();

    // The empty path stands for a failed executable-location lookup, which the
    // resolver has to skip rather than treat as the current directory.
    const auto selected = paths::selectResourceRoot(
        { std::filesystem::path{}, withoutConfig.path(), withConfig.path() });

    REQUIRE(std::filesystem::equivalent(selected, withConfig.path()));
}

TEST_CASE("selectResourceRoot names every candidate it tried when none matches", "[ResourcePaths]")
{
    const TemporaryDirectory first{ "select-miss-first" };
    const TemporaryDirectory second{ "select-miss-second" };

    REQUIRE_THROWS_MATCHES(
        paths::selectResourceRoot({ first.path(), second.path() }),
        std::runtime_error,
        Catch::Matchers::MessageMatches(
            Catch::Matchers::ContainsSubstring(first.path().generic_string())
            && Catch::Matchers::ContainsSubstring(second.path().generic_string())
            && Catch::Matchers::ContainsSubstring("NEAT_DNFS_ROOT")));
}

TEST_CASE("a test binary in a build tree resolves back to the source tree", "[ResourcePaths]")
{
    REQUIRE(std::filesystem::exists(paths::resourceRoot() / "config" / "neat_dnfs.json"));
    REQUIRE(std::filesystem::equivalent(paths::resourceRoot(), PROJECT_DIR));

    // Results keep landing in the repo's data/ folder, as they did before.
    REQUIRE(std::filesystem::equivalent(paths::dataRoot(), paths::resourceRoot()));
}
