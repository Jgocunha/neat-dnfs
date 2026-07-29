#include <catch2/catch_test_macros.hpp>

#include "neat_tools/utils.h"

TEST_CASE("Random Integer Generation", "[generateRandomInt]")
{
    int min = 1;
    int max = 10;
    int result = neat_dnfs::tools::utils::generateRandomInt(min, max);

    REQUIRE(result >= min);
    REQUIRE(result <= max);

    min = 0;
    max = 100;
    result = neat_dnfs::tools::utils::generateRandomInt(min, max);

    REQUIRE(result >= min);
    REQUIRE(result <= max);

    min = -50;
    max = -1;
    result = neat_dnfs::tools::utils::generateRandomInt(min, max);

    REQUIRE(result >= min);
    REQUIRE(result <= max);

    constexpr uint16_t attempts = 1000;

    for (uint16_t i = 0; i < attempts; ++i)
	{
		min = 0;
		max = 1;
		result = neat_dnfs::tools::utils::generateRandomInt(min, max);

        const bool assertion = result == 0 || result == 1;
		REQUIRE(assertion);
	}
}

TEST_CASE("Random Double Generation", "[generateRandomDouble]")
{
    double min = 1.0;
    double max = 10.0;
    double result = neat_dnfs::tools::utils::generateRandomDouble(min, max);

    REQUIRE(result >= min);
    REQUIRE(result <= max);

    min = 0.0;
    max = 100.0;
    result = neat_dnfs::tools::utils::generateRandomDouble(min, max);

    REQUIRE(result >= min);
    REQUIRE(result <= max);

    min = -50.0;
    max = -1.0;
    result = neat_dnfs::tools::utils::generateRandomDouble(min, max);

    REQUIRE(result >= min);
    REQUIRE(result <= max);
}

TEST_CASE("Random Float Generation", "[generateRandomFloat]")
{
    float min = 1.0f;
    float max = 10.0f;
    float result = neat_dnfs::tools::utils::generateRandomFloat(min, max);

    REQUIRE(result >= min);
    REQUIRE(result <= max);

    min = 0.0f;
    max = 100.0f;
    result = neat_dnfs::tools::utils::generateRandomFloat(min, max);

    REQUIRE(result >= min);
    REQUIRE(result <= max);

    min = -50.0f;
    max = -1.0f;
    result = neat_dnfs::tools::utils::generateRandomFloat(min, max);

    REQUIRE(result >= min);
    REQUIRE(result <= max);
}

TEST_CASE("Random Signal Generation is unbiased", "[generateRandomSignal]")
{
    constexpr int attempts = 100000;
    int positiveCount = 0;

    for (int i = 0; i < attempts; ++i)
    {
        const int result = neat_dnfs::tools::utils::generateRandomSignal();
        REQUIRE((result == 1 || result == -1));
        if (result == 1)
            ++positiveCount;
    }

    // Expect ~50% +1, allow generous tolerance (~5 sigma) to avoid flakiness
    // while still failing decisively against the old ~33% biased implementation.
    const double proportion = static_cast<double>(positiveCount) / attempts;
    REQUIRE(proportion > 0.47);
    REQUIRE(proportion < 0.53);
}

TEST_CASE("Random generators do not degenerate to a constant stream", "[generateRandomInt]")
{
    constexpr int attempts = 1000;
    int first = neat_dnfs::tools::utils::generateRandomInt(0, 1000000);
    bool sawDifferentValue = false;

    for (int i = 0; i < attempts; ++i)
    {
        if (neat_dnfs::tools::utils::generateRandomInt(0, 1000000) != first)
        {
            sawDifferentValue = true;
            break;
        }
    }

    REQUIRE(sawDifferentValue);
}

TEST_CASE("Random Double Generation stays in range over many draws", "[generateRandomDouble]")
{
    constexpr int attempts = 10000;
    constexpr double min = -5.0;
    constexpr double max = 5.0;

    for (int i = 0; i < attempts; ++i)
    {
        const double result = neat_dnfs::tools::utils::generateRandomDouble(min, max);
        REQUIRE(result >= min);
        REQUIRE(result <= max);
    }
}
