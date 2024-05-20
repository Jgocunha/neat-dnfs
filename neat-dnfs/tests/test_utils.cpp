#include <catch2/catch_test_macros.hpp>

#include "tools/utils.h"

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
