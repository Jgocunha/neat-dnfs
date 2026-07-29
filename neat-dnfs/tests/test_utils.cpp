#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cmath>

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

TEST_CASE("normalize maps value to [0, 1] within range", "[normalize]")
{
    using neat_dnfs::tools::utils::normalize;

    REQUIRE(normalize(0.0, 0.0, 10.0) == 0.0);
    REQUIRE(normalize(10.0, 0.0, 10.0) == 1.0);
    REQUIRE(normalize(5.0, 0.0, 10.0) == Catch::Approx(0.5));
    REQUIRE(normalize(2.5, 0.0, 10.0) == Catch::Approx(0.25));
}

TEST_CASE("normalize clamps values outside [min, max]", "[normalize]")
{
    using neat_dnfs::tools::utils::normalize;

    REQUIRE(normalize(-5.0, 0.0, 10.0) == 0.0);
    REQUIRE(normalize(15.0, 0.0, 10.0) == 1.0);
}

TEST_CASE("normalize with min == max produces a non-finite result at the boundary", "[normalize]")
{
    // value == min == max skips both clamp branches and divides by zero (0/0 = NaN).
    // This documents the existing behaviour rather than asserting a "correct" one.
    using neat_dnfs::tools::utils::normalize;

    const double result = normalize(5.0, 5.0, 5.0);
    REQUIRE(std::isnan(result));
}

TEST_CASE("normalizeWithGaussian peaks at the target and decays away from it", "[normalizeWithGaussian]")
{
    using neat_dnfs::tools::utils::normalizeWithGaussian;

    REQUIRE(normalizeWithGaussian(50.0, 50.0, 10.0) == Catch::Approx(1.0));

    // width == 0: (value-target)/width is 0/0 = NaN at the target, and
    // exp(-0.5 * inf^2) = 0.0 anywhere off-target.
    REQUIRE(std::isnan(normalizeWithGaussian(50.0, 50.0, 0.0)));
    REQUIRE(normalizeWithGaussian(60.0, 50.0, 0.0) == Catch::Approx(0.0));

    const double near = normalizeWithGaussian(52.0, 50.0, 10.0);
    const double far = normalizeWithGaussian(500.0, 50.0, 10.0);
    REQUIRE(near < 1.0);
    REQUIRE(near > far);
    REQUIRE(far == Catch::Approx(0.0).margin(1e-6));
}

TEST_CASE("normalizeWithFlatheadGaussian is 1.0 inside the flat region", "[normalizeWithFlatheadGaussian]")
{
    using neat_dnfs::tools::utils::normalizeWithFlatheadGaussian;

    REQUIRE(normalizeWithFlatheadGaussian(0.0, 0.0, 10.0, 5.0) == Catch::Approx(1.0));
    REQUIRE(normalizeWithFlatheadGaussian(5.0, 0.0, 10.0, 5.0) == Catch::Approx(1.0));
    REQUIRE(normalizeWithFlatheadGaussian(10.0, 0.0, 10.0, 5.0) == Catch::Approx(1.0));
}

TEST_CASE("normalizeWithFlatheadGaussian decreases away from the flat region", "[normalizeWithFlatheadGaussian]")
{
    using neat_dnfs::tools::utils::normalizeWithFlatheadGaussian;

    const double atEdge = normalizeWithFlatheadGaussian(10.0, 0.0, 10.0, 5.0);
    const double beyond = normalizeWithFlatheadGaussian(40.0, 0.0, 10.0, 5.0);
    REQUIRE(atEdge == Catch::Approx(1.0));
    REQUIRE(beyond < atEdge);
    REQUIRE(beyond >= 0.0);
}
