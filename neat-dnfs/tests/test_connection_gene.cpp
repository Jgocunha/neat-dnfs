#include <catch2/catch_test_macros.hpp>

#include "neat/connection_gene.h"

using namespace neat_dnfs;
using namespace dnf_composer::element;

TEST_CASE("ConnectionGene Initialization", "[ConnectionGene]")
{
    SECTION("Initialize ConnectionGene with ConnectionTuple")
	{
        ConnectionTuple connectionTuple(1, 2);
        ConnectionGene connectionGene(connectionTuple);

        REQUIRE(connectionGene.getInFieldGeneId() == 1);
        REQUIRE(connectionGene.getOutFieldGeneId() == 2);
        REQUIRE(connectionGene.isEnabled() == true);
        REQUIRE(connectionGene.getKernel() != nullptr);

        auto kernel = std::dynamic_pointer_cast<GaussKernel>(connectionGene.getKernel());
        REQUIRE(kernel != nullptr);
        REQUIRE(kernel->getParameters().sigma >= GaussKernelConstants::initialSigmaMin);
        REQUIRE(kernel->getParameters().sigma <= GaussKernelConstants::initialSigmaMax);
        REQUIRE(kernel->getParameters().amplitude >= GaussKernelConstants::initialAmplitudeMin);
        REQUIRE(kernel->getParameters().amplitude <= GaussKernelConstants::initialAmplitudeMax);
    }

    SECTION("Initialize ConnectionGene with ConnectionTuple and GaussKernelParameters")
	{
        ConnectionTuple connectionTuple(3, 4);
        GaussKernelParameters gkp{ 5.0, 3.0, false, false };
        ConnectionGene connectionGene(connectionTuple, gkp);

        REQUIRE(connectionGene.getInFieldGeneId() == 3);
        REQUIRE(connectionGene.getOutFieldGeneId() == 4);
        REQUIRE(connectionGene.isEnabled() == true);
        REQUIRE(connectionGene.getKernel() != nullptr);

        auto kernel = std::dynamic_pointer_cast<GaussKernel>(connectionGene.getKernel());
        REQUIRE(kernel != nullptr);
        REQUIRE(kernel->getParameters().sigma == 5.0);
        REQUIRE(kernel->getParameters().amplitude == 3.0);
    }
}

TEST_CASE("ConnectionGene Mutation", "[ConnectionGene]")
{
    const ConnectionTuple connectionTuple(1, 2);
    const ConnectionGene connectionGene(connectionTuple);

    const auto initialKernel = std::dynamic_pointer_cast<GaussKernel>(connectionGene.getKernel());
    const auto initialParams = initialKernel->getParameters();

    connectionGene.mutate();

    const auto mutatedKernel = std::dynamic_pointer_cast<GaussKernel>(connectionGene.getKernel());
    const auto mutatedParams = mutatedKernel->getParameters();

    const bool sigmaChanged = initialParams.sigma != mutatedParams.sigma;
    const bool amplitudeChanged = initialParams.amplitude != mutatedParams.amplitude;

    REQUIRE((sigmaChanged != amplitudeChanged)); // Only one should change
}

TEST_CASE("ConnectionGene Mutation Constraints", "[ConnectionGene]")
{
    ConnectionTuple connectionTuple(1, 2);
    ConnectionGene connectionGene(connectionTuple);

    auto initialKernel = std::dynamic_pointer_cast<GaussKernel>(connectionGene.getKernel());
    auto initialParams = initialKernel->getParameters();

    // Mutate multiple times to test constraints
    for (int i = 0; i < 100; ++i)
        connectionGene.mutate();

    auto mutatedKernel = std::dynamic_pointer_cast<GaussKernel>(connectionGene.getKernel());
    auto mutatedParams = mutatedKernel->getParameters();

    REQUIRE(mutatedParams.sigma >= MutationConstants::minSigma);
    REQUIRE(mutatedParams.sigma <= MutationConstants::maxSigma);
    REQUIRE(mutatedParams.amplitude >= MutationConstants::minAmplitude);
    REQUIRE(mutatedParams.amplitude <= MutationConstants::maxAmplitude);
}

TEST_CASE("ConnectionGene Disable and Toggle", "[ConnectionGene]")
{
    const ConnectionTuple connectionTuple(1, 2);
    ConnectionGene connectionGene(connectionTuple);

    REQUIRE(connectionGene.isEnabled() == true);

    connectionGene.disable();
    REQUIRE(connectionGene.isEnabled() == false);

    connectionGene.toggle();
    REQUIRE(connectionGene.isEnabled() == true);

    connectionGene.toggle();
    REQUIRE(connectionGene.isEnabled() == false);
}

TEST_CASE("ConnectionGene Set Innovation Number", "[ConnectionGene]")
{
    const ConnectionTuple connectionTuple(1, 2);
    ConnectionGene connectionGene(connectionTuple);

    connectionGene.setInnovationNumber(42);
    REQUIRE(connectionGene.getInnovationNumber() == 42);
}

TEST_CASE("ConnectionGene Comparison Operator", "[ConnectionGene]")
{
    const ConnectionTuple connectionTuple1(1, 2);
    const ConnectionTuple connectionTuple2(3, 4);

    ConnectionGene connectionGene1(connectionTuple1);
    ConnectionGene connectionGene2(connectionTuple2);

    REQUIRE(connectionGene1 == connectionGene1);
    REQUIRE(!(connectionGene1 == connectionGene2));

    connectionGene2.setInnovationNumber(connectionGene1.getInnovationNumber());
    REQUIRE(connectionGene1 == connectionGene2);
}

TEST_CASE("ConnectionGene Kernel Parameters Access", "[ConnectionGene]")
{
    const ConnectionTuple connectionTuple(1, 2);
    const GaussKernelParameters gkp{ 5.0, 3.0, false, false };
    const ConnectionGene connectionGene(connectionTuple, gkp);

    REQUIRE(connectionGene.getKernelWidth() == 5.0);
    REQUIRE(connectionGene.getKernelAmplitude() == 3.0);
}

TEST_CASE("ConnectionGene Initialization with Edge Values", "[ConnectionGene]")
{
    const ConnectionTuple connectionTuple(0, 0);
    const ConnectionGene connectionGene(connectionTuple);

    REQUIRE(connectionGene.getInFieldGeneId() == 0);
    REQUIRE(connectionGene.getOutFieldGeneId() == 0);
    REQUIRE(connectionGene.isEnabled() == true);
    REQUIRE(connectionGene.getKernel() != nullptr);
}

TEST_CASE("ConnectionGene Set Max Innovation Number", "[ConnectionGene]")
{
    const ConnectionTuple connectionTuple(1, 2);
    ConnectionGene connectionGene(connectionTuple);

    connectionGene.setInnovationNumber(std::numeric_limits<uint16_t>::max());
    REQUIRE(connectionGene.getInnovationNumber() == std::numeric_limits<uint16_t>::max());
}

TEST_CASE("ConnectionGene Multiple Mutations Consistency", "[ConnectionGene]")
{
    const ConnectionTuple connectionTuple(1, 2);
    const ConnectionGene connectionGene(connectionTuple);

    for (int i = 0; i < 1000; ++i)
        connectionGene.mutate();

    const auto mutatedKernel = std::dynamic_pointer_cast<GaussKernel>(connectionGene.getKernel());
    const auto mutatedParams = mutatedKernel->getParameters();

    REQUIRE(mutatedParams.sigma >= MutationConstants::minSigma);
    REQUIRE(mutatedParams.sigma <= MutationConstants::maxSigma);
    REQUIRE(mutatedParams.amplitude >= MutationConstants::minAmplitude);
    REQUIRE(mutatedParams.amplitude <= MutationConstants::maxAmplitude);
}
