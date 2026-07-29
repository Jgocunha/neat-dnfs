#include <catch2/catch_test_macros.hpp>

#include "neat/connection_gene.h"

using namespace neat_dnfs;
using namespace dnf_composer::element;

TEST_CASE("ConnectionGene Initialization", "[ConnectionGene]")
{
    SECTION("Initialize ConnectionGene with ConnectionTuple")
    {
        ConnectionTuple connectionTuple(1, 2);
        ConnectionGene connectionGene(connectionTuple, 0);

        REQUIRE(connectionGene.getInFieldGeneId() == 1);
        REQUIRE(connectionGene.getOutFieldGeneId() == 2);
        REQUIRE(connectionGene.isEnabled() == true);
        REQUIRE(connectionGene.getKernel() != nullptr);

        // Kernel type is probabilistic (80% Gauss, 20% MexicanHat)
        auto gaussKernel = std::dynamic_pointer_cast<GaussKernel>(connectionGene.getKernel());
        auto mhKernel    = std::dynamic_pointer_cast<MexicanHatKernel>(connectionGene.getKernel());
        REQUIRE((gaussKernel != nullptr || mhKernel != nullptr));
        if (gaussKernel)
        {
            // ConnectionGene amplitudes are signed (inhibitory connections have negative amplitude)
            REQUIRE(gaussKernel->getParameters().width >= GaussKernelConstants::widthMinVal);
            REQUIRE(gaussKernel->getParameters().width <= GaussKernelConstants::widthMaxVal);
            REQUIRE(std::abs(gaussKernel->getParameters().amplitude) >= GaussKernelConstants::ampMinVal);
            REQUIRE(std::abs(gaussKernel->getParameters().amplitude) <= GaussKernelConstants::ampMaxVal);
        }
    }

    SECTION("Initialize ConnectionGene with ConnectionTuple and GaussKernelParameters")
    {
        ConnectionTuple connectionTuple(3, 4);
        GaussKernelParameters gkp{ 5.0, 3.0, false, false };
        ConnectionGene connectionGene(connectionTuple, 0, gkp);

        REQUIRE(connectionGene.getInFieldGeneId() == 3);
        REQUIRE(connectionGene.getOutFieldGeneId() == 4);
        REQUIRE(connectionGene.isEnabled() == true);
        REQUIRE(connectionGene.getKernel() != nullptr);

        auto kernel = std::dynamic_pointer_cast<GaussKernel>(connectionGene.getKernel());
        REQUIRE(kernel != nullptr);
        REQUIRE(kernel->getParameters().width == 5.0);
        REQUIRE(kernel->getParameters().amplitude == 3.0);
    }
}

TEST_CASE("ConnectionGene Mutation", "[ConnectionGene]")
{
    const ConnectionTuple connectionTuple(1, 2);
    ConnectionGene connectionGene(connectionTuple, 0);

    // mutate() has three paths: kernel params, connection signal flip, kernel type change
    // — just verify it completes without throwing and the kernel remains valid
    REQUIRE_NOTHROW(connectionGene.mutate());
    REQUIRE(connectionGene.getKernel() != nullptr);
}

TEST_CASE("ConnectionGene Mutation Constraints", "[ConnectionGene]")
{
    ConnectionTuple connectionTuple(1, 2);
    ConnectionGene connectionGene(connectionTuple, 0);

    for (int i = 0; i < 100; ++i)
        connectionGene.mutate();

    auto mutatedKernel = std::dynamic_pointer_cast<GaussKernel>(connectionGene.getKernel());
    if (mutatedKernel)
    {
        auto mutatedParams = mutatedKernel->getParameters();
        REQUIRE(mutatedParams.width >= GaussKernelConstants::widthMinVal);
        REQUIRE(mutatedParams.width <= GaussKernelConstants::widthMaxVal);
        // amplitude is signed (inhibitory connections have negative amplitude)
        REQUIRE(std::abs(mutatedParams.amplitude) >= GaussKernelConstants::ampMinVal);
        REQUIRE(std::abs(mutatedParams.amplitude) <= GaussKernelConstants::ampMaxVal);
    }
    else
    {
        // Kernel type mutated to MexicanHat — still a valid state
        auto mhKernel = std::dynamic_pointer_cast<MexicanHatKernel>(connectionGene.getKernel());
        REQUIRE(mhKernel != nullptr);
    }
}

TEST_CASE("ConnectionGene Disable and Toggle", "[ConnectionGene]")
{
    const ConnectionTuple connectionTuple(1, 2);
    ConnectionGene connectionGene(connectionTuple, 0);

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
    ConnectionGene connectionGene(connectionTuple, 0);

    connectionGene.setInnovationNumber(42);
    REQUIRE(connectionGene.getInnovationNumber() == 42);
}

TEST_CASE("ConnectionGene Comparison Operator", "[ConnectionGene]")
{
    const ConnectionTuple connectionTuple1(1, 2);
    const ConnectionTuple connectionTuple2(3, 4);

    ConnectionGene connectionGene1(connectionTuple1, 1);
    ConnectionGene connectionGene2(connectionTuple2, 2);

    REQUIRE(connectionGene1 == connectionGene1);
    REQUIRE(!(connectionGene1 == connectionGene2));

    connectionGene2.setInnovationNumber(connectionGene1.getInnovationNumber());
    REQUIRE(connectionGene1 == connectionGene2);
}

TEST_CASE("ConnectionGene Kernel Parameters Access", "[ConnectionGene]")
{
    const ConnectionTuple connectionTuple(1, 2);
    const GaussKernelParameters gkp{ 5.0, 3.0, false, false };
    const ConnectionGene connectionGene(connectionTuple, 0, gkp);

    REQUIRE(connectionGene.getKernelWidth() == 5.0);
    REQUIRE(connectionGene.getKernelAmplitude() == 3.0);
}

TEST_CASE("ConnectionGene Initialization with Edge Values", "[ConnectionGene]")
{
    const ConnectionTuple connectionTuple(0, 0);
    const ConnectionGene connectionGene(connectionTuple, 0);

    REQUIRE(connectionGene.getInFieldGeneId() == 0);
    REQUIRE(connectionGene.getOutFieldGeneId() == 0);
    REQUIRE(connectionGene.isEnabled() == true);
    REQUIRE(connectionGene.getKernel() != nullptr);
}

TEST_CASE("ConnectionGene Set Max Innovation Number", "[ConnectionGene]")
{
    const ConnectionTuple connectionTuple(1, 2);
    ConnectionGene connectionGene(connectionTuple, 0);

    connectionGene.setInnovationNumber(std::numeric_limits<uint16_t>::max());
    REQUIRE(connectionGene.getInnovationNumber() == std::numeric_limits<uint16_t>::max());
}

TEST_CASE("ConnectionGene::clone preserves parameters and innovation number", "[ConnectionGene]")
{
    const ConnectionTuple connectionTuple(1, 2);
    const GaussKernelParameters gkp{ 5.0, 3.0, false, false };
    const ConnectionGene connectionGene(connectionTuple, 42, gkp);

    const ConnectionGene cloned = connectionGene.clone();

    REQUIRE(cloned.getInnovationNumber() == connectionGene.getInnovationNumber());
    REQUIRE(cloned.isEnabled() == connectionGene.isEnabled());
    REQUIRE(cloned.getKernelWidth() == connectionGene.getKernelWidth());
    REQUIRE(cloned.getKernelAmplitude() == connectionGene.getKernelAmplitude());
    REQUIRE(cloned.getKernel() != connectionGene.getKernel());
}

TEST_CASE("ConnectionGene::clearLastMutations empties getMutationsInLastGeneration", "[ConnectionGene]")
{
    const ConnectionTuple connectionTuple(1, 2);
    ConnectionGene connectionGene(connectionTuple, 0);
    connectionGene.mutate();

    connectionGene.clearLastMutations();

    REQUIRE(connectionGene.getMutationsInLastGeneration().empty());
}

TEST_CASE("ConnectionGene Multiple Mutations Consistency", "[ConnectionGene]")
{
    const ConnectionTuple connectionTuple(1, 2);
    ConnectionGene connectionGene(connectionTuple, 0);

    for (int i = 0; i < 1000; ++i)
        connectionGene.mutate();

    // After many mutations the kernel may be Gauss or MexicanHat — either is valid
    const bool isGauss = std::dynamic_pointer_cast<GaussKernel>(connectionGene.getKernel()) != nullptr;
    const bool isMH    = std::dynamic_pointer_cast<MexicanHatKernel>(connectionGene.getKernel()) != nullptr;
    REQUIRE((isGauss || isMH));
}
