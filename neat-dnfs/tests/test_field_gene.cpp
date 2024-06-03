#include <catch2/catch_test_macros.hpp>
#include "neat/field_gene.h"

using namespace neat_dnfs;
using namespace dnf_composer::element;

TEST_CASE("FieldGene Initialization", "[FieldGene]")
{
    SECTION("Initialize FieldGene as INPUT")
	{
        FieldGeneParameters params(FieldGeneType::INPUT, 1);
        FieldGene fieldGene(params);

        REQUIRE(fieldGene.getParameters().type == FieldGeneType::INPUT);
        REQUIRE(fieldGene.getNeuralField() != nullptr);
        REQUIRE(fieldGene.getKernel() != nullptr);
    }

    SECTION("Initialize FieldGene as OUTPUT")
	{
        FieldGeneParameters params(FieldGeneType::OUTPUT, 2);
        FieldGene fieldGene(params);

        REQUIRE(fieldGene.getParameters().type == FieldGeneType::OUTPUT);
        REQUIRE(fieldGene.getNeuralField() != nullptr);
        REQUIRE(fieldGene.getKernel() != nullptr);
    }

    SECTION("Initialize FieldGene as HIDDEN")
	{
        FieldGeneParameters params(FieldGeneType::HIDDEN, 3);
        FieldGene fieldGene(params);

        REQUIRE(fieldGene.getParameters().type == FieldGeneType::HIDDEN);
        REQUIRE(fieldGene.getNeuralField() != nullptr);
        REQUIRE(fieldGene.getKernel() != nullptr);
    }
}

TEST_CASE("FieldGene ID Verification", "[FieldGene]")
{
    // Reset the static variable currentFieldGeneId for testing purposes

    const FieldGeneParameters params1(FieldGeneType::INPUT, 1);
    const FieldGene fieldGene1(params1);
    REQUIRE(fieldGene1.getParameters().id == 1);

    const FieldGeneParameters params2(FieldGeneType::OUTPUT, 2);
    const FieldGene fieldGene2(params2);
    REQUIRE(fieldGene2.getParameters().id == 2);

    const FieldGeneParameters params3(FieldGeneType::HIDDEN, 3);
    const FieldGene fieldGene3(params3);
    REQUIRE(fieldGene3.getParameters().id == 3);
}


TEST_CASE("FieldGene Mutation Only One Parameter", "[FieldGene]")
{
    const FieldGeneParameters params({ FieldGeneType::HIDDEN, 4 });
    const FieldGene fieldGene(params);

	const auto initialKernel = std::dynamic_pointer_cast<GaussKernel>(fieldGene.getKernel());
    const auto initialParams = initialKernel->getParameters();

	fieldGene.mutate();
     
    const auto mutatedKernel = std::dynamic_pointer_cast<GaussKernel>(fieldGene.getKernel());
    const auto mutatedParams = mutatedKernel->getParameters();
     
    const bool sigmaChanged = initialParams.sigma != mutatedParams.sigma;
    const bool amplitudeChanged = initialParams.amplitude != mutatedParams.amplitude;

    REQUIRE((sigmaChanged != amplitudeChanged)); // Only one should change
}

TEST_CASE("FieldGene Mutation Constraints", "[FieldGene]")
{
    FieldGeneParameters params(FieldGeneType::HIDDEN, 5);
    FieldGene fieldGene(params);

    auto initialKernel = std::dynamic_pointer_cast<GaussKernel>(fieldGene.getKernel());
    auto initialParams = initialKernel->getParameters();

    // Mutate multiple times to test constraints
    for (int i = 0; i < 100; ++i) 
        fieldGene.mutate();

    auto mutatedKernel = std::dynamic_pointer_cast<GaussKernel>(fieldGene.getKernel());
    auto mutatedParams = mutatedKernel->getParameters();

    REQUIRE(mutatedParams.sigma >= MutationConstants::minSigma);
    REQUIRE(mutatedParams.sigma <= MutationConstants::maxSigma);
    REQUIRE(mutatedParams.amplitude >= MutationConstants::minAmplitude);
    REQUIRE(mutatedParams.amplitude <= MutationConstants::maxAmplitude);
}

TEST_CASE("FieldGene Comparison Operator", "[FieldGene]")
{
    const FieldGeneParameters params1(FieldGeneType::INPUT, 1);
    const FieldGeneParameters params2(FieldGeneType::INPUT, 2);

    const FieldGene fieldGene1(params1);
    const FieldGene fieldGene2(params2);

    REQUIRE(fieldGene1 == fieldGene1);
    REQUIRE(!(fieldGene1 == fieldGene2));
}

TEST_CASE("FieldGene Invalid Mutation", "[FieldGene]")
{
    const FieldGeneParameters params(FieldGeneType::INPUT, 1);
    const FieldGene fieldGene(params);

    auto initialKernel = fieldGene.getKernel();
    fieldGene.mutate();  // This should not change the kernel as it's not hidden

    auto kernelAfterMutation = fieldGene.getKernel();
    REQUIRE(initialKernel == kernelAfterMutation);
}

TEST_CASE("FieldGene Mutate Non-GaussKernel", "[FieldGene]")
{
    const FieldGeneParameters params(FieldGeneType::OUTPUT, 6);
    const FieldGene fieldGene(params);

    auto initialKernel = fieldGene.getKernel();
    fieldGene.mutate();  // This should log an error and throw an exception because it's not a GaussKernel

    auto kernelAfterMutation = fieldGene.getKernel();
    REQUIRE(initialKernel == kernelAfterMutation);
}

TEST_CASE("FieldGene NeuralField Parameters", "[FieldGene]")
{
    const FieldGeneParameters params(FieldGeneType::INPUT, 9);
    const FieldGene fieldGene(params);

    auto neuralField = fieldGene.getNeuralField();
    REQUIRE(neuralField != nullptr);

    const auto neuralFieldParams = neuralField->getParameters();
    REQUIRE(neuralFieldParams.tau == NeuralFieldConstants::tau);
    REQUIRE(neuralFieldParams.startingRestingLevel == NeuralFieldConstants::restingLevel);
}

TEST_CASE("FieldGene Kernel Parameters Access", "[FieldGene]")
{
    const FieldGeneParameters params(FieldGeneType::HIDDEN, 10);
    const FieldGene fieldGene(params);

    const auto kernel = std::dynamic_pointer_cast<GaussKernel>(fieldGene.getKernel());
    REQUIRE(kernel != nullptr);
    const auto gkp = kernel->getParameters();
    REQUIRE(gkp.sigma >= GaussKernelConstants::initialAmplitudeMin);
    REQUIRE(gkp.sigma <= GaussKernelConstants::initialAmplitudeMax);
    REQUIRE(gkp.amplitude >= GaussKernelConstants::initialAmplitudeMin);
    REQUIRE(gkp.amplitude <= GaussKernelConstants::initialAmplitudeMax);
}
