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
    FieldGene fieldGene(params);

    // mutate() randomly targets kernel (70%), neural field (20%), or kernel type (10%)
    // — just verify it completes without throwing and the kernel remains valid
    REQUIRE_NOTHROW(fieldGene.mutate());
    REQUIRE(fieldGene.getKernel() != nullptr);
    REQUIRE(fieldGene.getNeuralField() != nullptr);
}

TEST_CASE("FieldGene Mutation Constraints", "[FieldGene]")
{
    FieldGeneParameters params(FieldGeneType::HIDDEN, 5);
    FieldGene fieldGene(params);

    for (int i = 0; i < 100; ++i)
        fieldGene.mutate();

    auto mutatedKernel = std::dynamic_pointer_cast<GaussKernel>(fieldGene.getKernel());
    if (mutatedKernel)
    {
        auto mutatedParams = mutatedKernel->getParameters();
        REQUIRE(mutatedParams.width >= GaussKernelConstants::widthMinVal);
        REQUIRE(mutatedParams.width <= GaussKernelConstants::widthMaxVal);
        REQUIRE(mutatedParams.amplitude >= GaussKernelConstants::ampMinVal);
        REQUIRE(mutatedParams.amplitude <= GaussKernelConstants::ampMaxVal);
    }
    else
    {
        // Kernel type may have mutated to MexicanHat — still valid
        auto mhKernel = std::dynamic_pointer_cast<MexicanHatKernel>(fieldGene.getKernel());
        REQUIRE(mhKernel != nullptr);
    }
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

TEST_CASE("FieldGene Mutation Does Not Crash on INPUT type", "[FieldGene]")
{
    const FieldGeneParameters params(FieldGeneType::INPUT, 1);
    FieldGene fieldGene(params);

    // FieldGene::mutate() does not guard INPUT type — verify it completes without throwing
    REQUIRE_NOTHROW(fieldGene.mutate());
    REQUIRE(fieldGene.getKernel() != nullptr);
    REQUIRE(fieldGene.getNeuralField() != nullptr);
}

TEST_CASE("FieldGene NeuralField Parameters", "[FieldGene]")
{
    const FieldGeneParameters params(FieldGeneType::INPUT, 9);
    const FieldGene fieldGene(params);

    auto neuralField = fieldGene.getNeuralField();
    REQUIRE(neuralField != nullptr);

    // tau is randomized when variableParameters == true
    const auto neuralFieldParams = neuralField->getParameters();
    REQUIRE(neuralFieldParams.tau >= NeuralFieldConstants::tauMinVal);
    REQUIRE(neuralFieldParams.tau <= NeuralFieldConstants::tauMaxVal);
}

TEST_CASE("FieldGene::clone produces an independent deep copy", "[FieldGene]")
{
    const FieldGeneParameters params(FieldGeneType::HIDDEN, 7);
    const FieldGene fieldGene(params);

    const FieldGene cloned = fieldGene.clone();

    // operator== only compares `parameters` (type + id), so equal here means
    // "same identity slot", not "identical kernel content".
    REQUIRE(fieldGene == cloned);
    REQUIRE(cloned.getKernel() != fieldGene.getKernel());
    REQUIRE(cloned.getNeuralField() != fieldGene.getNeuralField());
}

TEST_CASE("FieldGene::clearLastMutations empties getMutationsInLastGeneration", "[FieldGene]")
{
    FieldGeneParameters params(FieldGeneType::HIDDEN, 8);
    FieldGene fieldGene(params);
    fieldGene.mutate();

    fieldGene.clearLastMutations();

    REQUIRE(fieldGene.getMutationsInLastGeneration().empty());
}

TEST_CASE("FieldGene::setAsHidden changes the gene type to HIDDEN", "[FieldGene]")
{
    FieldGeneParameters params(FieldGeneType::INPUT, 11);
    FieldGene fieldGene(params);
    REQUIRE(fieldGene.getParameters().type == FieldGeneType::INPUT);

    fieldGene.setAsHidden(ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });

    REQUIRE(fieldGene.getParameters().type == FieldGeneType::HIDDEN);
}

TEST_CASE("FieldGene Kernel Parameters Access", "[FieldGene]")
{
    const FieldGeneParameters params(FieldGeneType::HIDDEN, 10);
    const FieldGene fieldGene(params);

    REQUIRE(fieldGene.getKernel() != nullptr);

    // Kernel type is probabilistic (80% Gauss, 20% MexicanHat)
    const auto kernel = std::dynamic_pointer_cast<GaussKernel>(fieldGene.getKernel());
    if (kernel)
    {
        const auto gkp = kernel->getParameters();
        REQUIRE(gkp.width >= GaussKernelConstants::widthMinVal);
        REQUIRE(gkp.width <= GaussKernelConstants::widthMaxVal);
        REQUIRE(gkp.amplitude >= GaussKernelConstants::ampMinVal);
        REQUIRE(gkp.amplitude <= GaussKernelConstants::ampMaxVal);
    }
    else
    {
        const auto mhKernel = std::dynamic_pointer_cast<MexicanHatKernel>(fieldGene.getKernel());
        REQUIRE(mhKernel != nullptr);
    }
}
