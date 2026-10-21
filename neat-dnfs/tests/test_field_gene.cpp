#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <set>

#include "neat/field_gene.h"
#include "neat_tools/config_loader.h"
#include "test_helpers.h"

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

namespace
{
    // Restores the input and output parameters a test fixes.
    class FixedFieldsGuard
    {
    public:
        FixedFieldsGuard() : input(FieldGeneConstants::fixedInputField), output(FieldGeneConstants::fixedOutputField) {}
        ~FixedFieldsGuard()
        {
            FieldGeneConstants::fixedInputField = input;
            FieldGeneConstants::fixedOutputField = output;
        }
        FixedFieldsGuard(const FixedFieldsGuard&) = delete;
        FixedFieldsGuard& operator=(const FixedFieldsGuard&) = delete;
    private:
        FixedFieldParameters input;
        FixedFieldParameters output;
    };

    // The serial-order reference design's input: a detection field, its peak gone once the stimulus is.
    const FixedFieldParameters detectionField{ 10.0, -10.0, GaussKernelParameters{ 2.0, 3.0, 0.0, true, false } };

    GaussKernelParameters gaussKernelOf(const FieldGene& gene)
    {
        const auto kernel = std::dynamic_pointer_cast<GaussKernel>(gene.getKernel());
        REQUIRE(kernel != nullptr);
        return kernel->getParameters();
    }

    void requireDetectionField(const FieldGene& gene)
    {
        CHECK(gene.getNeuralField()->getParameters().tau == 10.0);
        CHECK(gene.getNeuralField()->getParameters().startingRestingLevel == -10.0);
        const auto kernel = gaussKernelOf(gene);
        CHECK(kernel.width == 2.0);
        CHECK(kernel.amplitude == 3.0);
        CHECK(kernel.amplitudeGlobal == 0.0);
    }
}

TEST_CASE("FieldGene: an input field takes the parameters a task fixes for it", "[FieldGene][FixedFields]")
{
    FixedFieldsGuard guard;
    FieldGeneConstants::fixedInputField = detectionField;

    const FieldGene input(FieldGeneParameters(FieldGeneType::INPUT, 1));

    requireDetectionField(input);
}

TEST_CASE("FieldGene: mutation never changes a field's fixed parameters", "[FieldGene][FixedFields]")
{
    FixedFieldsGuard guard;
    FieldGeneConstants::fixedInputField = detectionField;
    FieldGene input(FieldGeneParameters(FieldGeneType::INPUT, 1));

    for (int mutation = 0; mutation < 500; ++mutation)
    {
        input.mutate();
    }

    requireDetectionField(input);
}

TEST_CASE("FieldGene: an output with only its time scale fixed keeps it while the rest evolves", "[FieldGene][FixedFields]")
{
    FixedFieldsGuard guard;
    FieldGeneConstants::fixedOutputField = FixedFieldParameters{ 10.0, std::nullopt, std::nullopt };
    FieldGene output(FieldGeneParameters(FieldGeneType::OUTPUT, 2));
    const double restingLevel = output.getNeuralField()->getParameters().startingRestingLevel;
    const auto kernel = output.getKernel();
    bool restingLevelChanged = false;
    bool kernelChanged = false;

    for (int mutation = 0; mutation < 500; ++mutation)
    {
        output.mutate();
        restingLevelChanged = restingLevelChanged || output.getNeuralField()->getParameters().startingRestingLevel != restingLevel;
        // One mutation changes either the kernel or the neural field.
        const std::string mutations = output.getMutationsInLastGeneration();
        kernelChanged = kernelChanged || (!mutations.empty() && mutations.find("(fg nf") == std::string::npos);
        output.clearLastMutations();
    }

    CHECK(output.getNeuralField()->getParameters().tau == 10.0);
    CHECK(restingLevelChanged);
    CHECK(kernelChanged);
}

TEST_CASE("FieldGene: hidden fields ignore the parameters fixed for the input and output", "[FieldGene][FixedFields]")
{
    FixedFieldsGuard guard;
    FieldGeneConstants::fixedInputField = detectionField;
    FieldGeneConstants::fixedOutputField = FixedFieldParameters{ 10.0, std::nullopt, std::nullopt };

    std::set<double> taus;
    for (int id = 3; id < 13; ++id)
    {
        taus.insert(FieldGene(FieldGeneParameters(FieldGeneType::HIDDEN, id)).getNeuralField()->getParameters().tau);
    }

    CHECK(taus.size() > 1);
}

TEST_CASE("FieldGene: the reference config fixes no field parameters", "[FieldGene][FixedFields]")
{
    FixedFieldsGuard guard;
    FieldGeneConstants::fixedInputField = detectionField;
    FieldGeneConstants::fixedOutputField = detectionField;

    ConfigLoader::loadGlobalConfig(ConfigLoader::defaultGlobalConfigPath());

    for (const auto& fixed : { FieldGeneConstants::fixedInputField, FieldGeneConstants::fixedOutputField })
    {
        CHECK_FALSE(fixed.tau);
        CHECK_FALSE(fixed.restingLevel);
        CHECK_FALSE(fixed.gaussKernel);
    }
}

// The input only detects: as in the reference design, its peak should go when the stimulus does, so it cannot hold
// the events. The output must be fast enough to step and release within a recall cycle.
TEST_CASE("FieldGene: serial-order-fitness-test fixes its input as a detection field and its output's time scale", "[FieldGene][FixedFields]")
{
    FixedFieldsGuard guard;
    const neat_dnfs::test::ScopedTaskConfig taskConfig{ "serial-order-fitness-test" };

    const FieldGene input(FieldGeneParameters(FieldGeneType::INPUT, 1));
    requireDetectionField(input);
    const auto& output = FieldGeneConstants::fixedOutputField;
    CHECK(output.tau == 10.0);
    CHECK_FALSE(output.restingLevel);
    CHECK_FALSE(output.gaussKernel);
}

TEST_CASE("FieldGene: a misspelt key in a fixed field is an error, not a parameter left to evolve", "[FieldGene][FixedFields]")
{
    FixedFieldsGuard guard;
    auto config = ConfigLoader::loadJsonFile(ConfigLoader::defaultGlobalConfigPath());
    config.at("FieldGeneConstants")["fixedInputField"] = { { "Tau", 10.0 } };
    const auto path = std::filesystem::temp_directory_path() / "neat-dnfs-config-misspelt-fixed-field.json";
    std::ofstream(path) << config.dump();

    CHECK_THROWS(ConfigLoader::loadGlobalConfig(path.generic_string()));

    ConfigLoader::loadGlobalConfig(ConfigLoader::defaultGlobalConfigPath());
    std::filesystem::remove(path);
}
