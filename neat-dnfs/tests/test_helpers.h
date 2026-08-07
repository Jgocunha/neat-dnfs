#pragma once
#include "neat/solution.h"
#include "neat/species.h"

namespace neat_dnfs::test {

// FieldGene's usual constructor randomizes everything that decides whether a
// field can hold a bump: tau and restingLevel (FieldGene::initializeNeuralField),
// the kernel type (80/20 Gauss/Mexican-hat), and the Gauss kernel's width,
// amplitude and global amplitude (FieldGene::initializeGaussKernel). Unlucky
// draws land in a regime where no bump forms, so any fixture asserting on bump
// formation flakes -- that is what made
// "twoBumpsAtPositionWithAmplitudeAndWidth does not credit the same bump ..."
// fail intermittently in CI.
//
// This builds a gene through FieldGene's explicit-field/kernel constructor
// instead, which draws no random numbers at all. The values are the same ones
// production uses when FieldGeneConstants::variableParameters is false, i.e. a
// tuned regime that forms one stable bump well clear of the detection
// threshold. That margin is the point: the simulation is still stochastic
// (every field carries a NormalNoise whose RNG lives in dnf_composer and has no
// seed hook), so the bump has to be stable enough that noise cannot flip it,
// rather than merely reproducible.
//
// Mirrors the convention test_genome.cpp / test_connection_gene.cpp already use
// on the ConnectionGene side: explicit-kernel constructor + a comment naming
// the randomization being escaped.
inline FieldGene makeFixedFieldGene(const FieldGeneType type, const int id)
{
    using namespace dnf_composer::element;

    const ElementDimensions dimensions{ DimensionConstants::xSize, DimensionConstants::dx };

    // Named "nf <id>": Solution::buildPhenotype() copies the element name off
    // the gene's own NeuralField, and fixtures look the field up by that name.
    const ElementCommonParameters nfcp{
        std::format("{}{}", NeuralFieldConstants::namePrefix, id), dimensions };
    const NeuralFieldParameters nfp{ NeuralFieldConstants::tau,
        NeuralFieldConstants::restingLevel,
        NeuralFieldConstants::activationFunction };

    const ElementCommonParameters gkcp{
        std::format("{}{}", GaussKernelConstants::namePrefix, id), dimensions };
    const GaussKernelParameters gkp{ GaussKernelConstants::width,
        GaussKernelConstants::amplitude,
        GaussKernelConstants::amplitudeGlobal,
        KernelConstants::circularity,
        KernelConstants::normalization };

    return FieldGene{ FieldGeneParameters{ type, id },
        std::make_shared<NeuralField>(nfcp, nfp),
        std::make_shared<GaussKernel>(gkcp, gkp) };
}

inline SolutionTopology makeTopology(int inputs, int outputs, int hidden = 0)
{
    using dnf_composer::element::ElementDimensions;
    std::vector<std::pair<FieldGeneType, ElementDimensions>> genes;
    for (int i = 0; i < inputs;  ++i) genes.push_back({FieldGeneType::INPUT,  ElementDimensions{100, 1.0}});
    for (int i = 0; i < outputs; ++i) genes.push_back({FieldGeneType::OUTPUT, ElementDimensions{100, 1.0}});
    for (int i = 0; i < hidden;  ++i) genes.push_back({FieldGeneType::HIDDEN, ElementDimensions{100, 1.0}});
    return SolutionTopology(genes);
}

// Genome::globalInnovationNumber, Solution::uniqueIdentifierCounter, and
// Species::currentSpeciesId are process-global statics not reset between
// TEST_CASEs. Call this at the start of any test that asserts on ids,
// innovation numbers, or generational mutation state, so the suite is
// order-independent under --order rand.
inline void resetGlobalState()
{
    Genome::resetGlobalInnovationNumber();
    Genome::clearGenerationalInnovations();
    Solution::resetUniqueIdentifier();
    Species::resetUniqueIdentifier();
}

} // namespace neat_dnfs::test
