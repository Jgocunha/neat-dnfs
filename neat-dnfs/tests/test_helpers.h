#pragma once
#include "neat/solution.h"

namespace neat_dnfs::test {

inline SolutionTopology makeTopology(int inputs, int outputs, int hidden = 0)
{
    using dnf_composer::element::ElementDimensions;
    std::vector<std::pair<FieldGeneType, ElementDimensions>> genes;
    for (int i = 0; i < inputs;  ++i) genes.push_back({FieldGeneType::INPUT,  ElementDimensions{100, 1.0}});
    for (int i = 0; i < outputs; ++i) genes.push_back({FieldGeneType::OUTPUT, ElementDimensions{100, 1.0}});
    for (int i = 0; i < hidden;  ++i) genes.push_back({FieldGeneType::HIDDEN, ElementDimensions{100, 1.0}});
    return SolutionTopology(genes);
}

} // namespace neat_dnfs::test
