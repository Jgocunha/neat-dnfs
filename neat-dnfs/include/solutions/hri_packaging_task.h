#pragma once

#include "neat/solution.h"
#include "neat_tools/utils.h"

namespace neat_dnfs
{
    /// @brief Evolves action selection for a robot sharing a packaging workspace with a human.
    ///
    /// @details The architecture has three input fields and one output field, all over the
    /// same one-dimensional workspace. "nf 1" perceives the small objects, "nf 2" the large
    /// object, "nf 3" the position of the human's hand, and "nf 4" is the output, where a peak
    /// marks the object the robot acts on. Two small objects sit at positions A and B, and the
    /// large object appears between them at C.
    ///
    /// The run goes through five phases:
    ///
    /// Phase 1 -- the two small objects are presented. "nf 1" must hold a peak at each, and the
    /// output must select exactly one of them; which one is arbitrary.
    ///
    /// Phase 2 -- the hand appears at C. "nf 3" must hold a peak there, and the output must be
    /// inhibited at C: a trough a fixed depth below its resting level, not merely an absence of
    /// a peak. The hand's position negatively pre-shapes the robot's choice.
    ///
    /// Phase 3 -- the hand moves to A and then to B. Each time the output must select the other
    /// small object, so the robot takes the object the human is not reaching for.
    ///
    /// Phase 4 -- the large object appears at C, and the hand visits C, A, C and B in turn.
    /// "nf 2" must hold a peak at C. While the hand is at a small object the output must again
    /// select the other one; while the hand is at C the output must select C, so the robot
    /// joins the human on the large object.
    ///
    /// Phase 5 -- every stimulus is removed, and the mean activation of all four fields must
    /// return to its resting level. Nothing may be held as working memory.
    ///
    /// These checks are combined into seven partial fitness terms, in this order:
    ///
    /// - f1, small object detection: the two peaks in "nf 1", and "nf 1" back at rest.
    /// - f2, large object detection: the peak in "nf 2", and "nf 2" back at rest.
    /// - f3, hand detection: the peak in "nf 3", and "nf 3" back at rest.
    /// - f4, assistive action selection: the output selects C both times the hand is at C.
    /// - f5, complementary action selection: the output selects B with the hand at A in
    ///   phase 3, and A with the hand at B in phase 4 (checked twice).
    /// - f6, complementary action selection: the output selects A with the hand at B in
    ///   phase 3, B with the hand at A in phase 4, and the arbitrary selection of phase 1.
    /// - f7, action inhibition: the trough at C in phase 2, and the output back at rest.
    ///
    /// In evolution runs, solutions with only direct connections from the inputs to the output
    /// satisfied every term except f5 and f6, and every solution that met all seven had grown
    /// at least one hidden field.
    class HRIPackagingTask final : public Solution
    {
    public:
        explicit HRIPackagingTask(const SolutionTopology& topology);
        HRIPackagingTask(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype);
        SolutionPtr clone() const override;
        SolutionPtr copy() const override;
    private:
        void testPhenotype() override;
        void createPhenotypeEnvironment() override;
    };
}