#pragma once

#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

namespace neat_dnfs
{
    /// @brief One individual as the Pareto ranking sees it: its objective vector
    /// (maximised) and its total constraint violation (0 when feasible).
    struct RankedPoint
    {
        std::vector<double> objectives;
        double violation{0.0};
    };

    /// @brief Epsilon-dominance for maximisation: @p a dominates @p b iff
    /// a_i >= b_i - epsilon for every i and a_j > b_j + epsilon for some j.
    /// @param a Objective vector of the candidate dominator.
    /// @param b Objective vector of the candidate dominated point; same length as @p a.
    /// @param epsilon Tolerance absorbing simulation noise; 0 gives plain Pareto dominance.
    /// @return True if @p a dominates @p b. Equal vectors never dominate each other.
    [[nodiscard]] bool dominates(std::span<const double> a, std::span<const double> b, double epsilon);

    /// @brief Constrained-domination (Deb 2002, section V).
    /// @details A feasible point (violation 0) beats an infeasible one; between two
    /// infeasible points the smaller violation wins; between two feasible points
    /// dominates() decides. With every violation at 0 this is exactly dominates().
    /// @param a The candidate dominator.
    /// @param b The candidate dominated point.
    /// @param epsilon Tolerance passed to dominates() when both points are feasible.
    /// @return True if @p a constrained-dominates @p b.
    [[nodiscard]] bool constrainedDominates(const RankedPoint& a, const RankedPoint& b, double epsilon);

    /// @brief Total shortfall of the partial fitnesses below a feasibility floor.
    /// @param partials Raw partial fitness values.
    /// @param floor Minimum acceptable partial fitness; 0 or less disables the constraint.
    /// @return Sum over partials of max(0, floor - partial); 0 when the floor is disabled.
    [[nodiscard]] double constraintViolation(std::span<const double> partials, double floor);

    /// @brief NSGA-II fast non-dominated sort under constrainedDominates().
    /// @details Every point is placed in exactly one front. Indices within a front are
    /// in ascending order. A positive epsilon can make dominance cyclic (a > b > c > a);
    /// when no remaining point is undominated, the remaining points dominated by the
    /// fewest other remaining points form the next front, so the sort always terminates.
    /// @param points The population to rank.
    /// @param epsilon Dominance tolerance passed to constrainedDominates().
    /// @return Fronts as lists of indices into @p points, front 0 (the non-dominated set) first.
    [[nodiscard]] std::vector<std::vector<size_t>> nonDominatedSort(std::span<const RankedPoint> points, double epsilon);

    /// @brief NSGA-II crowding distance of the members of one front.
    /// @details Per objective, the members are ordered by value; the first and last get
    /// +infinity, and each interior member adds the gap between its neighbours divided
    /// by the objective's range. An objective with zero range contributes nothing, not
    /// even infinite boundaries. A front of one or two members is all +infinity.
    /// @param points The population the front indexes into.
    /// @param front Indices into @p points of the front's members.
    /// @return One distance per member, in the order of @p front.
    [[nodiscard]] std::vector<double> crowdingDistances(std::span<const RankedPoint> points, std::span<const size_t> front);

    /// @brief Collapses raw partial fitnesses into one objective per group.
    /// @details Each objective is the weighted mean of its group's partials, with the
    /// weights renormalised within the group; a group whose weights sum to zero uses the
    /// plain mean. The groups are assumed to index validly into @p partials and @p weights.
    /// @param partials Raw partial fitness values.
    /// @param groups Indices into @p partials, one list per objective; empty means one
    /// objective per partial.
    /// @param weights The task's fitness weights, one per partial.
    /// @return The objective vector: one value per group, or @p partials unchanged when
    /// @p groups is empty.
    [[nodiscard]] std::vector<double> groupObjectives(std::span<const double> partials,
        const std::vector<std::vector<size_t>>& groups, std::span<const double> weights);

    /// @brief Checks that objective groups partition a task's partial-fitness indices.
    /// @details Valid groups use every index in [0, @p partialCount) exactly once and
    /// contain no empty group. An empty list of groups is always valid (one objective
    /// per partial).
    /// @param groups The candidate grouping, one index list per objective.
    /// @param partialCount Number of partial-fitness terms the task produces.
    /// @param taskName Task slug, named in the error message.
    /// @throws std::runtime_error naming @p taskName if the groups are not a partition.
    void validateObjectiveGroups(const std::vector<std::vector<size_t>>& groups, size_t partialCount, std::string_view taskName);

    /// @brief One point stored in the ParetoArchive.
    struct ParetoArchiveEntry
    {
        int solutionId{-1};
        int speciesId{-1};
        int generationFound{0};
        std::vector<double> objectives;
        std::vector<double> partials;
        double fitness{0.0};
        double violation{0.0};
    };

    /// @brief Bounded archive of the mutually non-dominated feasible points found so far.
    /// @details Only feasible entries (violation 0) are stored. When an insertion takes
    /// the archive above its capacity, the member with the smallest crowding distance is
    /// evicted, and a boundary point (infinite crowding distance) never is.
    class ParetoArchive
    {
    public:
        /// @brief Creates an empty archive.
        /// @param capacity Maximum number of members; must be at least 1.
        explicit ParetoArchive(size_t capacity);

        /// @brief Offers an entry to the archive.
        /// @details The entry's violation is always recorded for bestViolationSeen(). An
        /// infeasible entry is then rejected, as is one that an existing member dominates
        /// or equals within @p epsilon on every objective. Otherwise every member it
        /// dominates is removed, it is added, and the most crowded non-boundary member is
        /// evicted if the archive is above capacity (which may be the entry itself).
        /// @param entry The candidate point.
        /// @param epsilon Dominance and equality tolerance.
        /// @return True if the entry is in the archive after the call.
        [[nodiscard]] bool tryInsert(const ParetoArchiveEntry& entry, double epsilon);

        /// @brief The current members, in insertion order.
        /// @return A const reference to the members.
        [[nodiscard]] const std::vector<ParetoArchiveEntry>& members() const;

        /// @brief The current number of members.
        /// @return The member count.
        [[nodiscard]] size_t size() const;

        /// @brief The smallest violation of any entry ever offered to tryInsert().
        /// @details Serves as the improvement signal while nothing is feasible yet.
        /// @return The smallest violation seen, or +infinity if nothing has been offered.
        [[nodiscard]] double bestViolationSeen() const;

    private:
        size_t capacity;
        std::vector<ParetoArchiveEntry> entries;
        double lowestViolation;

        [[nodiscard]] bool isDominatedOrMatched(const ParetoArchiveEntry& entry, double epsilon) const;
        void removeDominatedBy(const ParetoArchiveEntry& entry, double epsilon);
        /// Index of the member to evict when over capacity: the smallest crowding
        /// distance, or the newest member when every member is a boundary point.
        [[nodiscard]] size_t evictionCandidate() const;
    };
}
