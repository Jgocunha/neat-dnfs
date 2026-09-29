#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <algorithm>
#include <limits>
#include <vector>

#include "neat/pareto.h"

using namespace neat_dnfs;

namespace
{
    constexpr double infinity = std::numeric_limits<double>::infinity();

    std::vector<RankedPoint> feasiblePoints(const std::vector<std::vector<double>>& objectives)
    {
        std::vector<RankedPoint> points;
        for (const auto& objective : objectives)
            points.push_back({ objective, 0.0 });
        return points;
    }

    ParetoArchiveEntry entryAt(const int solutionId, const std::vector<double>& objectives,
        const double violation = 0.0)
    {
        return { solutionId, 1, 1, objectives, objectives, 0.5, violation };
    }

    std::vector<int> memberIds(const ParetoArchive& archive)
    {
        std::vector<int> ids;
        for (const auto& member : archive.members())
            ids.push_back(member.solutionId);
        std::ranges::sort(ids);
        return ids;
    }
}

TEST_CASE("dominates without epsilon", "[Pareto]")
{
    using V = std::vector<double>;
    REQUIRE(dominates(V{ 1.0, 1.0 }, V{ 0.0, 0.0 }, 0.0));
    REQUIRE(dominates(V{ 1.0, 0.5 }, V{ 1.0, 0.0 }, 0.0));
    REQUIRE_FALSE(dominates(V{ 1.0, 0.0 }, V{ 0.0, 1.0 }, 0.0));
    REQUIRE_FALSE(dominates(V{ 0.0, 1.0 }, V{ 1.0, 0.0 }, 0.0));
    REQUIRE_FALSE(dominates(V{ 0.0, 0.0 }, V{ 1.0, 1.0 }, 0.0));

    SECTION("equal vectors do not dominate each other")
    {
        REQUIRE_FALSE(dominates(V{ 0.5, 0.5 }, V{ 0.5, 0.5 }, 0.0));
    }
}

TEST_CASE("dominates with epsilon", "[Pareto]")
{
    using V = std::vector<double>;
    constexpr double epsilon = 0.1;

    SECTION("an improvement no larger than epsilon does not dominate")
    {
        REQUIRE(dominates(V{ 0.55, 0.5 }, V{ 0.5, 0.5 }, 0.0));
        REQUIRE_FALSE(dominates(V{ 0.55, 0.5 }, V{ 0.5, 0.5 }, epsilon));
    }

    SECTION("a loss within epsilon is tolerated when another objective improves by more than epsilon")
    {
        REQUIRE_FALSE(dominates(V{ 0.7, 0.45 }, V{ 0.5, 0.5 }, 0.0));
        REQUIRE(dominates(V{ 0.7, 0.45 }, V{ 0.5, 0.5 }, epsilon));
    }

    SECTION("a loss larger than epsilon still blocks dominance")
    {
        REQUIRE_FALSE(dominates(V{ 0.9, 0.35 }, V{ 0.5, 0.5 }, epsilon));
    }

    SECTION("equal vectors do not dominate each other")
    {
        REQUIRE_FALSE(dominates(V{ 0.5, 0.5 }, V{ 0.5, 0.5 }, epsilon));
    }
}

TEST_CASE("constraintViolation", "[Pareto]")
{
    using V = std::vector<double>;

    SECTION("a floor of zero disables the constraint")
    {
        REQUIRE(constraintViolation(V{ 0.0, 0.2, 1.0 }, 0.0) == 0.0);
    }

    SECTION("sums the shortfall of every partial below the floor")
    {
        REQUIRE(constraintViolation(V{ 0.1, 0.5, 0.25 }, 0.3) == Catch::Approx(0.25));
        REQUIRE(constraintViolation(V{ 0.0, 0.0 }, 0.1) == Catch::Approx(0.2));
    }

    SECTION("partials at or above the floor contribute nothing")
    {
        REQUIRE(constraintViolation(V{ 0.3, 0.9 }, 0.3) == 0.0);
    }
}

TEST_CASE("constrainedDominates", "[Pareto]")
{
    SECTION("a feasible point beats an infeasible one that is better on every objective")
    {
        const RankedPoint feasible{ { 0.1, 0.1 }, 0.0 };
        const RankedPoint infeasible{ { 1.0, 1.0 }, 0.2 };
        REQUIRE(constrainedDominates(feasible, infeasible, 0.0));
        REQUIRE_FALSE(constrainedDominates(infeasible, feasible, 0.0));
    }

    SECTION("between infeasible points the smaller violation wins, whatever the objectives")
    {
        const RankedPoint smallerViolation{ { 0.0, 0.0 }, 0.1 };
        const RankedPoint largerViolation{ { 1.0, 1.0 }, 0.3 };
        REQUIRE(constrainedDominates(smallerViolation, largerViolation, 0.0));
        REQUIRE_FALSE(constrainedDominates(largerViolation, smallerViolation, 0.0));
    }

    SECTION("infeasible points with equal violation do not dominate each other")
    {
        const RankedPoint first{ { 1.0, 1.0 }, 0.2 };
        const RankedPoint second{ { 0.0, 0.0 }, 0.2 };
        REQUIRE_FALSE(constrainedDominates(first, second, 0.0));
        REQUIRE_FALSE(constrainedDominates(second, first, 0.0));
    }

    SECTION("reduces to dominates when both points are feasible")
    {
        const std::vector<std::pair<std::vector<double>, std::vector<double>>> pairs{
            { { 1.0, 1.0 }, { 0.0, 0.0 } },
            { { 1.0, 0.0 }, { 0.0, 1.0 } },
            { { 0.5, 0.5 }, { 0.5, 0.5 } },
            { { 0.7, 0.45 }, { 0.5, 0.5 } },
            { { 0.55, 0.5 }, { 0.5, 0.5 } },
        };
        for (const double epsilon : { 0.0, 0.1 })
        {
            for (const auto& [a, b] : pairs)
            {
                const RankedPoint first{ a, 0.0 };
                const RankedPoint second{ b, 0.0 };
                REQUIRE(constrainedDominates(first, second, epsilon) == dominates(a, b, epsilon));
                REQUIRE(constrainedDominates(second, first, epsilon) == dominates(b, a, epsilon));
            }
        }
    }
}

TEST_CASE("nonDominatedSort on a known two-objective set", "[Pareto]")
{
    const auto points = feasiblePoints({
        { 1.0, 0.0 },  // 0: front 0
        { 0.0, 1.0 },  // 1: front 0
        { 0.5, 0.5 },  // 2: front 0
        { 0.4, 0.4 },  // 3: front 1, dominated by 2
        { 0.0, 0.0 },  // 4: front 2, dominated by 3 and 5
        { 0.9, 0.0 },  // 5: front 1, dominated by 0
        { 0.3, 0.6 },  // 6: front 0
    });

    const auto fronts = nonDominatedSort(points, 0.0);

    const std::vector<std::vector<size_t>> expected{ { 0, 1, 2, 6 }, { 3, 5 }, { 4 } };
    REQUIRE(fronts == expected);
}

TEST_CASE("nonDominatedSort puts identical points in a single front", "[Pareto]")
{
    const auto points = feasiblePoints({ { 0.5, 0.5 }, { 0.5, 0.5 }, { 0.5, 0.5 } });

    const auto fronts = nonDominatedSort(points, 0.0);

    const std::vector<std::vector<size_t>> expected{ { 0, 1, 2 } };
    REQUIRE(fronts == expected);
}

TEST_CASE("nonDominatedSort of no points is empty", "[Pareto]")
{
    REQUIRE(nonDominatedSort(std::vector<RankedPoint>{}, 0.0).empty());
}

TEST_CASE("nonDominatedSort ranks every feasible point ahead of every infeasible one", "[Pareto]")
{
    const std::vector<RankedPoint> points{
        { { 0.1, 0.1 }, 0.0 },   // 0: feasible
        { { 1.0, 1.0 }, 0.1 },   // 1: infeasible, best objectives
        { { 0.2, 0.0 }, 0.0 },   // 2: feasible
        { { 0.9, 0.9 }, 0.05 },  // 3: infeasible, smaller violation
    };

    const auto fronts = nonDominatedSort(points, 0.0);

    const std::vector<std::vector<size_t>> expected{ { 0, 2 }, { 3 }, { 1 } };
    REQUIRE(fronts == expected);
}

TEST_CASE("nonDominatedSort assigns every point even when epsilon-dominance is cyclic", "[Pareto]")
{
    // With epsilon = 0.1 each point epsilon-dominates the next: a > b > c > a.
    const auto points = feasiblePoints({
        { 0.25, 0.18, 0.10 },
        { 0.10, 0.25, 0.18 },
        { 0.18, 0.10, 0.25 },
    });
    REQUIRE(dominates(points[0].objectives, points[1].objectives, 0.1));
    REQUIRE(dominates(points[1].objectives, points[2].objectives, 0.1));
    REQUIRE(dominates(points[2].objectives, points[0].objectives, 0.1));

    const auto fronts = nonDominatedSort(points, 0.1);

    const std::vector<std::vector<size_t>> expected{ { 0, 1, 2 } };
    REQUIRE(fronts == expected);
}

TEST_CASE("crowdingDistances", "[Pareto]")
{
    const auto points = feasiblePoints({
        { 0.0, 1.0 },
        { 0.25, 0.75 },
        { 0.5, 0.5 },
        { 1.0, 0.0 },
    });

    SECTION("boundary points are infinite and interior points sum normalised neighbour gaps")
    {
        const std::vector<size_t> front{ 3, 1, 0, 2 };

        const auto distances = crowdingDistances(points, front);

        REQUIRE(distances.size() == 4);
        REQUIRE(distances[0] == infinity);
        REQUIRE(distances[1] == Catch::Approx(1.0));
        REQUIRE(distances[2] == infinity);
        REQUIRE(distances[3] == Catch::Approx(1.5));
    }

    SECTION("a front of one or two points is all boundary")
    {
        REQUIRE(crowdingDistances(points, std::vector<size_t>{ 2 }) == std::vector<double>{ infinity });
        REQUIRE(crowdingDistances(points, std::vector<size_t>{ 0, 3 }) ==
            std::vector<double>{ infinity, infinity });
    }
}

TEST_CASE("crowdingDistances ignores an objective with zero range", "[Pareto]")
{
    const auto points = feasiblePoints({ { 0.0, 0.5 }, { 0.5, 0.5 }, { 1.0, 0.5 } });
    const std::vector<size_t> front{ 0, 1, 2 };

    const auto distances = crowdingDistances(points, front);

    REQUIRE(distances[0] == infinity);
    REQUIRE(distances[1] == Catch::Approx(1.0));
    REQUIRE(distances[2] == infinity);
}

TEST_CASE("groupObjectives", "[Pareto]")
{
    const std::vector<double> partials{ 0.2, 0.4, 0.6, 0.8 };
    const std::vector<std::vector<size_t>> groups{ { 0, 2 }, { 1, 3 } };

    SECTION("equal weights give the plain mean of each group")
    {
        const std::vector<double> weights{ 0.25, 0.25, 0.25, 0.25 };

        const auto objectives = groupObjectives(partials, groups, weights);

        REQUIRE(objectives.size() == 2);
        REQUIRE(objectives[0] == Catch::Approx(0.4));
        REQUIRE(objectives[1] == Catch::Approx(0.6));
    }

    SECTION("unequal weights are renormalised within each group")
    {
        const std::vector<double> weights{ 0.1, 0.2, 0.3, 0.4 };

        const auto objectives = groupObjectives(partials, groups, weights);

        REQUIRE(objectives[0] == Catch::Approx((0.1 * 0.2 + 0.3 * 0.6) / 0.4));
        REQUIRE(objectives[1] == Catch::Approx((0.2 * 0.4 + 0.4 * 0.8) / 0.6));
    }

    SECTION("a group whose weights sum to zero falls back to the plain mean")
    {
        const std::vector<double> weights{ 0.0, 0.5, 0.0, 0.5 };

        const auto objectives = groupObjectives(partials, groups, weights);

        REQUIRE(objectives[0] == Catch::Approx(0.4));
    }

    SECTION("no groups means one objective per partial")
    {
        const std::vector<double> weights{ 0.1, 0.2, 0.3, 0.4 };

        REQUIRE(groupObjectives(partials, {}, weights) == partials);
    }
}

TEST_CASE("ParetoArchive accepts its first feasible entry", "[Pareto]")
{
    ParetoArchive archive(10);

    REQUIRE(archive.tryInsert(entryAt(1, { 0.5, 0.5 }), 0.0));
    REQUIRE(archive.size() == 1);
    REQUIRE(archive.members().front().solutionId == 1);
}

TEST_CASE("ParetoArchive rejects dominated and equal entries", "[Pareto]")
{
    ParetoArchive archive(10);
    REQUIRE(archive.tryInsert(entryAt(1, { 0.5, 0.5 }), 0.0));

    SECTION("dominated")
    {
        REQUIRE_FALSE(archive.tryInsert(entryAt(2, { 0.4, 0.5 }), 0.0));
    }

    SECTION("equal")
    {
        REQUIRE_FALSE(archive.tryInsert(entryAt(2, { 0.5, 0.5 }), 0.0));
    }

    SECTION("equal within epsilon")
    {
        REQUIRE_FALSE(archive.tryInsert(entryAt(2, { 0.505, 0.495 }), 0.01));
    }

    REQUIRE(memberIds(archive) == std::vector<int>{ 1 });
}

TEST_CASE("ParetoArchive evicts the members a new entry dominates", "[Pareto]")
{
    ParetoArchive archive(10);
    REQUIRE(archive.tryInsert(entryAt(1, { 0.5, 0.2 }), 0.0));
    REQUIRE(archive.tryInsert(entryAt(2, { 0.2, 0.5 }), 0.0));
    REQUIRE(archive.tryInsert(entryAt(3, { 0.0, 0.9 }), 0.0));

    REQUIRE(archive.tryInsert(entryAt(4, { 0.6, 0.6 }), 0.0));

    REQUIRE(memberIds(archive) == std::vector<int>{ 3, 4 });
}

TEST_CASE("ParetoArchive truncates at capacity by crowding and keeps boundary points", "[Pareto]")
{
    ParetoArchive archive(3);
    REQUIRE(archive.tryInsert(entryAt(1, { 0.0, 1.0 }), 0.0));
    REQUIRE(archive.tryInsert(entryAt(2, { 1.0, 0.0 }), 0.0));
    REQUIRE(archive.tryInsert(entryAt(3, { 0.6, 0.4 }), 0.0));

    SECTION("the most crowded existing member is evicted")
    {
        // crowding: (0.5, 0.5) -> 1.2, (0.6, 0.4) -> 1.0
        REQUIRE(archive.tryInsert(entryAt(4, { 0.5, 0.5 }), 0.0));
        REQUIRE(memberIds(archive) == std::vector<int>{ 1, 2, 4 });
    }

    SECTION("the newcomer is evicted when it is the most crowded")
    {
        // crowding: (0.65, 0.35) -> 0.8, (0.6, 0.4) -> 1.3
        REQUIRE_FALSE(archive.tryInsert(entryAt(4, { 0.65, 0.35 }), 0.0));
        REQUIRE(memberIds(archive) == std::vector<int>{ 1, 2, 3 });
    }
}

TEST_CASE("ParetoArchive never evicts a boundary point", "[Pareto]")
{
    ParetoArchive archive(1);
    REQUIRE(archive.tryInsert(entryAt(1, { 1.0, 0.0 }), 0.0));

    REQUIRE_FALSE(archive.tryInsert(entryAt(2, { 0.0, 1.0 }), 0.0));

    REQUIRE(memberIds(archive) == std::vector<int>{ 1 });
}

TEST_CASE("ParetoArchive rejects infeasible entries but records their violation", "[Pareto]")
{
    ParetoArchive archive(10);
    REQUIRE(archive.bestViolationSeen() == infinity);

    REQUIRE_FALSE(archive.tryInsert(entryAt(1, { 1.0, 1.0 }, 0.3), 0.0));
    REQUIRE_FALSE(archive.tryInsert(entryAt(2, { 1.0, 1.0 }, 0.1), 0.0));
    REQUIRE_FALSE(archive.tryInsert(entryAt(3, { 1.0, 1.0 }, 0.2), 0.0));

    REQUIRE(archive.size() == 0);
    REQUIRE(archive.bestViolationSeen() == Catch::Approx(0.1));

    REQUIRE(archive.tryInsert(entryAt(4, { 0.1, 0.1 }), 0.0));
    REQUIRE(archive.bestViolationSeen() == 0.0);
}
