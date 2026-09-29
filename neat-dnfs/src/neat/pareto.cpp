#include "neat/pareto.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace neat_dnfs
{
    namespace
    {
        constexpr double infinity = std::numeric_limits<double>::infinity();

        struct DominationGraph
        {
            std::vector<std::vector<size_t>> dominatedBy;
            std::vector<size_t> dominationCount;
        };

        bool isFeasible(const RankedPoint& point)
        {
            return point.violation <= 0.0;
        }

        bool equalWithin(std::span<const double> a, std::span<const double> b, const double epsilon)
        {
            return std::ranges::equal(a, b, [epsilon](const double x, const double y)
                {
                    return std::abs(x - y) <= epsilon;
                });
        }

        std::vector<size_t> indicesUpTo(const size_t count)
        {
            std::vector<size_t> indices(count);
            std::iota(indices.begin(), indices.end(), size_t{ 0 });
            return indices;
        }

        std::vector<RankedPoint> toRankedPoints(const std::vector<ParetoArchiveEntry>& entries)
        {
            std::vector<RankedPoint> points;
            points.reserve(entries.size());
            for (const auto& entry : entries)
            {
                points.push_back({ entry.objectives, entry.violation });
            }
            return points;
        }

        DominationGraph buildDominationGraph(std::span<const RankedPoint> points, const double epsilon)
        {
            const size_t count = points.size();
            DominationGraph graph{ std::vector<std::vector<size_t>>(count), std::vector<size_t>(count, 0) };
            for (size_t i = 0; i < count; ++i)
            {
                for (size_t j = i + 1; j < count; ++j)
                {
                    if (constrainedDominates(points[i], points[j], epsilon))
                    {
                        graph.dominatedBy[i].push_back(j);
                        ++graph.dominationCount[j];
                    }
                    else if (constrainedDominates(points[j], points[i], epsilon))
                    {
                        graph.dominatedBy[j].push_back(i);
                        ++graph.dominationCount[i];
                    }
                }
            }
            return graph;
        }

        // Moves every member of `remaining` whose domination count equals `count`
        // into the returned front, keeping ascending index order.
        std::vector<size_t> takeWithCount(std::vector<size_t>& remaining,
            const std::vector<size_t>& dominationCount, const size_t count)
        {
            std::vector<size_t> front;
            std::erase_if(remaining, [&](const size_t i)
                {
                    if (dominationCount[i] != count)
                    {
                        return false;
                    }
                    front.push_back(i);
                    return true;
                });
            return front;
        }

        // Normally the next front is the undominated remainder. Epsilon-dominance can
        // be cyclic, leaving no remaining point undominated; the least-dominated
        // remainder is taken instead so every point still gets a front.
        std::vector<size_t> takeNextFront(std::vector<size_t>& remaining, const std::vector<size_t>& dominationCount)
        {
            auto front = takeWithCount(remaining, dominationCount, 0);
            if (front.empty())
            {
                const size_t fewest = std::ranges::min(remaining, {}, [&](const size_t i) { return dominationCount[i]; });
                front = takeWithCount(remaining, dominationCount, dominationCount[fewest]);
            }
            return front;
        }
    }

    bool dominates(std::span<const double> a, std::span<const double> b, const double epsilon)
    {
        bool strictlyBetterSomewhere = false;
        for (size_t i = 0; i < a.size(); ++i)
        {
            if (a[i] < b[i] - epsilon)
            {
                return false;
            }
            if (a[i] > b[i] + epsilon)
            {
                strictlyBetterSomewhere = true;
            }
        }
        return strictlyBetterSomewhere;
    }

    bool constrainedDominates(const RankedPoint& a, const RankedPoint& b, const double epsilon)
    {
        const bool aFeasible = isFeasible(a);
        const bool bFeasible = isFeasible(b);
        if (aFeasible && bFeasible)
        {
            return dominates(a.objectives, b.objectives, epsilon);
        }
        if (aFeasible != bFeasible)
        {
            return aFeasible;
        }
        return a.violation < b.violation;
    }

    double constraintViolation(std::span<const double> partials, const double floor)
    {
        if (floor <= 0.0)
        {
            return 0.0;
        }
        return std::accumulate(partials.begin(), partials.end(), 0.0, [floor](const double sum, const double partial)
            {
                return sum + std::max(0.0, floor - partial);
            });
    }

    std::vector<std::vector<size_t>> nonDominatedSort(std::span<const RankedPoint> points, const double epsilon)
    {
        auto [dominatedBy, dominationCount] = buildDominationGraph(points, epsilon);
        auto remaining = indicesUpTo(points.size());
        std::vector<std::vector<size_t>> fronts;

        while (!remaining.empty())
        {
            auto front = takeNextFront(remaining, dominationCount);
            for (const size_t dominator : front)
            {
                for (const size_t dominated : dominatedBy[dominator])
                {
                    --dominationCount[dominated];
                }
            }
            fronts.push_back(std::move(front));
        }
        return fronts;
    }

    std::vector<double> crowdingDistances(std::span<const RankedPoint> points, std::span<const size_t> front)
    {
        const size_t size = front.size();
        if (size <= 2)
        {
            return std::vector<double>(size, infinity);
        }

        std::vector<double> distances(size, 0.0);
        const size_t objectiveCount = points[front[0]].objectives.size();

        for (size_t objective = 0; objective < objectiveCount; ++objective)
        {
            const auto valueAt = [&](const size_t position) { return points[front[position]].objectives[objective]; };
            auto order = indicesUpTo(size);
            std::ranges::stable_sort(order, {}, valueAt);

            const double range = valueAt(order.back()) - valueAt(order.front());
            if (range <= 0.0)
            {
                continue;
            }

            distances[order.front()] = infinity;
            distances[order.back()] = infinity;
            for (size_t k = 1; k + 1 < size; ++k)
            {
                distances[order[k]] += (valueAt(order[k + 1]) - valueAt(order[k - 1])) / range;
            }
        }
        return distances;
    }

    std::vector<double> groupObjectives(std::span<const double> partials,
        const std::vector<std::vector<size_t>>& groups, std::span<const double> weights)
    {
        if (groups.empty())
        {
            return { partials.begin(), partials.end() };
        }

        std::vector<double> objectives;
        objectives.reserve(groups.size());
        for (const auto& group : groups)
        {
            double weightSum = 0.0;
            double weightedSum = 0.0;
            double plainSum = 0.0;
            for (const size_t index : group)
            {
                weightSum += weights[index];
                weightedSum += weights[index] * partials[index];
                plainSum += partials[index];
            }
            objectives.push_back(weightSum > 0.0
                ? weightedSum / weightSum
                : plainSum / static_cast<double>(group.size()));
        }
        return objectives;
    }

    ParetoArchive::ParetoArchive(const size_t capacity)
        : capacity(capacity), lowestViolation(infinity)
    {
        if (capacity == 0)
        {
            throw std::invalid_argument("ParetoArchive: capacity must be at least 1.");
        }
    }

    bool ParetoArchive::tryInsert(const ParetoArchiveEntry& entry, const double epsilon)
    {
        lowestViolation = std::min(lowestViolation, entry.violation);
        if (entry.violation > 0.0 || isDominatedOrMatched(entry, epsilon))
        {
            return false;
        }

        removeDominatedBy(entry, epsilon);
        entries.push_back(entry);
        if (entries.size() <= capacity)
        {
            return true;
        }

        const size_t victim = evictionCandidate();
        entries.erase(entries.begin() + static_cast<std::ptrdiff_t>(victim));
        return victim != entries.size();
    }

    const std::vector<ParetoArchiveEntry>& ParetoArchive::members() const
    {
        return entries;
    }

    size_t ParetoArchive::size() const
    {
        return entries.size();
    }

    double ParetoArchive::bestViolationSeen() const
    {
        return lowestViolation;
    }

    bool ParetoArchive::isDominatedOrMatched(const ParetoArchiveEntry& entry, const double epsilon) const
    {
        return std::ranges::any_of(entries, [&](const ParetoArchiveEntry& member)
            {
                return dominates(member.objectives, entry.objectives, epsilon)
                    || equalWithin(member.objectives, entry.objectives, epsilon);
            });
    }

    void ParetoArchive::removeDominatedBy(const ParetoArchiveEntry& entry, const double epsilon)
    {
        std::erase_if(entries, [&](const ParetoArchiveEntry& member)
            {
                return dominates(entry.objectives, member.objectives, epsilon);
            });
    }

    size_t ParetoArchive::evictionCandidate() const
    {
        const auto points = toRankedPoints(entries);
        const auto distances = crowdingDistances(points, indicesUpTo(points.size()));

        const auto mostCrowded = std::ranges::min_element(distances);
        if (std::isinf(*mostCrowded))
        {
            return entries.size() - 1;
        }
        return static_cast<size_t>(std::distance(distances.begin(), mostCrowded));
    }
}
