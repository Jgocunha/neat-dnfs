#pragma once

// <windows.h> (pulled in transitively via deps on this platform) leaks the
// max/min/ERROR macros, which collide with std::max/std::min calls in this
// header and with LogLevel::ERROR elsewhere in the codebase. Undefine them
// defensively wherever this header may be the first to include windows.h.
#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

#ifdef ERROR
#undef ERROR
#endif

#include <algorithm>
#include <iostream>
#include <random>
#include <cstdint>
#include <bit>
#include <thread>
#include <atomic>
#include <chrono>
#include <charconv>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace neat_dnfs
{
	namespace tools
	{
		namespace utils
		{
            /// @brief Source and target field gene ids, and innovation number if the
            /// name carried one, parsed out of a connection-gene coupling kernel's name.
            struct ParsedConnectionKernelName
            {
                int inFieldGeneId;
                int outFieldGeneId;
                std::optional<int> innovationNumber;
            };

            namespace detail
            {
                inline bool consumePrefix(std::string_view& remaining, const std::string_view prefix)
                {
                    if (!remaining.starts_with(prefix))
                    {
                        return false;
                    }
                    remaining.remove_prefix(prefix.size());
                    return true;
                }

                inline std::optional<int> parseInt(std::string_view& remaining)
                {
                    const auto start = remaining.data();
                    int value = 0;
                    const auto [parseEnd, error] =
                        std::from_chars(start, start + remaining.size(), value);
                    if (error != std::errc{})
                    {
                        return std::nullopt;
                    }
                    remaining.remove_prefix(static_cast<size_t>(parseEnd - start));
                    return value;
                }

                inline void consumeSpaces(std::string_view& remaining)
                {
                    while (remaining.starts_with(' '))
                    {
                        remaining.remove_prefix(1);
                    }
                }
            }

            /// @brief Parses a coupling kernel's element name back into the connection
            /// it describes.
            /// @details A connection-gene kernel's name is "<in>-<out>" or "<in> -
            /// <out>", optionally prefixed with "cg " and optionally followed by an
            /// innovation number, after the kernel type's namePrefix (e.g. "gk "/
            /// "gk cg "). A field's own self-kernel name ("gk 3") carries no dash and
            /// is rejected here rather than mistaken for a connection.
            /// @param name Full element name of a GAUSS_KERNEL or MEXICAN_HAT_KERNEL.
            /// @param namePrefix The kernel type's plain prefix (e.g. GaussKernelConstants::namePrefix).
            /// @param namePrefixConnectionGene The kernel type's connection-gene prefix
            /// (e.g. GaussKernelConstants::namePrefixConnectionGene).
            /// @return The parsed ids and optional innovation number, or nullopt if
            /// `name` is not a connection-gene kernel name (including a field's own
            /// self-kernel, or anything unparseable).
            [[nodiscard]] inline std::optional<ParsedConnectionKernelName> parseConnectionKernelName(
                const std::string_view name,
                const std::string_view namePrefix,
                const std::string_view namePrefixConnectionGene)
            {
                std::string_view remaining = name;
                if (!detail::consumePrefix(remaining, namePrefixConnectionGene) &&
                    !detail::consumePrefix(remaining, namePrefix))
                {
                    return std::nullopt;
                }

                const auto inFieldGeneId = detail::parseInt(remaining);
                if (!inFieldGeneId)
                {
                    return std::nullopt;
                }

                detail::consumeSpaces(remaining);
                if (!detail::consumePrefix(remaining, "-"))
                {
                    return std::nullopt;
                }
                detail::consumeSpaces(remaining);

                const auto outFieldGeneId = detail::parseInt(remaining);
                if (!outFieldGeneId)
                {
                    return std::nullopt;
                }

                detail::consumeSpaces(remaining);
                if (remaining.empty())
                {
                    return ParsedConnectionKernelName{ *inFieldGeneId, *outFieldGeneId, std::nullopt };
                }

                const auto innovationNumber = detail::parseInt(remaining);
                if (!innovationNumber || !remaining.empty())
                {
                    return std::nullopt;
                }

                return ParsedConnectionKernelName{ *inFieldGeneId, *outFieldGeneId, *innovationNumber };
            }

            // splitmix64: avalanches a single 64-bit seed into well-distributed state words.
            // Used only to seed xoshiro256++, which must not start from an all-zero state.
            inline std::uint64_t splitmix64(std::uint64_t& state)
            {
                std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
                z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
                z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
                return z ^ (z >> 31);
            }

            // xoshiro256++ (public domain, D. Blackman & S. Vigna). 32 bytes of state,
            // no dynamic allocation, no syscalls per draw: orders of magnitude faster
            // than reseeding std::mt19937 on every call.
            class Xoshiro256pp
            {
            public:
                using result_type = std::uint64_t;

                explicit Xoshiro256pp(const std::uint64_t seed)
                {
                    std::uint64_t sm = seed;
                    for (auto& word : s)
                        word = splitmix64(sm);
                }

                static constexpr result_type min() { return 0; }
                static constexpr result_type max() { return UINT64_MAX; }

                result_type operator()() noexcept
                {
                    const std::uint64_t result = std::rotl(s[0] + s[3], 23) + s[0];
                    const std::uint64_t t = s[1] << 17;

                    s[2] ^= s[0];
                    s[3] ^= s[1];
                    s[1] ^= s[2];
                    s[0] ^= s[3];
                    s[2] ^= t;
                    s[3] = std::rotl(s[3], 45);

                    return result;
                }

            private:
                std::uint64_t s[4]{};
            };

            // random_device::result_type is only 32 bits, and on some implementations
            // random_device degrades to a fixed, deterministic sequence. Mix in a
            // per-thread hash and a monotonically incrementing counter so that no two
            // threads can ever end up with the same seed, even in that degraded case.
            inline std::uint64_t makeThreadSeed()
            {
                static std::atomic<std::uint64_t> counter{ 0 };

                const std::uint64_t entropy = std::random_device{}();
                const std::uint64_t threadHash = std::hash<std::thread::id>{}(std::this_thread::get_id());
                const std::uint64_t time = static_cast<std::uint64_t>(
                    std::chrono::steady_clock::now().time_since_epoch().count());
                const std::uint64_t sequence = counter.fetch_add(1, std::memory_order_relaxed);

                return entropy ^ (threadHash + 0x9E3779B97F4A7C15ULL + (entropy << 6) + (entropy >> 2))
                    ^ time ^ (sequence * 0xBF58476D1CE4E5B9ULL);
            }

            // One engine per thread, constructed once and reused for every draw.
            inline Xoshiro256pp& engine()
            {
                thread_local Xoshiro256pp gen{ makeThreadSeed() };
                return gen;
            }

            inline int generateRandomInt(const int min, const int max)
            {
                std::uniform_int_distribution<int> dist(min, max);
                return dist(engine());
            }

            inline double generateRandomDouble(const double min, const double max)
            {
                std::uniform_real_distribution<double> dist(min, max);
                return dist(engine());
            }

            inline float generateRandomFloat(const float min, const float max)
            {
                std::uniform_real_distribution<float> dist(min, max);
                return dist(engine());
            }

            inline double normalize(const double value, const double min, const double max)
			{
				if (value < min)
				{
					return 0.0;
				}
				if (value > max)
				{
					return 1.0;
				}
				return (value - min) / (max - min);
			}

            inline double normalizeWithFlatheadGaussian(const double value, const double min, const double max, const double width)
			{
                const double center = (min + max) / 2;
                const double gaussian = exp(-0.5 * pow((value - center) / width, 2));
                const double flat_top = (value >= min && value <= max) ? 1.0 : 0.0;
            	return std::max(gaussian, flat_top);
			}

            inline double normalizeWithGaussian(const double value, const double target, const double width)
            {
	            return exp(-0.5 * pow((value - target) / width, 2));
            }

            /// @brief Score in [0,1] for how close a field's activation sits to its resting level.
            ///
            /// Two distances are measured and the larger one decides the score. The signed mean
            /// distance is not lifted by zero-mean noise, but a peak balanced by an inhibitory
            /// surround cancels out in it. The mean distance beyond @p noiseTolerance does not
            /// cancel, and ignores cells that are off rest by no more than noise.
            /// @param activation Activation of every cell of the field.
            /// @param restingLevel Resting level the cells are compared against.
            /// @param noiseTolerance Distance from rest a cell may have before it counts as off rest.
            /// @return 1 / (1 + the larger of the two distances).
            [[nodiscard]] inline double closenessOfActivationToRestingLevel(const std::span<const double> activation, const double restingLevel, const double noiseTolerance)
            {
                double signedDistance = 0.0;
                double distanceBeyondTolerance = 0.0;
                for (const double cell : activation)
                {
                    const double distance = cell - restingLevel;
                    signedDistance += distance;
                    distanceBeyondTolerance += std::max(std::abs(distance) - noiseTolerance, 0.0);
                }
                const auto cellCount = static_cast<double>(activation.size());
                const double deviation = std::max(std::abs(signedDistance / cellCount), distanceBeyondTolerance / cellCount);
                return 1.0 / (1.0 + deviation);
            }

            /// @brief Score in [0,1] for how far one item of a sequence came in its place.
            ///
            /// The first item is in place as far as the others did not come before it; every later item, as far as
            /// the items due before it came before it. An item that came at the same time as another, or one that
            /// never came, did not come before it.
            /// @param firstTimes When each item first came, in the order the items are due; empty for one that never
            /// came.
            /// @param index Position of the item in the sequence.
            /// @return 0 if the item never came, otherwise the share of the items, as above, in the right order with it.
            [[nodiscard]] inline double shareInPlace(const std::span<const std::optional<double>> firstTimes, const size_t index)
            {
                const auto& came = firstTimes[index];
                if (!came)
                {
                    return 0.0;
                }
                const auto cameBefore = [&came](const std::optional<double>& other) { return other && *other < *came; };
                if (index == 0)
                {
                    const auto others = firstTimes.subspan(1);
                    return others.empty() ? 1.0 : 1.0 - static_cast<double>(std::ranges::count_if(others, cameBefore))
                        / static_cast<double>(others.size());
                }
                return static_cast<double>(std::ranges::count_if(firstTimes.first(index), cameBefore))
                    / static_cast<double>(index);
            }

            /// @brief Middle value of @p values, unmoved by a minority of outliers such as a few peaks in a field.
            /// @param values Values to take the median of; at least one. Taken by value, since finding the median
            /// reorders them.
            /// @return The middle value, or the mean of the two middle values when there is an even number of them.
            [[nodiscard]] inline double median(std::vector<double> values)
            {
                const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
                std::ranges::nth_element(values, middle);
                if (values.size() % 2 == 1)
                {
                    return *middle;
                }
                const double lowerMiddle = *std::ranges::max_element(values.begin(), middle);
                return (lowerMiddle + *middle) / 2.0;
            }

            inline int generateRandomSignal()
            {
                std::uniform_int_distribution<int> dist(0, 1);
                return dist(engine()) ? 1 : -1; // Randomly selects -1 or 1 with equal probability
            }

		}
	}
}