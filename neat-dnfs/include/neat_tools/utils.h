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

#include <iostream>
#include <random>
#include <cstdint>
#include <bit>
#include <thread>
#include <atomic>
#include <chrono>
#include <charconv>
#include <optional>
#include <string_view>

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

            inline int generateRandomSignal()
            {
                std::uniform_int_distribution<int> dist(0, 1);
                return dist(engine()) ? 1 : -1; // Randomly selects -1 or 1 with equal probability
            }

		}
	}
}