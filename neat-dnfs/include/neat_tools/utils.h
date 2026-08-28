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

namespace neat_dnfs
{
	namespace tools
	{
		namespace utils
		{
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

            // Xoshiro256pp keeps only its state words, so the seed it was built from
            // would otherwise be lost. Tracked alongside the engine, per thread, so
            // getSeed() can report it even when the engine was never explicitly seeded.
            inline std::uint64_t& engineSeed()
            {
                thread_local std::uint64_t seed = makeThreadSeed();
                return seed;
            }

            // One engine per thread, constructed once and reused for every draw.
            inline Xoshiro256pp& engine()
            {
                thread_local Xoshiro256pp gen{ engineSeed() };
                return gen;
            }

            /// @brief Reseeds the calling thread's random engine, so subsequent draws
            /// from generateRandomInt/generateRandomDouble/generateRandomFloat/
            /// generateRandomSignal follow a reproducible sequence.
            ///
            /// Only reproduces single-threaded runs (Population::evaluate() with
            /// PopulationParameters::parallelEvolution = false) and tests that call
            /// this directly. It does NOT make parallel evolution reproducible:
            /// Population::evaluate() (src/neat/population.cpp) dispatches evaluation
            /// via std::async, so per-thread engines still advance in whatever order
            /// the OS schedules those tasks, and structural innovations are still
            /// registered in a nondeterministic order.
            /// @param seed Value the calling thread's engine is reseeded with.
            inline void setSeed(const std::uint64_t seed)
            {
                engineSeed() = seed;
                engine() = Xoshiro256pp{ seed };
            }

            /// @brief The seed the calling thread's random engine was constructed
            /// with -- either the value last passed to setSeed(), or, if setSeed()
            /// was never called on this thread, the value makeThreadSeed() produced
            /// when the engine was first used. Lets an unseeded run record what seed
            /// it happened to use.
            /// @return The calling thread's current engine seed.
            inline std::uint64_t getSeed()
            {
                return engineSeed();
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