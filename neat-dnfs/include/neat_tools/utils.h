#pragma once

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

#include <iostream>
#include <random>
#include <cstdint>
#include <bit>

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

            // One engine per thread, constructed once and reused for every draw.
            inline Xoshiro256pp& engine()
            {
                thread_local Xoshiro256pp gen{ std::random_device{}() };
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