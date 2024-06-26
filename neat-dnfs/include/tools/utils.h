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

namespace neat_dnfs
{
	namespace tools
	{
		namespace utils
		{
            inline int generateRandomInt(const int min, const int max) 
            {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<> dist(min, max);
                return dist(gen);
            }

            inline double generateRandomDouble(const double min, const double max) 
            {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<> dist(min, max);
                return dist(gen);
            }

            inline float generateRandomFloat(const float min, const float max) 
            {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<float> dist(min, max);
                return dist(gen);
            }

            inline double normalize(const double value, const double min, const double max)
			{
                if (value < min) return 0.0;
                if (value > max) return 1.0;
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

		}
	}
}