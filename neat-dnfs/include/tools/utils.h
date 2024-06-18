#pragma once

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
                if (value > max) return 0.0;
				return (value - min) / (max - min);
			}

		}
	}
}