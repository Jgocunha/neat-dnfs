#pragma once

#include <chrono>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifdef NEAT_DNFS_PROFILE
#include <mutex>
#include <unordered_map>
#endif

namespace neat_dnfs
{
	namespace tools
	{
		/// @brief Per-phase wall-clock timing for the evolution loop, active only
		/// when the project is built with @c -DNEAT_DNFS_PROFILE=ON.
		///
		/// All state lives in this header behind `#ifdef NEAT_DNFS_PROFILE`, so a
		/// default (profiling OFF) build carries no timing storage, no locking, and
		/// no clock reads: @c ScopedTimer becomes an empty RAII type and the query
		/// functions are constant-folded to their zero/empty answers.
		///
		/// Only ever populated from the main thread (see @c Population::evaluate --
		/// its parallel worker path is deliberately left uninstrumented so this
		/// accumulator never needs cross-thread synchronization beyond the mutex
		/// guarding accidental concurrent use).
		namespace profiler
		{
#ifdef NEAT_DNFS_PROFILE
			namespace detail
			{
				inline std::mutex& bucketsMutex()
				{
					static std::mutex mutex;
					return mutex;
				}

				inline std::unordered_map<std::string, double>& buckets()
				{
					static std::unordered_map<std::string, double> buckets;
					return buckets;
				}
			}
#endif

			/// @brief Clears every accumulated phase bucket.
			/// @details Call once per generation, before timing that generation's
			/// phases, so each generation's row reflects only its own work.
			inline void resetGeneration()
			{
#ifdef NEAT_DNFS_PROFILE
				const std::lock_guard<std::mutex> lock(detail::bucketsMutex());
				detail::buckets().clear();
#endif
			}

			/// @brief Total time accumulated under a phase name so far.
			/// @param phaseName Name of the phase, as passed to @c ScopedTimer.
			/// @return Seconds accumulated, or 0.0 if the phase was never timed
			/// or profiling is compiled out.
			[[nodiscard]] inline double elapsedSeconds(std::string_view phaseName)
			{
#ifdef NEAT_DNFS_PROFILE
				const std::lock_guard<std::mutex> lock(detail::bucketsMutex());
				const auto& buckets = detail::buckets();
				const auto it = buckets.find(std::string(phaseName));
				return it == buckets.end() ? 0.0 : it->second;
#else
				(void)phaseName;
				return 0.0;
#endif
			}

			/// @brief All accumulated phase buckets.
			/// @note Buckets are not necessarily disjoint: the evolution loop
			/// times "save" inside the "upkeep" scope, so "upkeep" is inclusive
			/// of "save" and the columns do not sum to the generation's total.
			/// @return Phase name/seconds pairs, in unspecified order. Empty if
			/// profiling is compiled out.
			[[nodiscard]] inline std::vector<std::pair<std::string, double>> snapshot()
			{
				std::vector<std::pair<std::string, double>> result;
#ifdef NEAT_DNFS_PROFILE
				const std::lock_guard<std::mutex> lock(detail::bucketsMutex());
				result.reserve(detail::buckets().size());
				for (const auto& [name, seconds] : detail::buckets())
				{
					result.emplace_back(name, seconds);
				}
#endif
				return result;
			}

#ifdef NEAT_DNFS_PROFILE
			/// @brief RAII stopwatch that adds its scope's elapsed time to the
			/// named phase bucket when it goes out of scope.
			class ScopedTimer
			{
			public:
				/// @brief Starts timing a phase.
				/// @param phaseName Name of the phase this scope's time is added to.
				explicit ScopedTimer(std::string_view phaseName)
					: phaseName(phaseName), start(std::chrono::steady_clock::now())
				{}

				/// @brief Adds the elapsed time since construction to the phase bucket.
				~ScopedTimer()
				{
					const auto end = std::chrono::steady_clock::now();
					const double seconds = std::chrono::duration<double>(end - start).count();
					const std::lock_guard<std::mutex> lock(detail::bucketsMutex());
					detail::buckets()[phaseName] += seconds;
				}

				ScopedTimer(const ScopedTimer&) = delete;
				ScopedTimer& operator=(const ScopedTimer&) = delete;
				ScopedTimer(ScopedTimer&&) = delete;
				ScopedTimer& operator=(ScopedTimer&&) = delete;

			private:
				std::string phaseName;
				std::chrono::steady_clock::time_point start;
			};
#else
			/// @brief No-op stand-in for the profiling build's @c ScopedTimer.
			/// @details Carries no members and its constructor takes no action, so
			/// it compiles to nothing rather than a disabled branch at each call site.
			class ScopedTimer
			{
			public:
				/// @brief No-op; profiling is compiled out.
				/// @param phaseName Unused.
				explicit ScopedTimer(std::string_view phaseName)
				{
					(void)phaseName;
				}

				ScopedTimer(const ScopedTimer&) = delete;
				ScopedTimer& operator=(const ScopedTimer&) = delete;
				ScopedTimer(ScopedTimer&&) = delete;
				ScopedTimer& operator=(ScopedTimer&&) = delete;
			};
#endif
		}
	}
}
