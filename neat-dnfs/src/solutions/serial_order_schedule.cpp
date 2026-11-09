#include "solutions/serial_order_schedule.h"

#include <algorithm>
#include <format>
#include <utility>

#include "constants.h"

namespace neat_dnfs::serial_order_schedule
{
	std::vector<Stimulus> stimuli()
	{
		return stimuli(cueTimes);
	}

	std::vector<Stimulus> stimuli(const std::span<const double> cues)
	{
		using dnf_composer::element::TimedGaussStimulusParameters;

		std::vector<std::pair<double, double>> cueWindows;
		for (const double cue : cues)
		{
			cueWindows.emplace_back(cue, cue + pulseDuration);
		}
		std::vector<Stimulus> result{ { cueStimulusName, TimedGaussStimulusParameters{ cueWidth, GaussStimulusConstants::amplitude,
			DimensionConstants::xSize / 2.0, cueWindows, false, false } } };

		for (const auto& event : sequence)
		{
			result.push_back({ std::format("tgs {}", event.label), TimedGaussStimulusParameters{
				GaussStimulusConstants::width, GaussStimulusConstants::amplitude, event.position,
				{ { event.onset, event.onset + pulseDuration } }, GaussStimulusConstants::circularity,
				GaussStimulusConstants::normalization } });
		}
		return result;
	}

	size_t eventsShownBy(const double time)
	{
		return static_cast<size_t>(std::ranges::count_if(sequence, [time](const Event& event) { return event.onset < time; }));
	}

	std::vector<double> positionsOf(const size_t first, const size_t last)
	{
		std::vector<double> positions;
		for (size_t i = first; i < last; ++i)
		{
			positions.push_back(sequence[i].position);
		}
		return positions;
	}
}
