#include "solutions/serial_order_schedule.h"

#include <format>
#include <utility>

#include "constants.h"

namespace neat_dnfs::serial_order_schedule
{
	std::vector<Stimulus> stimuli()
	{
		using dnf_composer::element::TimedGaussStimulusParameters;

		std::vector<std::pair<double, double>> cueWindows;
		for (const double cue : cueTimes)
		{
			cueWindows.emplace_back(cue, cue + pulseDuration);
		}
		std::vector<Stimulus> result{ { "tgs X", TimedGaussStimulusParameters{ cueWidth, GaussStimulusConstants::amplitude,
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
}
