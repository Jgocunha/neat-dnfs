#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include <dnf_composer/elements/timed_gauss_stimulus.h>

namespace neat_dnfs::serial_order_schedule
{
	/// @brief One demonstrated event: a Gaussian pulse on the input field.
	struct Event
	{
		/// @brief Name of the event, as its stimulus is named: "tgs <label>".
		const char* label;
		/// @brief Where the pulse is centred on the input field.
		double position;
		/// @brief When the pulse starts, after the demonstration's cue X.
		double onset;
	};

	/// @brief Number of events demonstrated and recalled in each cycle: D, C, A and B.
	inline constexpr std::size_t eventCount = 4;
	/// @brief D, C, A and B, demonstrated in this order, 100, 600, 800 and 1800 after X: recall must replay them in it.
	inline constexpr std::array<Event, eventCount> sequence{ { { "D", 180.0, 100.0 }, { "C", 120.0, 600.0 },
		{ "A", 20.0, 800.0 }, { "B", 80.0, 1800.0 } } };
	/// @brief How long each pulse, event or X, stays on.
	inline constexpr double pulseDuration = 50.0;
	/// @brief The demonstration's X, then one X starting each recall cycle: 1000, 800 and 1500 long.
	inline constexpr std::array<double, 4> cueTimes{ 0.0, 2550.0, 3550.0, 4350.0 };
	/// @brief When the trial ends, 1500 after the last X.
	inline constexpr double endTime = 5850.0;
	/// @brief Width of X's pulse: wide enough to lift the whole input field evenly, as a boost would, as a stimulus a
	/// saved file can replay.
	inline constexpr double cueWidth = 1e6;
	/// @brief How often the fields are sampled through the trial.
	inline constexpr double sampleInterval = 10.0;

	/// @brief One stimulus of the trial, on the input field.
	struct Stimulus
	{
		/// @brief Unique name of the stimulus element: "tgs X", or "tgs " and the event's label.
		std::string name;
		/// @brief Its width, amplitude, position and the windows it is on.
		dnf_composer::element::TimedGaussStimulusParameters parameters;
	};

	/// @brief The trial's stimuli: X at every cue, and each event's pulse, shaped by GaussStimulusConstants, X centred on
	/// the configured field.
	/// @return X, then D, C, A and B.
	[[nodiscard]] std::vector<Stimulus> stimuli();
}
