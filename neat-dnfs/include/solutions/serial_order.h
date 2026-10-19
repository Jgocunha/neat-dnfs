#pragma once

#include <array>

#include "solutions/sequence_timing.h"

namespace neat_dnfs
{
	/// @brief Evolves an architecture that recalls the serial order of a demonstrated event sequence, one event
	/// after another, every time a start cue is given.
	///
	/// @details Serial order on its own, without timing: the first half of the sequence-timing problem. The
	/// fields and events are those of SequenceTimingBoost, but the demonstration spaces them evenly, since only
	/// their order is recalled. The start event X is a homogeneous boost of the input field "nf 1", and D (180),
	/// C (120), A (20) and B (80) are Gaussian pulses at 200, 400, 600 and 800 after it.
	///
	/// Recall trial (one) -- after the demonstration and a gap drawn from 300-1500, X is given three times, each
	/// starting a recall cycle whose length is drawn afresh from 800-1500. After every X, the output field "nf 2"
	/// must form a peak at D, C, A and B in turn, each alone, and fall silent again before the cycle ends. The
	/// intervals between peaks are the architecture's own: only their order counts. Because the cycle lengths
	/// vary and X comes three times, the order must be recalled on every trigger, not replayed once at a fixed lag.
	///
	/// Probe trial (one) -- the demonstration and the same waiting time without any X. The output must stay silent.
	///
	/// The seven partial fitness terms are those of SequenceTiming, with each recall cycle scored like one recall
	/// trial, f5 giving order credit only (each consecutive pair rising one after another, fully at 100 or more
	/// apart), f6 also requiring each peak to be the only one in the output on the step it rises, so the previous
	/// peak must have gone first (SequenceTiming::oneAtATimeFromRiseScore), and f4 to f6 averaged over the three
	/// cycles. Each event's share of f4 to f6 grows with its peak's height, in full from a height of 2
	/// (SequenceTiming::peakHeightCredit), so an output that only flickers across threshold earns little of it.
	/// Each cycle's f4 to f6 are also scaled by how far the
	/// output's activity during that cycle lies near the event positions, beyond what activity spread evenly over
	/// the field would (SequenceTiming::activityAtEvents), so peaks sweeping across the whole field cannot earn
	/// recall credit by crossing the events by chance.
	class SerialOrder final : public SequenceTiming
	{
	public:
		/// @brief Number of recall cycles cued after the demonstration.
		static constexpr size_t recallCycleCount = 3;

		/// @brief Builds a task solution whose genome is created from @p topology.
		/// @param topology Fields to start from: the input and the output.
		explicit SerialOrder(const SolutionTopology& topology);
		/// @brief Builds a task solution whose genome is decoded from an existing phenotype, such as a template.
		/// @param initialTopology Fields the task declares: the input and the output.
		/// @param phenotype Simulation the genome is translated from.
		SerialOrder(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype);
		/// @brief Creates a solution with a fresh genome built from this one's initial topology and a new id.
		/// @return The new solution.
		SolutionPtr clone() const override;
		/// @brief Creates an independent solution with this one's phenotype and a new id.
		/// @return The new solution.
		SolutionPtr copy() const override;
	private:
		void testPhenotype() override;

		/// @brief Demonstrates the sequence, waits @p gap, then cues and scores one recall per cycle.
		/// @param gap Simulation time between the end of the demonstration and the first cue.
		/// @param cycleLengths Simulation time from each cue to the next, and from the last cue to the end.
		/// @param scores Accumulates every partial fitness term.
		/// @return The trial's trace: the demonstrated onsets, output silence before the first cue, and what each
		/// recall cycle did.
		nlohmann::json runCuedTrial(double gap, const std::array<double, recallCycleCount>& cycleLengths, TaskScores& scores);
		/// @brief Demonstrates the sequence, then waits as long as the cued trial without any cue.
		/// @param waitingTime Simulation time to wait after the demonstration.
		/// @param scores Accumulates the input and output-silence terms, and records the silence without a cue
		/// that scales recall credit.
		/// @return The trial's trace: the demonstrated onsets, and output silence during the demonstration and
		/// without a cue.
		nlohmann::json runUncuedTrial(double waitingTime, TaskScores& scores);
		/// @brief Draws the gap between the demonstration and the first cue.
		/// @return A gap in the gap range.
		[[nodiscard]] static double drawGap();
		/// @brief Draws each recall cycle's length independently.
		/// @return One length per cycle, each in the cycle-length range.
		[[nodiscard]] static std::array<double, recallCycleCount> drawCycleLengths();
	};
}
