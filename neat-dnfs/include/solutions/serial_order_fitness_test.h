#pragma once

#include <array>
#include <functional>
#include <optional>
#include <vector>

#include <nlohmann/json.hpp>

#include "neat/solution.h"
#include "solutions/serial_order_schedule.h"

namespace neat_dnfs
{
	/// @brief The serial-order task, scored by how the fields respond at every stage, after a reference design.
	///
	/// @details The demonstration is a start cue X (a boost of the input field "nf 1"), then D (180), C (120), A (20)
	/// and B (80) as Gaussian pulses at 100, 600, 800 and 1800 after it. After a gap of 700, X is given three times,
	/// starting recall cycles of 1000, 800 and 1500. createPhenotypeEnvironment() lays that schedule down as timed
	/// stimuli, which the evaluation runs on and every saved solution carries, so a saved file replays the evaluation
	/// in dnf-composer. Every field is sampled every 10 through that schedule, and scored against what the hand-built
	/// design of 18 October 2026 (tests/fixtures/serial-order-fitness-test-reference.dnf) does there:
	/// - the input peaks at each event, and is back at rest by 50 after each pulse: it detects events, it does not hold
	///   them;
	/// - the output "nf 2" stays below threshold through the demonstration, pre-shaped at every stored event, the
	///   earliest highest. Pre-shapes count as memory only as far as the input is at rest and the output has no peak.
	/// - after each X the output peaks at D, C, A and B in turn, one at a time. Recall is sampled through the 600 after X
	///   and scored by order, not by when each peak comes. An event earns its best recall score in the cycle as far as it
	///   first peaks in its place: D as far as no other event peaked before it, each later event as far as the events
	///   due before it peaked before it. A peak counts only within a couple of samples of its event, and in proportion
	///   to its height, in full from 2. A sample's recall score is shared among the output's peaks at that moment.
	/// - While they wait, the events not yet recalled stay pre-shaped in order, counted as far as the input is at rest.
	///   A cycle where no event waits is not scored.
	/// - Once an event's peak ends, it stays released: half the credit for lying below threshold, graded from none 2
	///   above it to all of it 2 below; the other half for lying below the rest of the field, in full from 2 below.
	/// - once recall ends, the output has no peak and every event lies below the rest of the field. No peak counts only
	///   as far as the cycle recalled in order: an output that never peaked has nothing to stop.
	///
	/// A release and a trough count only as far as the event was recalled in its place.
	///
	/// The seven partial fitness terms are those stages: f1 the input, f2 the quiet output and f3 its pre-shapes
	/// during the demonstration, f4 the recall in order, f5 the pre-shapes still to come, f6 the release of each recalled
	/// event and f7 the output after recall.
	///
	/// Every evaluation also records a trace in SolutionParameters::evaluationTrace. At each point it is scored, the
	/// trace gives:
	/// - the input's bumps;
	/// - the output's bumps, its median activation and its activation at each event.
	///
	/// For each recall cycle it also gives each event's first peak, best recall score, recall and release.
	class SerialOrderFitnessTest final : public Solution
	{
	public:
		/// @brief Number of events demonstrated and recalled in each cycle: D, C, A and B.
		static constexpr size_t eventCount = serial_order_schedule::eventCount;

		/// @brief Builds a task solution whose genome is created from @p topology.
		/// @param topology Fields to start from: the input and the output.
		explicit SerialOrderFitnessTest(const SolutionTopology& topology);
		/// @brief Builds a task solution whose genome is decoded from an existing phenotype, such as a template.
		/// @param initialTopology Fields the task declares: the input and the output.
		/// @param phenotype Simulation the genome is translated from.
		SerialOrderFitnessTest(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype);
		/// @brief Creates a solution with a fresh genome built from this one's initial topology and a new id.
		/// @return The new solution.
		SolutionPtr clone() const override;
		/// @brief Creates an independent solution with this one's phenotype and a new id.
		/// @return The new solution.
		SolutionPtr copy() const override;
	private:
		void testPhenotype() override;
		void createPhenotypeEnvironment() override;

		/// @brief Simulates until the simulation time reaches @p time.
		/// @param time Simulation time to stop at.
		void runUntil(double time);
		/// @brief Simulates until @p time, stopping every 10 to score the input at rest and to pass the sample on.
		/// @param time Simulation time to stop at.
		/// @param inputRest Input rest scores, one appended per sample outside a pulse and the 50 after it.
		/// @param onSample Called at each sample with its time and, outside a pulse and the 50 after it, the input's
		/// rest score.
		void sampleUntil(double time, std::vector<double>& inputRest,
			const std::function<void(double, std::optional<double>)>& onSample);
		/// @brief Scores the output pre-shaped at @p positions, above the rest of the field and in that order.
		/// @param positions Event positions, highest pre-shape expected first.
		/// @return The mean pre-shape, with half of it kept only as far as the pre-shapes are in order.
		[[nodiscard]] double preShapedInOrder(const std::vector<double>& positions) const;
		/// @brief What one event did during one recall cycle.
		struct EventRecall
		{
			/// @brief Highest recall score: one output bump at the event, near it and tall enough, shared among the
			/// peaks up at that sample.
			double bestScore = 0.0;
			/// @brief Highest closeness times height credit of the output's nearest bump: how fully the event peaked.
			double recalled = 0.0;
			/// @brief When the event first peaked, if it did.
			std::optional<double> firstPeak;
			/// @brief Whether its first peak has ended.
			bool released = false;
			/// @brief Release score at each sample since: half for lying below threshold, graded across it, half for
			/// lying below the field.
			std::vector<double> releaseScores;
		};
		/// @brief What every event did during one recall cycle, in sequence order.
		using CycleRecall = std::array<EventRecall, eventCount>;

		/// @brief How fully a real recalled peak stands at @p position: near it, and tall enough.
		/// @param position Position of the event.
		/// @return How close the output's nearest bump lies to @p position, times that bump's height credit (full from 2).
		[[nodiscard]] double recalledAt(double position) const;
		/// @brief Samples the output during recall: updates each event's record, and scores the pre-shapes of the events
		/// not yet recalled.
		/// @param time Simulation time of the sample.
		/// @param recall Each event's record so far this cycle, updated in place.
		/// @param waitingPreShapes Pre-shape scores of the events still waiting, one appended per sample where any wait.
		/// @param inputAtRest How far the input is at rest at this sample, or, inside a pulse and the 50 after it, at the
		/// last sample before: the share of the waiting pre-shapes that counts.
		void observeRecall(double time, CycleRecall& recall, std::vector<double>& waitingPreShapes,
			double inputAtRest) const;
		/// @brief How far one event first peaked in its place in the cycle.
		/// @param recall Each event's record for the cycle.
		/// @param event Index of the event in the sequence.
		/// @return For D, the share of the other events that did not peak before it; for any later event, the share of
		/// the events due before it that peaked before it; 0 if it never peaked.
		[[nodiscard]] static double placeOf(const CycleRecall& recall, size_t event);
		/// @brief How far one event was recalled in its place in the cycle.
		/// @param recall Each event's record for the cycle.
		/// @param event Index of the event in the sequence.
		/// @return Its best recall score times its place.
		[[nodiscard]] static double creditInPlace(const CycleRecall& recall, size_t event);
		/// @brief Scores one cycle's recall by order.
		/// @param recall Each event's record for the cycle.
		/// @return The mean, over the events, of each one's credit in place.
		[[nodiscard]] static double recallInOrder(const CycleRecall& recall);
		/// @brief Scores how each recalled event stayed released once its peak ended.
		/// @param recall Each event's record for the cycle.
		/// @return The mean, over the events, of each one's credit in place times its mean release score, 0 if never
		/// released.
		[[nodiscard]] static double releaseAfterRecall(const CycleRecall& recall);
		/// @brief Scores how the output stopped once recall ended: no peak, and every event below the rest of the field.
		/// @param recall Each event's record for the cycle.
		/// @return Half for no peak, counted as far as the cycle recalled in order; half for the troughs.
		[[nodiscard]] double stoppedAfterRecall(const CycleRecall& recall) const;
		/// @brief Scores the output pushed below the rest of the field now at every event, each as far as it was recalled
		/// in its place.
		/// @param recall Each event's record for the cycle.
		/// @return The mean, over the events, of trough score times credit in place.
		[[nodiscard]] double troughsAfterRecall(const CycleRecall& recall) const;
		/// @brief Summarises one cycle's recall for the evaluation trace.
		/// @param cycle Number of the recall cycle, from 1.
		/// @param recall Each event's record for the cycle.
		/// @return Each event's first peak, place, best recall score, recall and mean release.
		[[nodiscard]] static nlohmann::json recallSummary(size_t cycle, const CycleRecall& recall);
		/// @brief Records what the input and output look like now, for the evaluation trace.
		/// @param stage Name of the point in the schedule being scored, such as "C shown" or "recall 2 A".
		/// @return The time, the input's bumps, and the output's bumps, median and activation at each event.
		[[nodiscard]] nlohmann::json snapshot(const std::string& stage) const;
	};
}
