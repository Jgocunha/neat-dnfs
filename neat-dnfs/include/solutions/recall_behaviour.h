#pragma once

#include <array>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "neat/solution.h"
#include "solutions/serial_order_schedule.h"

namespace neat_dnfs
{
	/// @brief A task scored by what its output does as it recalls the demonstrated sequence D, C, A, B: the scoring the
	/// serial-order and sequence-timing behaviour tasks share.
	///
	/// @details The input is "nf 1" and the output "nf 2". The derived task runs the schedule and samples the fields;
	/// this class scores what each sample shows:
	/// - the input peaks at each event, and is back at rest by 50 after each pulse: it detects events, it does not hold
	///   them;
	/// - during recall, an event earns its best recall score as far as it first peaks in its place: D as far as no other
	///   event peaked before it, each later event as far as the events due before it peaked before it. A peak counts only
	///   within a couple of samples of its event, in proportion to its height, in full from 2, as far as it is as narrow
	///   as a peak (full up to 30 wide, nothing from 60) and as far as it stays in place (its centre moving up to 1 per
	///   sample over two successive samples, nothing from 3). A sample's recall score is shared among the output's peaks
	///   at that moment;
	/// - once an event's peak ends, it stays released below threshold, graded from none 2 above it to all of it 2 below;
	/// - once the sequence is over, the output has no peak at any sample, counted as far as the recall was in order: an
	///   output that never peaked has nothing to stop.
	///
	/// A release counts only as far as the event was recalled in its place. With the guides scored, the fitness also pays
	/// what the reference designs hold below threshold: the stored and waiting events pre-shaped in order, and a trough
	/// at every released event.
	class RecallBehaviour : public Solution
	{
	public:
		/// @brief Number of events demonstrated and recalled: D, C, A and B.
		static constexpr size_t eventCount = serial_order_schedule::eventCount;
	protected:
		/// @brief Whether the fitness also pays the guides: what the reference design holds below threshold.
		enum class Guides
		{
			unscored, ///< The intended behaviour alone: peaks, and the lack of them.
			scored    ///< Also the pre-shapes of the events stored or still to come, and the troughs of those recalled.
		};

		/// @brief What one event did during one recall.
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
			/// @brief How far below threshold it lay at each sample since its release: none 2 above threshold, all of it
			/// 2 below.
			std::vector<double> belowThreshold;
			/// @brief How far below the rest of the field it lay at each sample since its release, in full from 2 below.
			std::vector<double> belowField;
			/// @brief Centre of the output's bump nearest the event at the last sample, if the output had a bump.
			std::optional<double> lastCentroid;
			/// @brief How far that bump had moved since the sample before.
			double lastMove = 0.0;
		};
		/// @brief What every event did during one recall, in sequence order.
		using CycleRecall = std::array<EventRecall, eventCount>;
		/// @brief What the output did in one recall from the end of the sequence on.
		struct AfterRecall
		{
			/// @brief When the sequence was over.
			std::optional<double> from;
			/// @brief How far the output had no peak, at each sample since.
			std::vector<double> quiet;
			/// @brief How far each event lay below the rest of the field, at each sample since; recorded only with the
			/// guides scored.
			std::array<std::vector<double>, eventCount> troughs;
		};

		/// @brief The input field's name.
		inline static const std::string inputField{ "nf 1" };
		/// @brief The output field's name.
		inline static const std::string outputField{ "nf 2" };
		/// @brief How long after a pulse ends a detection field's peak may take to decay before the input must be back at
		/// rest: five time constants of the reference's input.
		static constexpr double settleTime = 50.0;

		/// @brief Builds a task whose genome is created from @p topology.
		/// @param topology Fields to start from: the input and the output.
		/// @param guides Whether the fitness also pays the guides.
		RecallBehaviour(const SolutionTopology& topology, Guides guides);
		/// @brief Builds a task whose genome is decoded from an existing phenotype, such as a template.
		/// @param initialTopology Fields the task declares: the input and the output.
		/// @param phenotype Simulation the genome is translated from.
		/// @param guides Whether the fitness also pays the guides.
		RecallBehaviour(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype, Guides guides);

		/// @brief Whether the input should be back at rest at @p time: no pulse, X or event, started within a pulse and
		/// its settling time.
		/// @param time Simulation time of the sample.
		/// @return True if no pulse started within a pulse and its settling time before @p time.
		[[nodiscard]] virtual bool inputSettledAt(double time) const = 0;

		/// @brief Simulates until the simulation time reaches @p time.
		/// @param time Simulation time to stop at.
		void runUntil(double time);
		/// @brief Simulates until @p time, stopping every 10 to score the input at rest and to pass the sample on.
		/// @param time Simulation time to stop at.
		/// @param inputRest Input rest scores, one appended per sample where inputSettledAt() holds.
		/// @param onSample Called at each sample with its time and, where inputSettledAt() holds, the input's rest score.
		void sampleUntil(double time, std::vector<double>& inputRest,
			const std::function<void(double, std::optional<double>)>& onSample);
		/// @brief Scores the input's peak at an event as its pulse ends.
		/// @param position Position of the event.
		/// @return How far the input holds one bump at @p position, of about the reference's height and width.
		[[nodiscard]] double inputPeakedAt(double position) const;
		/// @brief Scores the output pre-shaped at @p positions, above the rest of the field and in that order.
		/// @param positions Event positions, highest pre-shape expected first.
		/// @return The mean pre-shape, with half of it kept only as far as the pre-shapes are in order.
		[[nodiscard]] double preShapedInOrder(const std::vector<double>& positions) const;
		/// @brief How fully a real recalled peak stands at @p position: near it, tall enough, and no wider than a peak.
		/// @param position Position of the event.
		/// @return How close the output's nearest bump lies to @p position, times that bump's height credit (full from 2)
		/// and its width credit (full up to 30, nothing from 60).
		[[nodiscard]] double recalledAt(double position) const;
		/// @brief How far the output's bump nearest an event moved since the last sample.
		/// @param lastCentroid Centre of the nearest bump at the last sample, if there was a bump.
		/// @param centroid Centre of the nearest bump now, if there is one.
		/// @return The distance between the two, around the field; 0 if either is missing or they lie more than 10 apart
		/// (a different bump: the events lie at least 40 apart).
		[[nodiscard]] static double movedSince(std::optional<double> lastCentroid, std::optional<double> centroid);
		/// @brief How far a bump stays in place, from its last two moves.
		/// @details A peak's centre can shift by several positions in the sample it ignites, as its first stretch above
		/// threshold widens to one side, while a bump travelling across the field moves at every sample. So only a move
		/// kept up over two successive samples counts.
		/// @param lastMove How far the bump moved over the sample before.
		/// @param move How far it moved since.
		/// @return 1 if the smaller of the two moves is up to 1, falling linearly to 0 at 3.
		[[nodiscard]] static double stayingInPlace(double lastMove, double move);
		/// @brief Samples the output during recall: updates each event's record, and scores the pre-shapes of the events
		/// not yet recalled.
		/// @param time Simulation time of the sample.
		/// @param recall Each event's record so far this recall, updated in place.
		/// @param waitingPreShapes Pre-shape scores of the events still waiting, one appended per sample where any wait.
		/// @param inputAtRest How far the input is at rest at this sample, or, inside a pulse and the 50 after it, at the
		/// last sample before: the share of the waiting pre-shapes that counts.
		void observeRecall(double time, CycleRecall& recall, std::vector<double>& waitingPreShapes,
			double inputAtRest) const;
		/// @brief Whether the recall's sequence is over: every event has peaked and been released.
		/// @param recall Each event's record so far this recall.
		/// @return True once every event's first peak has ended.
		[[nodiscard]] static bool sequenceReleased(const CycleRecall& recall);
		/// @brief Samples the output once the sequence is over: whether it has a peak, and with the guides scored, how
		/// far each event lies below the rest of the field.
		/// @param time Simulation time of the sample.
		/// @param after What the output did since the sequence was over, updated in place.
		void observeAfterRecall(double time, AfterRecall& after) const;
		/// @brief How far one event first peaked in its place in the recall.
		/// @param recall Each event's record for the recall.
		/// @param event Index of the event in the sequence.
		/// @return For D, the share of the other events that did not peak before it; for any later event, the share of
		/// the events due before it that peaked before it; 0 if it never peaked.
		[[nodiscard]] static double placeOf(const CycleRecall& recall, size_t event);
		/// @brief How far one event was recalled in its place in the recall.
		/// @param recall Each event's record for the recall.
		/// @param event Index of the event in the sequence.
		/// @return Its best recall score times its place.
		[[nodiscard]] static double creditInPlace(const CycleRecall& recall, size_t event);
		/// @brief Scores one recall by order.
		/// @param recall Each event's record for the recall.
		/// @return The mean, over the events, of each one's credit in place.
		[[nodiscard]] static double recallInOrder(const CycleRecall& recall);
		/// @brief How well one event stayed released since its peak ended.
		/// @param observed The event's record for the recall; it must have been released.
		/// @return Its mean score below threshold or, with the guides scored, half that and half its mean score below
		/// the field.
		[[nodiscard]] double releaseScore(const EventRecall& observed) const;
		/// @brief How well one event stayed released since its peak ended, scored with or without the guides whatever
		/// this task pays.
		/// @param observed The event's record for the recall; it must have been released.
		/// @param scoring Whether to score the guides.
		/// @return Its mean score below threshold or, with the guides scored, half that and half its mean score below
		/// the field.
		[[nodiscard]] static double releaseScore(const EventRecall& observed, Guides scoring);
		/// @brief Scores how each recalled event stayed released once its peak ended.
		/// @param recall Each event's record for the recall.
		/// @return The mean, over the events, of each one's credit in place times its release score, 0 if never
		/// released.
		[[nodiscard]] double releaseAfterRecall(const CycleRecall& recall) const;
		/// @brief Scores how each recalled event stayed released once its peak ended, with or without the guides
		/// whatever this task pays.
		/// @param recall Each event's record for the recall.
		/// @param scoring Whether to score the guides.
		/// @return The mean, over the events, of each one's credit in place times its release score, 0 if never
		/// released.
		[[nodiscard]] static double releaseAfterRecall(const CycleRecall& recall, Guides scoring);
		/// @brief Scores how the output stopped once the sequence was over: no peak at any later sample and, with the
		/// guides scored, every event below the rest of the field.
		/// @param recall Each event's record for the recall.
		/// @param after What the output did from the end of the sequence on.
		/// @return The mean of no peak over those samples, counted as far as the recall was in order; with the guides
		/// scored, half that and half for the troughs.
		[[nodiscard]] double stoppedAfterRecall(const CycleRecall& recall, const AfterRecall& after) const;
		/// @brief Scores how the output stopped once the sequence was over, with or without the guides whatever this task
		/// pays.
		/// @param recall Each event's record for the recall.
		/// @param after What the output did from the end of the sequence on; its troughs are recorded only by a task that
		/// pays the guides.
		/// @param scoring Whether to score the guides; only a task that pays them can.
		/// @return The mean of no peak over those samples, counted as far as the recall was in order; with the guides
		/// scored, half that and half for the troughs.
		[[nodiscard]] static double stoppedAfterRecall(const CycleRecall& recall, const AfterRecall& after, Guides scoring);
		/// @brief Scores the output pushed below the rest of the field at every event from the end of the sequence on,
		/// each event as far as it was recalled in its place.
		/// @param recall Each event's record for the recall.
		/// @param after What the output did from the end of the sequence on.
		/// @return The mean, over the events, of the mean trough score times credit in place.
		[[nodiscard]] static double troughsAfterRecall(const CycleRecall& recall, const AfterRecall& after);
		/// @brief Summarises one recall for the evaluation trace.
		/// @param cycle Number of the recall, from 1.
		/// @param recall Each event's record for the recall.
		/// @param after What the output did from the end of the sequence on.
		/// @return Each event's first peak, place, best recall score, recall and release score; and when the sequence was
		/// over and how far the output had no peak from then on.
		[[nodiscard]] nlohmann::json recallSummary(size_t cycle, const CycleRecall& recall, const AfterRecall& after) const;
		/// @brief Records what the input and output look like now, for the evaluation trace.
		/// @param stage Name of the point in the schedule being scored, such as "C shown" or "recall 2 A".
		/// @return The time, the input's bumps, and the output's bumps, median and activation at each event.
		[[nodiscard]] nlohmann::json snapshot(const std::string& stage) const;
		/// @brief Rounds a value written to the evaluation trace.
		/// @param value Value to round.
		/// @return @p value to two decimals, finer than anything the trace is read for.
		[[nodiscard]] static double rounded(double value);

		/// @brief Whether the fitness also pays the guides.
		Guides guides;
	};
}
