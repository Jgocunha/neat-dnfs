#pragma once

#include <array>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "neat/solution.h"

namespace neat_dnfs
{
	/// @brief The sequence-and-timing recall task: a demonstrated sequence of ten events, recalled in order and at the
	/// demonstrated times once the memory recall trigger comes.
	///
	/// @details Inspired by Wojtak et al. (2023), "Adaptive timing in a dynamic field architecture for natural
	/// human-robot interactions". The input is "nf 1" and the output "nf 2", both 240 wide.
	///
	/// A trial starts from rest. The memory recall trigger, a boost of the whole input field, comes at 0-50; LA2B (40),
	/// RA3A (160), LA1A (60), RA1B (100), RA2B (120), LA3A (80), RA1A (140), LA1B (20), HOL (200) and HOS (220) follow
	/// as Gaussian pulses 50 long, at 50, 150, 350, 450, 650, 750, 950, 1050, 1900 and 2800. A recall trial gives the
	/// trigger again after a wait, at 2850 plus the wait, and records the 3300 after it; the probe trial never gives it,
	/// and lasts as long as the evaluation's longest recall trial. Each evaluation runs the probe, then three recall
	/// trials, with waits drawn afresh from each third of 300-1500 and rounded to the sampling interval, then one
	/// consecutive trial: one demonstration, then the trigger three times, with waits drawn afresh from each third in a
	/// shuffled order, the first from the end of the demonstration and each later one from the end of the previous
	/// recall window. Its three recalls are scored as the recall trials' are, so a design that recalls once and cannot
	/// recall again loses their credit. In the recall terms each recall right after a demonstration (the three recall
	/// trials' and the consecutive trial's first) counts twice the consecutive trial's second or third. The schedule is laid down as timed stimuli, so a saved solution replays a recall
	/// trial with a wait of 900 in dnf-composer.
	///
	/// Every field is sampled every 10 and scored against what the reference design
	/// (tests/fixtures/sequence-and-timing-recall-reference.dnf) does there:
	/// - the input detects each event, and is back at rest by 50 after each pulse;
	/// - the output stays below threshold through each demonstration and wait;
	/// - after the trigger the output peaks at each event, first peaking in its place: LA2B as far as no other event
	///   peaked before it, each later event as far as the events due before it peaked before it. Overlap is allowed: an
	///   event's peak score is its own, however many peaks are up with it. A peak scores as far as it lies near the event,
	///   in proportion to its height (full from 2), as far as it is as narrow as a peak (full up to 30 wide, nothing from
	///   60) and as far as it stays in place;
	/// - each at the demonstrated time after the trigger, credited three quarters as a Gaussian 0.2 of that time wide
	///   (scalar timing) and one quarter as the smaller of the delay and that time over the larger;
	/// - once an event's peak ends, it stays released below threshold;
	/// - once the sequence is over, the output has no peak until the recall ends. The sequence is over once every event
	///   has peaked and been released, or 300 after HOS is due if that never happens.
	///
	/// The guides pay what the reference holds below threshold, measured from the output's level far from every event
	/// (its median over the places more than 10 from all of them), which the troughs of a recall do not lower:
	/// - through each demonstration and wait, the stored events pre-shaped, the earliest highest, as far as the input is
	///   at rest and the output has no peak;
	/// - while they wait, the events not yet recalled pre-shaped in order, as far as the input is at rest;
	/// - a released event earns half its release for lying below threshold and half for a trough;
	/// - once the sequence is over, half the stop is for a trough at every event, each as far as it was recalled in its
	///   place.
	///
	/// Each recall's credit is scaled by how far the output had no peak before its trigger (since the trial began, or
	/// since the previous recall window ended), and all of it by how far the output had no peak in the probe trial after
	/// its demonstration, and by how far no event peaked there.
	///
	/// The eight partial fitness terms are: f1 the input, f2 the quiet output and f3 its pre-shapes before the trigger,
	/// f4 the recall in order, f5 its timing, f6 the pre-shapes still to come, f7 the release of each recalled event and
	/// f8 the output after recall, weighted by config/solutions/sequence-and-timing-recall.json. The intended behaviour
	/// alone is six of them, f1, f2, f4, f5, and f7 and f8 without the guides, weighted by
	/// config/solutions/sequence-and-timing-recall-intended.json. The fitness is the larger of the two weighted sums, so a
	/// design that recalls on time without the guides is not ranked below one that has them.
	///
	/// Every evaluation records a trace in SolutionParameters::evaluationTrace: both weighted sums; per trial, its kind,
	/// wait and how quiet the output stayed before the trigger; for the probe trial, how far each event peaked without
	/// the trigger; for each recall, what each event did and its timing credit, and snapshots of the input and output
	/// as each event is shown and when each is due. The consecutive trial lists its three recalls, each with its wait,
	/// trigger and how quiet the output stayed before it.
	class SequenceAndTimingRecall final : public Solution
	{
	public:
		/// @brief Number of events demonstrated and recalled.
		static constexpr size_t eventCount = 10;

		/// @brief Builds a task solution whose genome is created from @p topology.
		/// @param topology Fields to start from: the input and the output.
		explicit SequenceAndTimingRecall(const SolutionTopology& topology);
		/// @brief Builds a task solution whose genome is decoded from an existing phenotype, such as a template.
		/// @param initialTopology Fields the task declares: the input and the output.
		/// @param phenotype Simulation the genome is translated from.
		SequenceAndTimingRecall(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype);
		/// @brief Creates a solution with a fresh genome built from this one's initial topology and a new id.
		/// @return The new solution.
		SolutionPtr clone() const override;
		/// @brief Creates an independent solution with this one's phenotype and a new id.
		/// @return The new solution.
		SolutionPtr copy() const override;
	private:
		/// @brief What one event did during one recall.
		struct EventRecall
		{
			/// @brief Highest peak score: one output bump near the event, tall enough, narrow and staying in place.
			double bestScore = 0.0;
			/// @brief When the event first peaked, if it did.
			std::optional<double> firstPeak;
			/// @brief When its first peak ended, if it has.
			std::optional<double> releasedAt;
			/// @brief How far below threshold it lay at each sample since its release.
			std::vector<double> belowThreshold;
			/// @brief How deep a trough it lay in at each sample since its release.
			std::vector<double> troughs;
			/// @brief Centre of the output's bump nearest the event at the last sample, if the output had a bump.
			std::optional<double> lastCentroid;
			/// @brief How far that bump had moved since the sample before.
			double lastMove = 0.0;
		};
		/// @brief What every event did during one recall, in sequence order.
		using Recall = std::array<EventRecall, eventCount>;
		/// @brief What the output did in one recall from the end of the sequence on.
		struct AfterRecall
		{
			/// @brief When the sequence was over.
			std::optional<double> from;
			/// @brief How far the output had no peak, at each sample since.
			std::vector<double> quiet;
			/// @brief How deep a trough each event lay in, at each sample since.
			std::array<std::vector<double>, eventCount> troughs;
		};
		/// @brief What every trial of an evaluation scored, gathered for the partial fitness terms.
		struct Scores
		{
			std::vector<double> inputPeaks;
			std::vector<double> inputRest;
			std::vector<double> outputQuiet;
			std::vector<double> outputMemory;
			std::vector<double> recalled;
			std::vector<double> timing;
			std::vector<double> toCome;
			std::vector<double> released;
			std::vector<double> stopped;
			std::vector<double> releasedWithoutGuides;
			std::vector<double> stoppedWithoutGuides;
			/// @brief Each recall's weight in the recall terms, in the order the recalls ran.
			std::vector<double> recallWeights;
			/// @brief The weight of each recall that had events waiting to score in toCome.
			std::vector<double> toComeWeights;
		};

		void testPhenotype() override;
		void createPhenotypeEnvironment() override;

		/// @brief Draws the evaluation's three recall waits, one from each third of 300-1500, rounded to the sampling
		/// interval so that the trigger falls on a sample.
		/// @return The waits, shortest third first.
		[[nodiscard]] static std::array<double, 3> drawRecallWaits();
		/// @brief Sets the trial's memory recall trigger: at 0-50 and again at each of @p cues.
		/// @param cues When the trial gives the trigger again: none for the probe trial, one for a recall trial, three for
		/// the consecutive trial.
		void setCues(const std::vector<double>& cues);
		/// @brief Whether the input should be back at rest at @p time: no trigger or event pulse started within a pulse
		/// and its settling time.
		/// @param time Simulation time of the sample.
		/// @return True if no pulse started within a pulse and its settling time before @p time.
		[[nodiscard]] bool inputSettledAt(double time) const;
		/// @brief Simulates until the simulation time reaches @p time.
		/// @param time Simulation time to stop at.
		void runUntil(double time);
		/// @brief Simulates until @p time, stopping every 10 to score the input at rest and to pass the sample on.
		/// @param time Simulation time to stop at.
		/// @param inputRest Input rest scores, one appended per sample where inputSettledAt() holds.
		/// @param onSample Called at each sample with its time and, where inputSettledAt() holds, the input's rest score.
		void sampleUntil(double time, std::vector<double>& inputRest,
			const std::function<void(double, std::optional<double>)>& onSample);
		/// @brief Runs a trial from rest through its demonstration and up to @p until, scoring the input, the quiet output
		/// and its memory.
		/// @param until When to stop: the trigger of a recall trial, or the end of the demonstration in the probe trial.
		/// @param scores Accumulates the input, quiet output and memory terms.
		/// @param snapshots Receives a snapshot before the first event and as each event is shown.
		/// @return How far the output had no peak over the demonstration and the wait.
		double demonstrate(double until, Scores& scores, nlohmann::json& snapshots);
		/// @brief Runs the probe trial: a demonstration, then the wait and a recall's length without the trigger.
		/// @param wait How long a recall trial would wait before the trigger: the evaluation's longest.
		/// @param scores Accumulates the input and quiet output terms.
		/// @param trials Receives the trial's trace.
		/// @return How far the output had no peak after the demonstration, times the mean, over the events, of how far
		/// each did not peak there (1 minus its best peak score).
		double runProbeTrial(double wait, Scores& scores, nlohmann::json& trials);
		/// @brief Runs one recall trial and scores its recall.
		/// @param wait How long the trial waits after the demonstration before the trigger.
		/// @param scores Accumulates every term; the recall terms are scaled by how far the output had no peak before the
		/// trigger.
		/// @param trials Receives the trial's trace.
		void runRecallTrial(double wait, Scores& scores, nlohmann::json& trials);
		/// @brief Runs the consecutive trial: one demonstration, then a recall after each of @p waits, and scores each
		/// recall as a recall trial's.
		/// @param waits The waits, the first from the end of the demonstration and each later one from the end of the
		/// previous recall window.
		/// @param scores Accumulates every term; each recall's terms are scaled by how far the output had no peak since
		/// the trial began, for the first, or since the previous recall window ended.
		/// @param trials Receives the trial's trace.
		void runConsecutiveTrial(const std::array<double, 3>& waits, Scores& scores, nlohmann::json& trials);
		/// @brief Samples the recall window after the trigger at @p cue and scores the recall.
		/// @param cue When the trigger was given.
		/// @param quietBeforeCue How far the output had no peak before the trigger: every recall term is scaled by it.
		/// @param weight The recall's weight in the recall terms: 2 right after a demonstration, 1 for a later recall of
		/// the consecutive trial.
		/// @param scores Accumulates the input rest and the recall terms.
		/// @param snapshots Receives a snapshot when each event is due and at the window's end.
		/// @return The recall's summary for the trace (recallSummary()).
		nlohmann::json recallAfterCue(double cue, double quietBeforeCue, double weight, Scores& scores,
			nlohmann::json& snapshots);

		/// @brief The output's level far from every event: its median over baselinePositions.
		/// @return The median activation there.
		[[nodiscard]] double baseline() const;
		/// @brief Scores the output pre-shaped at the events at @p positions, above @p level and in that order.
		/// @param positions Event positions, highest pre-shape expected first.
		/// @param level The output's baseline at this sample.
		/// @return The mean pre-shape, with half of it kept only as far as the pre-shapes are in order.
		[[nodiscard]] double preShapedInOrder(const std::vector<double>& positions, double level) const;
		/// @brief How fully a peak stands at @p position: near it, tall enough, and no wider than a peak.
		/// @param position Position of the event.
		/// @return How close the output's nearest bump lies to @p position, times that bump's height credit and its width
		/// credit.
		[[nodiscard]] double peakAt(double position) const;
		/// @brief How far the output's bump nearest an event moved since the last sample.
		/// @param lastCentroid Centre of the nearest bump at the last sample, if there was a bump.
		/// @param centroid Centre of the nearest bump now, if there is one.
		/// @return The distance between the two, around the field; 0 if either is missing or they lie more than 10 apart
		/// (a different bump: the events lie 20 apart).
		[[nodiscard]] static double movedSince(std::optional<double> lastCentroid, std::optional<double> centroid);
		/// @brief How far a bump stays in place, from its last two moves: a peak's centre can shift as it ignites, while
		/// a bump travelling across the field moves at every sample.
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
		void observeRecall(double time, Recall& recall, std::vector<double>& waitingPreShapes, double inputAtRest) const;
		/// @brief Samples the output once the sequence is over: whether it has a peak, and how deep a trough each event
		/// lies in.
		/// @param time Simulation time of the sample.
		/// @param after What the output did since the sequence was over, updated in place.
		void observeAfterRecall(double time, AfterRecall& after) const;
		/// @brief Whether the recall's sequence is over: every event has peaked and been released.
		/// @param recall Each event's record so far this recall.
		/// @return True once every event's first peak has ended.
		[[nodiscard]] static bool sequenceReleased(const Recall& recall);
		/// @brief How far one event first peaked in its place in the recall.
		/// @param recall Each event's record for the recall.
		/// @param event Index of the event in the sequence.
		/// @return For LA2B, the share of the other events that did not peak before it; for any later event, the share of
		/// the events due before it that peaked before it; 0 if it never peaked.
		[[nodiscard]] static double placeOf(const Recall& recall, size_t event);
		/// @brief How far one event was recalled in its place in the recall.
		/// @param recall Each event's record for the recall.
		/// @param event Index of the event in the sequence.
		/// @return Its best peak score times its place.
		[[nodiscard]] static double creditInPlace(const Recall& recall, size_t event);
		/// @brief Scores one recall by order.
		/// @param recall Each event's record for the recall.
		/// @return The mean, over the events, of each one's credit in place.
		[[nodiscard]] static double recallInOrder(const Recall& recall);
		/// @brief How close one event's first peak came to its demonstrated time after the trigger.
		/// @param recall Each event's record for the recall.
		/// @param event Index of the event in the sequence.
		/// @param cue When the trigger was given.
		/// @return Three quarters a Gaussian of the error, 0.2 of the demonstrated time wide, and one quarter the smaller
		/// of the delay and the demonstrated time over the larger; 0 if the event never peaked.
		[[nodiscard]] static double onTime(const Recall& recall, size_t event, double cue);
		/// @brief Scores one recall's timing.
		/// @param recall Each event's record for the recall.
		/// @param cue When the trigger was given.
		/// @return The mean, over the events, of each one's credit in place times how close its first peak came to its
		/// demonstrated time.
		[[nodiscard]] static double timingOf(const Recall& recall, double cue);
		/// @brief Scores how each recalled event stayed released once its peak ended.
		/// @param recall Each event's record for the recall.
		/// @param withGuides Whether half of each release is for a trough.
		/// @return The mean, over the events, of each one's credit in place times its mean score below threshold or,
		/// with the guides, half that and half its mean trough; 0 for an event never released.
		[[nodiscard]] static double releaseAfterRecall(const Recall& recall, bool withGuides);
		/// @brief Scores how the output stopped once the sequence was over.
		/// @param recall Each event's record for the recall.
		/// @param after What the output did from the end of the sequence on.
		/// @param withGuides Whether half of the stop is for a trough at every event.
		/// @return The mean of no peak over those samples, counted as far as the recall was in order; with the guides,
		/// half that and half the mean, over the events, of each one's mean trough times its credit in place.
		[[nodiscard]] static double stoppedAfterRecall(const Recall& recall, const AfterRecall& after, bool withGuides);
		/// @brief Summarises one recall for the evaluation trace.
		/// @param recall Each event's record for the recall.
		/// @param after What the output did from the end of the sequence on.
		/// @param cue When the trigger was given.
		/// @return Each event's first peak, first peak after the trigger, release, place, best peak score, timing credit
		/// and release score; and when the sequence was over and how far the output had no peak from then on.
		[[nodiscard]] nlohmann::json recallSummary(const Recall& recall, const AfterRecall& after, double cue) const;
		/// @brief Records what the input and output look like now, for the evaluation trace.
		/// @param stage Name of the point in the schedule being scored, such as "RA3A shown" or "recall HOL".
		/// @return The time, the input's bumps, and the output's bumps, baseline and activation at each event.
		[[nodiscard]] nlohmann::json snapshot(const std::string& stage) const;
		/// @brief Rounds a value written to the evaluation trace.
		/// @param value Value to round.
		/// @return @p value to two decimals, finer than anything the trace is read for.
		[[nodiscard]] static double rounded(double value);

		/// @brief When the current trial gives the trigger again: none in the probe trial.
		std::vector<double> trialCues;
		/// @brief The places farther than 10 from every event, where baseline() reads the output: set as each evaluation
		/// starts, from the configured field size.
		std::vector<double> baselinePositions;
		/// @brief Weights of the intended behaviour's six terms.
		std::vector<double> intendedFitnessWeights;
	};
}
