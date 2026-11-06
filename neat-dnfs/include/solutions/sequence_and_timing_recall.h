#pragma once

#include <array>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "neat/solution.h"

namespace neat_dnfs
{
	/// @brief The sequence-and-timing recall task: a demonstrated sequence of ten events, recalled in order and at the
	/// demonstrated times each time the memory recall trigger comes.
	///
	/// @details Inspired by Wojtak et al. (2023), "Adaptive timing in a dynamic field architecture for natural
	/// human-robot interactions". The input is "nf 1" and the output "nf 2", both 240 wide.
	///
	/// A demonstration starts from rest. The memory recall trigger, a boost of the whole input field, comes at 0-50;
	/// LA2B (40), RA3A (160), LA1A (60), RA1B (100), RA2B (120), LA3A (80), RA1A (140), LA1B (20), HOL (200) and HOS
	/// (220) follow as Gaussian pulses 50 long, at 50, 150, 350, 450, 650, 750, 950, 1050, 1900 and 2800. Each evaluation
	/// (the user, 6 November 2026) runs:
	/// - the probe: a demonstration, then 1500 with no trigger;
	/// - one demonstration, then the trigger five times in turn, each after a wait drawn afresh from 500-1500 on the
	///   sampling grid (the first from the end of the demonstration, each later one from the end of the recall window
	///   before it, 3300 after its trigger), then a tail of 1500 with no trigger.
	///
	/// A saved solution replays one demonstration and five triggers after waits of 500, 1500, 1000, 500 and 1500.
	///
	/// Every field is sampled every 10. An event first peaks at the first sample where its peak score, held over the
	/// samples before, reaches a quarter: one output bump near it, tall enough, as narrow as a peak and staying in place.
	/// Its credit is its best peak score (over three samples running, so a one-sample flash earns a third), if it came at
	/// its position: the k-th of the events that peaked, in time, and the k-th of them in the demonstration (the user,
	/// 6 November 2026: "did this one appear at the right order?"). Two events first peaking at one sample count as in
	/// order, three or more do not. The sequence is over 10 after every event's first peak has ended, or 3360 after the
	/// trigger (HOS's time and a fifth) if that never happens.
	///
	/// The eight terms, weighted by config/solutions/sequence-and-timing-recall.json:
	/// - f1, the input relays the demonstration: at each pulse's end, one input bump at the event and none elsewhere; no
	///   input bump between pulses;
	/// - f2, memory forms (a guide): through the demonstration and the first wait, while the input is settled, the shown
	///   events' output levels each below the one shown before it, as far as the output has no bump;
	/// - f3, the input quiet in recall: above threshold during each trigger, and with no bump from 100 after it to the
	///   next;
	/// - f4, order: the mean credit of the ten events, times the share of clean samples until the sequence is over (at
	///   most two events up, each bump at an event and no wider than a peak, and no released event peaking again);
	/// - f5, timing: the mean, over the events, of each one's credit times how close its first peak came to its time
	///   after the trigger (three quarters a Gaussian 0.2 of that time wide, one quarter the delay ratio);
	/// - f6, to-come (a guide): while the sequence runs, the waiting events' output levels each below the waiting event
	///   due before it; times f4;
	/// - f7, passed (a guide): each event below the output's far-from-events level once its first peak ends, until the
	///   sequence is over (at least 100); times its credit;
	/// - f8, after the recall: no output bump from the end of the sequence to the next trigger, or to the tail's end; the
	///   guided score adds the ten events' levels in descending order in the last 100 before it; times f4.
	///
	/// A level counts as below the one before it from 0.03 lower (the noise between places), fully from 0.08. f3-f8 are
	/// per recall and averaged over the five; f4-f8 are also multiplied by the gate: the share of samples with no output
	/// bump through the demonstration and first wait, the probe and the tail. The intended behaviour alone is f1, f3, f4,
	/// f5 and f8 without its guide, weighted by config/solutions/sequence-and-timing-recall-intended.json. The fitness is
	/// the larger of the two weighted sums, so a design that recalls on time without the guides is not ranked below one
	/// that has them. The target is reached when the fitness and every term of the larger weighted sum, each recall term at
	/// its worst recall, exceed it.
	///
	/// Every evaluation records a trace in SolutionParameters::evaluationTrace: both weighted sums; the gate and its three
	/// parts; for the probe, how long it was watched and how quiet the output stayed; for the run of triggers, each
	/// recall's wait, trigger, the memory before it, the share of clean samples, how many samples an event peaked again,
	/// the terms it earned and what each event did; the tail; and snapshots of the input and output as each event is shown
	/// and when each is due.
	class SequenceAndTimingRecall final : public Solution
	{
	public:
		/// @brief Number of events demonstrated and recalled.
		static constexpr size_t eventCount = 10;
		/// @brief How many samples running a peak is credited over: a peak up for fewer earns that share.
		static constexpr size_t peakSamples = 3;
		/// @brief How many times the trigger comes in turn after the one demonstration of an evaluation.
		static constexpr size_t recallTriggerCount = 5;

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
		/// @brief Whether the last evaluation reached the target: its fitness and every term of the weighted sum that was its
		/// fitness, each recall term at its worst recall, exceed it.
		/// @param targetFitness The level the fitness and each term must exceed.
		/// @return Whether the fitness and every such term exceed @p targetFitness.
		[[nodiscard]] bool reachedTarget(double targetFitness) const override;
	private:
		/// @brief What one event did during one recall.
		struct EventRecall
		{
			/// @brief Highest peak score over peakSamples samples running: one output bump near the event, tall enough,
			/// narrow and staying in place.
			double bestScore = 0.0;
			/// @brief Its peak scores at the last peakSamples samples, oldest first.
			std::array<double, peakSamples> lastScores{};
			/// @brief When the event first peaked, if it did.
			std::optional<double> firstPeak;
			/// @brief When its first peak ended, if it has.
			std::optional<double> releasedAt;
			/// @brief How deep a trough it lay in at each sample since its release, with the sample's time.
			std::vector<std::pair<double, double>> troughs;
			/// @brief The times it peaked again after its release, until the next trigger.
			std::vector<double> refiredAt;
			/// @brief Centre of the output's bump nearest the event at the last sample, if the output had a bump.
			std::optional<double> lastCentroid;
			/// @brief How far that bump had moved since the sample before.
			double lastMove = 0.0;
		};
		/// @brief What every event did during one recall, in sequence order.
		using Recall = std::array<EventRecall, eventCount>;
		/// @brief What the output did from one trigger to the next, sample by sample, each with its time.
		struct RecallSamples
		{
			/// @brief Each event's record.
			Recall events{};
			/// @brief Whether the sample was clean, in the recall window.
			std::vector<std::pair<double, bool>> clean;
			/// @brief How far the waiting events' levels fell in order, in the recall window, while two or more waited.
			std::vector<std::pair<double, double>> toCome;
			/// @brief How far the output had no bump.
			std::vector<std::pair<double, double>> quiet;
			/// @brief How far the ten events' levels fell in order, in the last 100 before the next trigger, as far as the
			/// output had no bump.
			std::vector<double> memoryBeforeNext;
			/// @brief Whether the input was above threshold during the trigger.
			bool inputTriggered = false;
			/// @brief Whether the input had no bump, from 100 after the trigger on.
			std::vector<double> inputQuiet;
		};
		/// @brief The terms one recall earned, before the gate, and the share of its clean samples.
		struct RecallTerms
		{
			double clean = 0.0;
			double input = 0.0;
			double order = 0.0;
			double timing = 0.0;
			double toCome = 0.0;
			double passed = 0.0;
			double afterIntended = 0.0;
			double afterGuided = 0.0;
		};
		/// @brief What a demonstration scored, up to the first trigger or the end of the probe.
		struct DemonstrationScores
		{
			/// @brief One score per pulse: one input bump at the event and none elsewhere.
			std::vector<double> inputPulses;
			/// @brief Whether the input had no bump, at each settled sample.
			std::vector<double> inputBetweenPulses;
			/// @brief The memory term's samples.
			std::vector<double> memory;
			/// @brief How far the output had no bump, at each sample.
			std::vector<double> quiet;
			/// @brief How far the ten events' levels fell in order in the last 100, as far as the output had no bump.
			std::vector<double> memoryAtEnd;
		};

		void testPhenotype() override;
		void createPhenotypeEnvironment() override;

		/// @brief The waits of the run of triggers in turn, each drawn afresh from 500-1500 on the sampling grid.
		/// @return Each trigger's wait, the first from the end of the demonstration, each later one from the end of the recall
		/// window before it.
		[[nodiscard]] std::vector<double> recallWaits() const;
		/// @brief Sets the trial's memory recall trigger: at 0-50 and again at each of @p cues.
		/// @param cues When the trial gives the trigger again: none for the probe, one for each recall of the run of
		/// triggers.
		void setCues(const std::vector<double>& cues);
		/// @brief Whether the input should be settled at @p time: no trigger or event pulse started within a pulse and its
		/// settling time.
		/// @param time Simulation time of the sample.
		/// @return True if no pulse started within a pulse and its settling time before @p time.
		[[nodiscard]] bool inputSettledAt(double time) const;
		/// @brief Simulates until the simulation time reaches @p time.
		/// @param time Simulation time to stop at.
		void runUntil(double time);
		/// @brief Simulates until @p time, stopping every 10 to pass the sample on.
		/// @param time Simulation time to stop at.
		/// @param onSample Called at each sample with its time.
		void sampleUntil(double time, const std::function<void(double)>& onSample);
		/// @brief Runs a trial from rest through its demonstration and up to @p until, scoring the input, the quiet output
		/// and its memory.
		/// @param until When to stop: the first trigger of the run of triggers, or the end of the demonstration for the
		/// probe.
		/// @param snapshots Receives a snapshot before the first event and as each event is shown.
		/// @return What the demonstration scored.
		DemonstrationScores demonstrate(double until, nlohmann::json& snapshots);
		/// @brief Runs the probe: a demonstration, then 1500 without the trigger.
		/// @param trials Receives the probe's trace.
		/// @return How far the output had no bump after the demonstration.
		double runProbe(nlohmann::json& trials);
		/// @brief Samples the output from the trigger at @p cue to @p following, recording what every term needs.
		/// @param cue When the trigger was given.
		/// @param following When the next trigger comes, or the tail ends.
		/// @param snapshots Receives a snapshot when each event is due and at the window's end.
		/// @return The recall's samples.
		RecallSamples sampleRecall(double cue, double following, nlohmann::json& snapshots);
		/// @brief Scores one recall from its samples.
		/// @param samples What the output did from the trigger to the next.
		/// @param cue When the trigger was given.
		/// @return The recall's terms, before the gate.
		[[nodiscard]] static RecallTerms scoreRecall(const RecallSamples& samples, double cue);

		/// @brief The output's level far from every event: its median over baselinePositions.
		/// @return The median activation there.
		[[nodiscard]] double baseline() const;
		/// @brief How far the output's levels at @p events fall in their order, highest first.
		/// @param events Indices of the events, in the order their levels should fall; at least two.
		/// @return descendingShare() of the levels, with a dead zone of the noise between places.
		[[nodiscard]] double levelsFallInOrder(const std::vector<size_t>& events) const;
		/// @brief Whether a field has a bump peaking near @p position, at least as tall as a first peak.
		/// @param fieldName Name of the neural field.
		/// @param position Field position of the event.
		/// @return True if a bump of the field lies near @p position and is tall enough.
		[[nodiscard]] bool bumpAt(const std::string& fieldName, double position) const;
		/// @brief Whether the output's bumps are clean at this sample: each at an event and no wider than a peak, at most two
		/// events up. Bumps are counted by the event they lie at: a peak being released can split into two flank bumps.
		/// @return True if the output's bumps are clean.
		[[nodiscard]] bool bumpsClean() const;
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
		/// @brief Samples the output during recall: updates each event's peak score, first peak, release, refiring and
		/// troughs.
		/// @param time Simulation time of the sample.
		/// @param recall Each event's record so far this recall, updated in place.
		/// @param inWindow Whether the sample lies in the recall window: only there does an event first peak or end its
		/// first peak; after it, a released event peaking again is still noted.
		void observeRecall(double time, Recall& recall, bool inWindow) const;
		/// @brief When the recall's sequence is over: 10 after every event's first peak ended, or 3360 after the trigger if
		/// that never happens.
		/// @param recall Each event's record for the recall.
		/// @param cue When the trigger was given.
		/// @return The time the sequence is over.
		[[nodiscard]] static double sequenceEndOf(const Recall& recall, double cue);
		/// @brief Whether one event came at its position in the recall: cameInPosition() of the first peaks.
		/// @param recall Each event's record for the recall.
		/// @param event Index of the event in the sequence.
		/// @return True if the event first peaked at its position.
		[[nodiscard]] static bool inPositionOf(const Recall& recall, size_t event);
		/// @brief One event's credit in the recall: its best peak score if it came at its position.
		/// @param recall Each event's record for the recall.
		/// @param event Index of the event in the sequence.
		/// @return Its best peak score, or 0 if it did not come at its position.
		[[nodiscard]] static double creditOf(const Recall& recall, size_t event);
		/// @brief How close one event's first peak came to its demonstrated time after the trigger.
		/// @param recall Each event's record for the recall.
		/// @param event Index of the event in the sequence.
		/// @param cue When the trigger was given.
		/// @return Three quarters a Gaussian of the error, 0.2 of the demonstrated time wide, and one quarter the smaller
		/// of the delay and the demonstrated time over the larger; 0 if the event never peaked.
		[[nodiscard]] static double onTime(const Recall& recall, size_t event, double cue);
		/// @brief How far one event lay in a trough once its first peak ended, until the sequence was over.
		/// @param observed The event's record for the recall.
		/// @param sequenceEnd When the sequence was over.
		/// @return Its mean trough from its release to the later of @p sequenceEnd and 100 after its release; 0 if it was
		/// never released.
		[[nodiscard]] static double passedOf(const EventRecall& observed, double sequenceEnd);
		/// @brief How many distinct samples any event peaked again after its release.
		/// @param recall Each event's record for the recall.
		/// @return The number of distinct sample times among every event's refiredAt.
		[[nodiscard]] static size_t refiredSamplesOf(const Recall& recall);
		/// @brief Summarises one recall for the evaluation trace.
		/// @param samples What the output did from the trigger to the next.
		/// @param cue When the trigger was given.
		/// @return Each event's first peak, first peak after the trigger, release, position, best peak score, timing credit
		/// and trough; and when the sequence was over and how far the output had no bump from then on.
		[[nodiscard]] static nlohmann::json recallSummary(const RecallSamples& samples, double cue);
		/// @brief Records what the input and output look like now, for the evaluation trace.
		/// @param stage Name of the point in the schedule being scored, such as "RA3A shown" or "recall HOL".
		/// @return The time, the input's bumps, and the output's bumps, baseline and activation at each event.
		[[nodiscard]] nlohmann::json snapshot(const std::string& stage) const;
		/// @brief Rounds a value written to the evaluation trace.
		/// @param value Value to round.
		/// @return @p value to two decimals, finer than anything the trace is read for.
		[[nodiscard]] static double rounded(double value);

		/// @brief When the current trial gives the trigger again: none in the probe.
		std::vector<double> trialCues;
		/// @brief The places farther than 10 from every event, where baseline() reads the output: set as each evaluation
		/// starts, from the configured field size.
		std::vector<double> baselinePositions;
		/// @brief Weights of the intended behaviour's five terms.
		std::vector<double> intendedFitnessWeights;
		/// @brief The terms the last evaluation's target is checked on: the larger weighted sum's, each recall term at its
		/// worst recall.
		std::vector<double> targetTerms;
	};
}
