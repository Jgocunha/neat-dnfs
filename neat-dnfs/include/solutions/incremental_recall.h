#pragma once

#include <array>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "neat/solution.h"

namespace neat_dnfs
{
	/// @brief The stages of the incremental-evolution test, each a step of more complex recall behaviour. The best design
	/// evolved at one stage seeds the next.
	enum class RecallStage
	{
		/// The output recalls the demonstrated order once: one peak at a time at each event, in order.
		singleOrder,
		/// The output recalls the demonstrated order once, each event at its demonstrated time after the trigger.
		singleOrderAndTiming,
		/// The output recalls the demonstrated order after a short wait or a long one: its memories hold.
		repeatableOrder,
		/// The output recalls the demonstrated order after a short wait or a long one, each event at its time.
		repeatableOrderAndTiming,
		/// The output recalls the demonstrated order each time the trigger comes again in one run.
		consecutiveOrder
	};

	/// @brief The incremental-evolution test's recall task at one stage: a demonstrated sequence of four events,
	/// recalled once the memory recall trigger comes.
	///
	/// @details The input is "nf 1" and the output "nf 2", both 100 wide. The memory recall trigger, a boost of the whole
	/// input field, comes at 0-50; E1 (40), E2 (90), E3 (15) and E4 (65) follow as Gaussian pulses 50 long, at 100, 250,
	/// 600 and 1200. The output then recalls three times, each after its own wait after the demonstration ends (1250), drawn
	/// from its own third of 300-500 (300-1500 from stage 3), and the 1700 after each trigger are recorded. All three branch off one run, which goes
	/// on without the trigger as the probe, watching the output for 3200 after the demonstration: as long as the longest
	/// wait of the later stages and a recall window. The state at each trigger's moment is saved in that run and restored,
	/// clock included, for its recall, so the demonstration and the waits are stepped once. A saved solution replays the
	/// demonstration and the trigger after a wait of 300 (1500 from stage 3).
	///
	/// Every field is sampled every 10, and stage 1 (RecallStage::singleOrder) scores:
	/// - f1 the input: it detects each event, and is back at rest by 50 after each pulse;
	/// - f2 the quiet output: no peak through the demonstration and the wait;
	/// - f3 memory, a guide: through the demonstration and the waits, the output pre-shaped at the events shown so far, the
	///   earliest highest, as far as the input is at rest and the output has no peak;
	/// - f4 recalled in order: after the trigger each event's first peak, in its place in the order (E1 as far as no
	///   other event peaked before it, each later event as far as the events due before it peaked before it), times how
	///   fully it peaked: its best mean peak score over three samples running, so a peak up for a single sample earns a
	///   third. A sample's peak score is how near the event the peak is, in proportion to its height (full from 2), as
	///   narrow as a peak (full up to 20 wide, nothing from 40) and staying in place;
	/// - f5 one at a time: each event's credit in place, times the share of the samples it peaks on with no other peak
	///   on the output;
	/// - f6 to come, a guide: while they wait after the trigger, the events not yet recalled pre-shaped in order;
	/// - f7 released: each event's credit in place, times how far it stays below threshold once its peak ends for half,
	///   and how deep a trough it lies in for the other half (the guide);
	/// - f8 stopped: no peak once the sequence is over, until 3200 after the trigger (as long as the probe watches), as
	///   far as it was recalled in order, for half, and a trough at each event, as far as it was recalled in its place,
	///   for the other half (the guide).
	/// Pre-shapes and troughs are measured from the output's level far from every event (its median over the places
	/// more than 10 from all of them). f4-f8 are scaled by how far the output had no peak before the trigger, and by how
	/// far it had no peak, and no event peaked, in the probe. There is no timing term: the order is what stage 1 recalls.
	/// f2 and f4-f8 are each the mean over the three recalls, so one lucky recall does not decide a design's rank.
	///
	/// The fitness is the larger of two weighted sums: the eight terms, weighted by
	/// config/solutions/incremental-recall-1-single-order.json, and the intended behaviour alone, f1, f2, f4, f5, and f7
	/// and f8 without their guides, weighted by config/solutions/incremental-recall-1-single-order-intended.json. The
	/// guides give the search a slope before any design recalls, without ranking a design that recalls another way
	/// below one that has them. A run stops on the terms of the larger sum: a design that recalls without the guides
	/// stops it once the intended behaviour's terms all pass the target.
	///
	/// Stage 2 (RecallStage::singleOrderAndTiming) adds a timing term after f5: each event's credit in place, times how
	/// near its first peak after the trigger lies to its demonstrated time (100, 250, 600 and 1200), three quarters as a
	/// Gaussian 0.2 of that time wide (scalar timing) and one quarter as the smaller of the delay and that time over the
	/// larger, as the sequence-and-timing-recall task credits timing. Its nine terms are weighted by
	/// config/solutions/incremental-recall-2-single-order-and-timing.json; its intended behaviour, with the timing term
	/// after one at a time, by config/solutions/incremental-recall-2-single-order-and-timing-intended.json.
	///
	/// Stage 3 (RecallStage::repeatableOrder) scores stage 1's eight terms, with no timing term, but draws the three
	/// waits from the thirds of 300-1500, so the memories must hold through a long wait as well as a short one. A saved
	/// solution replays the trigger after a wait of 1500. Its weights are in
	/// config/solutions/incremental-recall-3-repeatable-order.json and
	/// config/solutions/incremental-recall-3-repeatable-order-intended.json.
	///
	/// Stage 4 (RecallStage::repeatableOrderAndTiming) scores stage 2's nine terms, timing included, on stage 3's waits
	/// and replay. Its weights are in config/solutions/incremental-recall-4-repeatable-order-and-timing.json and
	/// config/solutions/incremental-recall-4-repeatable-order-and-timing-intended.json.
	///
	/// Stage 5 (RecallStage::consecutiveOrder) scores stage 1's eight terms on six triggers in turn in one run after the
	/// demonstration: six recalls. Their waits are 300 and 1500 twice each, and two middle waits drawn on the sampling grid,
	/// M1 from 700-900 and M2 from 900-1100, in one of two sequences drawn afresh each evaluation: 300, M1, 1500, 300,
	/// 1500, M2, or 1500, 300, M1, 1500, M2, 300. Each wait comes once among the first three triggers and once among the
	/// last three, and between the two sequences every wait follows every other. The run branches off the run without the
	/// trigger after its first wait from the end of the demonstration; each later trigger comes its wait after the recall
	/// window before it ends. Each recall's quiet is the output's through the wait before its trigger, and its silence is
	/// watched until the next trigger (after the last, until 3200 after it). Memory (f3) is read through the
	/// demonstration and the longest wait after it, in the run without the trigger.
	/// An event's place is the share of the other three on the right side of it, so both events of a swapped pair lose a
	/// third. Each recall's one-at-a-time credit loses a third for each sample two peaks are up together, and its release
	/// credit a third for each sample an event peaks again after its release, until the next trigger. The target asks
	/// each term of every recall, not their mean, so one recall that fails keeps a design short of it. Its confirmations
	/// of a reached target take the two sequences in turn. A saved solution replays the first sequence with each middle
	/// wait at the middle of its range: triggers after waits of 300, 800, 1500, 300, 1500 and 1000 in turn. Its weights
	/// are in
	/// config/solutions/incremental-recall-5-consecutive-order.json and
	/// config/solutions/incremental-recall-5-consecutive-order-intended.json.
	///
	/// The evaluation trace records both weighted sums, each recall trial (wait, trigger, quiet before it, the terms it
	/// earned, how many samples two peaks were up together and an event peaked again, and for each event its first peak,
	/// its place, its best peak score, how far it was alone, its timing credit and its release) and
	/// the probe (when it started watching, how long it watched, how quiet it stayed and how far each event peaked).
	/// @tparam stage The stage of behaviour the task scores.
	template <RecallStage stage>
	class IncrementalRecall final : public Solution
	{
	public:
		/// @brief How many events the demonstration shows.
		static constexpr size_t eventCount = 4;
		/// @brief How many samples running a peak is credited over: a peak up for fewer earns that share.
		static constexpr size_t peakSamples = 3;
		/// @brief How many moves of a peak's centre its drift is read over.
		static constexpr size_t driftSamples = 5;
		/// @brief How many waits one evaluation draws up to stage 4, each from its own third of the stage's waits: one
		/// recall after each.
		static constexpr size_t recallTrialCount = 3;
		/// @brief How many triggers come in turn in the one run of a consecutive stage, each its wait after the recall
		/// window before it.
		static constexpr size_t consecutiveTriggerCount = 6;

		/// @brief Creates the task from a topology, with a minimal genome.
		/// @param topology The input and output fields.
		explicit IncrementalRecall(const SolutionTopology& topology);
		/// @brief Creates the task from a saved design.
		/// @param initialTopology The input and output fields.
		/// @param phenotype The saved design.
		IncrementalRecall(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype);
		/// @brief A fresh solution of the same task and topology.
		/// @return The new solution.
		SolutionPtr clone() const override;
		/// @brief A copy of this solution, its design included.
		/// @return The copy.
		SolutionPtr copy() const override;
		/// @brief Whether the last evaluation reached the target: its fitness and every term of the weighted sum that was
		/// its fitness (from stage 5 each term's worst recall) exceed it.
		/// @param targetFitness The level the fitness and each term must exceed.
		/// @return Whether the fitness and every term of the larger weighted sum exceed @p targetFitness.
		[[nodiscard]] bool reachedTarget(double targetFitness) const override;
		/// @brief Prepares the evaluations that follow as one confirmation of a reached target: in a consecutive stage
		/// they recall after the sequence of triggers numbered @p confirmation (of the two, in turn), so any two
		/// confirmations cover both. Other stages draw their waits as before.
		/// @param confirmation Which confirmation the next evaluation is, from 0.
		void prepareTargetConfirmation(size_t confirmation) override;
	private:
		/// What one event did after the trigger (or in the probe).
		struct EventRecall
		{
			double bestScore = 0.0;
			std::array<double, peakSamples> lastScores{};
			std::optional<double> firstPeak;
			std::optional<double> releasedAt;
			std::vector<double> belowThreshold;
			std::vector<double> troughs;
			std::vector<double> alone;
			/// The times it peaked with another bump on the output.
			std::vector<double> sharedAt;
			/// The times it peaked again after its release, until the next trigger.
			std::vector<double> refiredAt;
			std::optional<double> lastCentroid;
			std::array<double, driftSamples> lastMoves{};
		};
		using Recall = std::array<EventRecall, eventCount>;
		/// What the output did once the sequence was over.
		struct AfterRecall
		{
			std::optional<double> from;
			std::vector<double> quiet;
			/// The quiet samples from the end of the recall window to the end of the watch.
			std::vector<double> quietPastWindow;
			std::array<std::vector<double>, eventCount> troughs;
		};
		/// The probe, from the end of the demonstration on, without the trigger: how quiet the output stayed, and what
		/// each event did.
		struct Probe
		{
			std::vector<double> quiet;
			Recall events{};
		};
		/// The moment one trigger comes, in the run without it: when, how far the output had no peak until then, and the
		/// state to recall from.
		struct CueMoment
		{
			double cue = 0.0;
			double quietBeforeCue = 0.0;
			SimulationState state;
		};
		/// What one recall earned, each recall term already scaled, and what it did, for the evaluation trace.
		struct RecallTrial
		{
			double quiet = 0.0;
			double recalled = 0.0;
			double oneAtATime = 0.0;
			double timing = 0.0;
			double toCome = 0.0;
			double released = 0.0;
			double stopped = 0.0;
			double releasedWithoutGuide = 0.0;
			double stoppedWithoutGuide = 0.0;
			/// How far the output had no peak from the end of the recall window to the end of the watch, unscaled: in a
			/// consecutive stage, the quiet before the next trigger.
			double quietPastWindow = 0.0;
			nlohmann::json trace;
		};
		using Waits = std::array<double, recallTrialCount>;

		std::vector<double> inputPeaks;
		std::vector<double> inputRest;
		std::vector<double> outputMemory;
		std::optional<double> cue;
		std::vector<double> baselinePositions;
		std::vector<double> intendedFitnessWeights;
		std::vector<double> targetTerms;
		std::optional<size_t> confirmation;

		void testPhenotype() override;
		void createPhenotypeEnvironment() override;
		/// @brief Draws each recall's wait before the trigger from its own third of the stage's waits, on the sampling grid.
		/// @return The waits, shortest first.
		[[nodiscard]] Waits drawRecallWaits() const;
		/// @brief A consecutive stage's waits in turn: one of its two sequences of six triggers, drawn afresh or the
		/// confirmation's, with its two middle waits drawn on the sampling grid.
		/// @return Each trigger's wait, the first from the end of the demonstration, each later one from the end of the
		/// recall window before it.
		[[nodiscard]] std::vector<double> consecutiveWaits() const;
		/// @brief Sets the trigger at 0-50 and, if @p cueTime is given, again then.
		/// @param cueTime When the trigger comes again, if it does.
		void setCue(std::optional<double> cueTime);
		/// @brief Rewinds the clock to @p cueTime and switches the trigger on, as the step that reached it would have.
		/// @param cueTime When the trigger comes, the moment the restored state was saved at.
		void triggerAt(double cueTime);
		/// @brief Whether the input should be back at rest at @p time: no trigger or event pulse within a pulse and its
		/// settling time before it.
		/// @param time The sample's time.
		/// @return Whether the input should be at rest.
		[[nodiscard]] bool inputSettledAt(double time) const;
		/// @brief Steps the simulation until @p time.
		/// @param time When to stop.
		void runUntil(double time);
		/// @brief Steps until @p time, calling @p onSample at every sample, and scoring the input's rest where it
		/// should be at rest.
		/// @param time When to stop.
		/// @param onSample Called with each sample's time and, where the input should be at rest, how far it is.
		void sampleUntil(double time, const std::function<void(double, std::optional<double>)>& onSample);
		/// @brief Runs the demonstration, then holds without the trigger as the probe, scoring the output's memory until
		/// the last trigger's moment (in a consecutive stage, until the longest wait) and saving the state at each.
		/// @param waits The wait after the demonstration of each trigger that branches off this run.
		/// @param probe Receives what the output did without the trigger.
		/// @return Each trigger's moment, in the order of @p waits.
		std::vector<CueMoment> demonstrateAndHold(std::span<const double> waits, Probe& probe);
		/// @brief Recalls from one trigger's moment in the run without the trigger and scores the recall.
		/// @param moment The trigger's moment.
		/// @param probeQuiet How far the output had no peak, and no event peaked, in the probe.
		/// @return What the recall earned and did.
		RecallTrial recallFrom(const CueMoment& moment, double probeQuiet);
		/// @brief Recalls after each trigger in turn in one run, from the first trigger's moment in the run without it.
		/// @param waits Each trigger's wait: the first from the end of the demonstration, each later one from the end of
		/// the recall window before it.
		/// @param first The first trigger's moment.
		/// @param probeQuiet How far the output had no peak, and no event peaked, in the probe.
		/// @return What each recall earned and did, in turn.
		std::vector<RecallTrial> recallInTurn(std::span<const double> waits, const CueMoment& first, double probeQuiet);
		/// @brief Gives the trigger now and scores the recall after it.
		/// @param cueTime When the trigger comes: the simulation's time.
		/// @param wait The wait before it, for the trace.
		/// @param quietBeforeCue How far the output had no peak through the wait before it.
		/// @param probeQuiet How far the output had no peak, and no event peaked, in the probe.
		/// @param watchedUntil Until when the output is watched for silence once the recall window ends.
		/// @return What the recall earned and did.
		RecallTrial recallAt(double cueTime, double wait, double quietBeforeCue, double probeQuiet, double watchedUntil);
		/// @brief Samples the recall window after the trigger, then the output's silence until @p watchedUntil.
		/// @param recall Receives what each event did.
		/// @param after Receives what the output did once the sequence was over.
		/// @param stillAtEvents Receives, at each sample with a bump on the output until the sequence is released, how
		/// far every bump stands still at an event.
		/// @param watchedUntil Until when the output is watched for silence once the recall window ends.
		/// @return How far the events still waiting were pre-shaped in order, at each sample some were waiting.
		std::vector<double> recordRecall(Recall& recall, AfterRecall& after, std::vector<double>& stillAtEvents,
			double watchedUntil);
		/// @brief The output's level far from every event: its median over baselinePositions.
		/// @return The level.
		[[nodiscard]] double baseline() const;
		/// @brief How far the output is pre-shaped at @p positions above @p level, and in that order.
		/// @param positions The events' positions, in the order their pre-shapes should descend.
		/// @param level The level pre-shapes are measured from.
		/// @return The mean pre-shape, half of it credited regardless of order and half as far as it descends in order.
		[[nodiscard]] double preShapedInOrder(const std::vector<double>& positions, double level) const;
		/// @brief How fully a peak stands at @p position: near it, tall enough, and no wider than a peak.
		/// @param position Position of the event.
		/// @return The nearest bump's closeness to @p position, times its height credit and its width credit.
		[[nodiscard]] double peakAt(double position) const;
		/// @brief Reads every event's peak, release, peaking again and trough at one sample.
		/// @param time The sample's time.
		/// @param recall What each event did so far; updated.
		/// @return How far the events not yet peaked are pre-shaped in order, if any are waiting.
		std::optional<double> observeRecall(double time, Recall& recall) const;
		/// @brief Reads whether the output has a peak, and how deep a trough each event lies in, once the sequence is over.
		/// @param time The sample's time.
		/// @param after What the output did so far once the sequence was over; updated.
		void observeAfterRecall(double time, AfterRecall& after) const;
		/// @brief How far the output's nearest bump moved since the last sample, if it is the same bump.
		/// @param lastCentroid Its centroid at the last sample, if any.
		/// @param centroid Its centroid now, if any.
		/// @return The distance moved, or 0 if either is missing or the bump is a new one.
		[[nodiscard]] static double movedSince(std::optional<double> lastCentroid, std::optional<double> centroid);
		/// @brief How far a peak stays in place: up to stage 4 over the last two samples (the smaller of its last two
		/// moves), from stage 5 over all its last moves (their sum without the largest, as a peak may split for a sample
		/// as it is released).
		/// @param lastMoves How far it moved over each of its last samples, the latest last.
		/// @return 1 up to a drift of 1, 0 from a drift of 3.
		[[nodiscard]] static double stayingInPlace(const std::array<double, driftSamples>& lastMoves);
		/// @brief How far every bump on the output stands still at an event: each bump within reach of an event, and
		/// staying in place as that event's nearest bump.
		/// @param recall What each event did, this sample's moves included.
		/// @return The least still bump's credit, 0 for a bump at no event, or nothing if the output has no bump.
		[[nodiscard]] std::optional<double> bumpsStillAtEvents(const Recall& recall) const;
		/// @brief Records, once the recall window is over, each released event peaking again.
		/// @param time The sample's time.
		/// @param recall What each event did; updated.
		void observeRefiring(double time, Recall& recall) const;
		/// @brief How many samples of a recall some event did something at: two peaks up together, say, or an event
		/// peaking again.
		/// @param recall What each event did after the trigger.
		/// @param times When each event did it.
		/// @return The number of distinct samples.
		[[nodiscard]] static size_t distinctSamples(const Recall& recall, const std::vector<double> EventRecall::* times);
		/// @brief Whether every event has peaked and been released.
		/// @param recall What each event did.
		/// @return Whether the sequence is over.
		[[nodiscard]] static bool sequenceReleased(const Recall& recall);
		/// @brief How far an event's first peak is in its place in the order.
		/// @param recall What each event did.
		/// @param event The event's index in the order.
		/// @return Its share in place.
		[[nodiscard]] static double placeOf(const Recall& recall, size_t event);
		/// @brief An event's best peak score times its place in the order.
		/// @param recall What each event did.
		/// @param event The event's index in the order.
		/// @return Its credit in place.
		[[nodiscard]] static double creditInPlace(const Recall& recall, size_t event);
		/// @brief The mean over the events of @p perEvent, each times the event's credit in place.
		/// @param recall What each event did.
		/// @param perEvent What each event earns beyond its credit in place.
		/// @return The mean credit.
		[[nodiscard]] static double meanCreditInPlace(const Recall& recall,
			const std::function<double(const EventRecall&)>& perEvent);
		/// @brief How near an event's first peak after the trigger lies to its demonstrated time after the trigger.
		/// @param recall What each event did.
		/// @param event The event's index in the order.
		/// @param cueTime When the trigger came.
		/// @return Three quarters a Gaussian 0.2 of that time wide, one quarter the smaller of the delay and that time over
		/// the larger; 0 if the event never peaked.
		[[nodiscard]] static double onTime(const Recall& recall, size_t event, double cueTime);
		/// @brief How far each event was recalled in its place and at its time.
		/// @param recall What each event did.
		/// @param cueTime When the trigger came.
		/// @return The mean over the events of the timing credit, each times the event's credit in place.
		[[nodiscard]] static double timingOf(const Recall& recall, double cueTime);
		/// @brief How each recalled event stayed released once its peak ended.
		/// @param recall What each event did.
		/// @param withGuides Whether half the release is for a trough.
		/// @return The mean over the events of the release, each times the event's credit in place.
		[[nodiscard]] static double releaseAfterRecall(const Recall& recall, bool withGuides);
		/// @brief How the output stopped once the sequence was over.
		/// @param recall What each event did.
		/// @param after What the output did once the sequence was over.
		/// @param withGuides Whether half the stop is for a trough at each event.
		/// @return No peak, as far as the recall was in order, and with guides the troughs, each as far as its event was
		/// recalled in its place.
		[[nodiscard]] static double stoppedAfterRecall(const Recall& recall, const AfterRecall& after, bool withGuides);
		/// @brief What the recall did, for the evaluation trace.
		/// @param recall What each event did.
		/// @param after What the output did once the sequence was over.
		/// @return Each event's first peak, place, best peak score, share alone and release, and the stop.
		[[nodiscard]] nlohmann::json recallSummary(const Recall& recall, const AfterRecall& after) const;
		/// @brief Rounds @p value to two decimals, for the evaluation trace.
		/// @param value The value.
		/// @return The rounded value.
		[[nodiscard]] static double rounded(double value);
	};
}
