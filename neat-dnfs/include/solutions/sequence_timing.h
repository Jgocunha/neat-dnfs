#pragma once

#include <array>
#include <optional>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "neat/solution.h"
#include "neat_tools/utils.h"

namespace neat_dnfs
{
	/// @brief Evolves an architecture that learns the serial order and relative timing of a
	/// demonstrated event sequence, then reproduces it when cued by a start event.
	///
	/// @details Inspired by Wojtak et al. (2023), "Adaptive timing in a dynamic field
	/// architecture for natural human-robot interactions": a robot watches a sequence of task
	/// events, stores their order and timing, and later recalls them on its own once the task
	/// is started.
	///
	/// The architecture has one input field, "nf 1", and one output field, "nf 2", both 200
	/// wide. Each task event is a transient Gaussian pulse at its own position in the input
	/// field: the start event X at 50, then D (180), C (120), A (20) and B (80). Onsets follow
	/// X by 100, 600, 800 and 1800 time units, i.e. the intervals X-D 100, D-C 500, C-A 200 and
	/// A-B 1000. All durations are in simulation time, so the protocol holds for any deltaT.
	///
	/// An evaluation runs three independent trials, each starting from rest and each beginning
	/// with a demonstration of the sequence:
	///
	/// Demonstration -- X, D, C, A and B are pulsed in order at their onset times. The input
	/// field must represent each pulse with a bump at the event's position and return to rest
	/// once the pulse ends, so it cannot itself be the memory. The output field must stay
	/// silent: the robot is observing, not acting.
	///
	/// Recall trials (three) -- after a gap during which the output stays silent, only X is
	/// pulsed. Without any further input, the output field must form a peak at D, C, A and B in
	/// turn, each at the demonstrated interval after the previous one, each giving way to the
	/// next, and the last one, B, dying out before recall ends. The three gaps are drawn afresh
	/// on every evaluation, one from each third of 300-1500. Because the gap differs between
	/// trials, only timing measured from X can be right in all of them: a delayed replay of the
	/// demonstration matches one at most, and fresh draws stop a solution from tuning itself to a
	/// fixed set of gaps.
	///
	/// Probe trial (one) -- after the first trial's gap, X is never given. The output must stay
	/// silent for as long as a recall would take, so recall cannot start without its cue.
	///
	/// The seven partial fitness terms, in order:
	///
	/// - f1, input representation: a bump at the pulse position at the end of every pulse.
	/// - f2, input returns to rest: the input field at rest between events and after each trial.
	/// - f3, output silent outside recall: during demonstrations, gaps and the whole probe
	///   trial, averaged over every step.
	/// - f4, recalled events: the share of D, C, A, B whose peak rises in the output after X.
	///   A peak only counts once it has been absent after X, so one carried over from the
	///   demonstration earns nothing.
	/// - f5, recalled timing: each event's onset, measured from X, against its demonstrated onset,
	///   with a tolerance proportional to that onset (scalar timing). An event that never appears
	///   scores 0.
	/// - f6, one peak at a time: shortly after each recalled onset, the output holds a single
	///   bump, at that event; and, if B appeared, the output is silent again when recall ends.
	/// - f7, output near rest: how far the output field's lowest activation lies below its resting
	///   level, averaged over every step outside recall and, separately, during recall. Full credit
	///   within 60 below rest, falling as 60 / depth beyond it; the two phases are averaged. Couplings
	///   that hold the output hundreds below rest otherwise pass as silence.
	///
	/// f4 to f6 are averaged over the three recall trials. Each trial's credit is scaled by how
	/// silent the output stayed before its cue (demonstration and gap together), so an output that
	/// is always active cannot earn recall credit from peaks that happen to sweep past the events.
	/// f4 to f6 are also scaled by how silent the output stayed in the probe trial after its
	/// demonstration, so peaks that rise without X, such as a replay at a fixed lag after the
	/// demonstration, cannot earn recall credit in the trials where X happens to precede them.
	///
	/// Every evaluation also records a trace in SolutionParameters::evaluationTrace: per trial, how
	/// silent the output stayed in each phase and, for recall trials, each event's onset and isolation
	/// and every output peak that rose after the cue (time and position). Neural fields here are translation
	/// invariant and carry no spatial bias of their own, so the event positions can only come
	/// from the demonstration: no genome can hardwire the recalled peaks.
	class SequenceTiming : public Solution
	{
	public:
		/// @brief Number of events recalled after the start cue: D, C, A and B.
		static constexpr size_t recalledEventCount = 4;

		/// @brief Builds a task solution whose genome is created from @p topology.
		/// @param topology Fields to start from: the input and the output.
		explicit SequenceTiming(const SolutionTopology& topology);
		/// @brief Builds a task solution whose genome is decoded from an existing phenotype, such as a template.
		/// @param initialTopology Fields the task declares: the input and the output.
		/// @param phenotype Simulation the genome is translated from.
		SequenceTiming(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype);
		/// @brief Creates a solution with a fresh genome built from this one's initial topology and a new id.
		/// @return The new solution.
		SolutionPtr clone() const override;
		/// @brief Creates an independent solution with this one's phenotype and a new id.
		/// @return The new solution.
		SolutionPtr copy() const override;
	protected:
		/// @brief How the start event X is delivered to the input field.
		enum class StartCue
		{
			gaussianPulse, ///< A transient Gaussian at X's position, like every other event.
			boost          ///< A transient homogeneous boost of the whole input field.
		};

		/// @brief What the recalled-timing term (f5) gives credit for.
		enum class TimingCredit
		{
			exactOnsets,   ///< Each onset against its demonstrated onset only.
			orderAndOnsets, ///< Half for the events rising one after another, half for exact onsets.
			orderOnly       ///< Only for the events rising one after another, whenever they do.
		};

		/// @brief Builds a task variant whose genome is created from @p topology.
		/// @param topology Fields to start from: the input and the output.
		/// @param startCue How X is delivered.
		/// @param timingCredit What the recalled-timing term gives credit for.
		/// @param taskName Display name, which also names the run's data folder.
		/// @param slug Task slug whose config supplies the fitness weights.
		SequenceTiming(const SolutionTopology& topology, StartCue startCue, TimingCredit timingCredit,
			const std::string& taskName, const std::string& slug);
		/// @brief Builds a task variant whose genome is decoded from an existing phenotype, such as a template.
		/// @param initialTopology Fields the task declares: the input and the output.
		/// @param phenotype Simulation the genome is translated from.
		/// @param startCue How X is delivered.
		/// @param timingCredit What the recalled-timing term gives credit for.
		/// @param taskName Display name, which also names the run's data folder.
		/// @param slug Task slug whose config supplies the fitness weights.
		SequenceTiming(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype,
			StartCue startCue, TimingCredit timingCredit, const std::string& taskName, const std::string& slug);
	private:
		StartCue startCue;
		TimingCredit timingCredit;

		/// @brief Running sums behind the seven partial fitness terms, accumulated across trials.
		struct TaskScores
		{
			double inputRepresentation = 0.0;
			int pulses = 0;
			double inputRest = 0.0;
			int restSamples = 0;
			double outputSilence = 0.0;
			int silentSteps = 0;
			double recalledEvents = 0.0;
			double recalledTiming = 0.0;
			double oneAtATime = 0.0;
			int recallTrials = 0;
			double probeSilence = 1.0; // output silence in the probe trial, after its demonstration
			double outputDepthOutsideRecall = 0.0; // per step, how far the output's lowest activation is below rest
			int outputDepthStepsOutsideRecall = 0;
			double outputDepthDuringRecall = 0.0;
			int outputDepthStepsDuringRecall = 0;
		};

		/// @brief What the output field did during one recall, one slot per recalled event.
		struct RecallObservation
		{
			std::array<std::optional<int>, recalledEventCount> onsetSteps{};
			std::array<bool, recalledEventCount> absentSinceCue{};
			std::array<double, recalledEventCount> isolation{};
			double lastPeakDecay = 0.0;
			std::vector<std::pair<double, double>> risingPeaks; // (time since the cue, position), first few only
			int risingPeakCount = 0;
		};

		void testPhenotype() override;
		void createPhenotypeEnvironment() override;

		/// @brief Demonstrates the sequence, waits @p gap, then cues and scores a recall.
		/// @param gap Simulation time between the end of the demonstration and the start cue.
		/// @param scores Accumulates every partial fitness term.
		/// @return The trial's trace: output silence per phase and what the recall did.
		nlohmann::json runRecallTrial(double gap, TaskScores& scores);
		/// @brief Demonstrates the sequence, then waits @p gap and a whole recall's length without the start cue.
		/// @param gap Simulation time a recall trial would wait before its cue.
		/// @param scores Accumulates the input and output-silence terms, and records the silence without the cue
		/// that scales recall credit.
		/// @return The trial's trace: output silence during the demonstration and without the cue.
		nlohmann::json runProbeTrial(double gap, TaskScores& scores);
		/// @brief Resets the simulation and pulses X, D, C, A and B at their onset times.
		/// @param scores Accumulates the input and output-silence terms.
		void demonstrateSequence(TaskScores& scores);
		/// @brief Pulses X, then records when each event's peak rises in the output and how cleanly.
		/// @param scores Accumulates the input terms.
		/// @return Onset step, isolation and last-peak decay of the recall.
		RecallObservation recallSequence(TaskScores& scores);
		/// @brief Delivers the start cue for the pulse duration, keeping the output silent.
		/// @param scores Accumulates the output-silence term, and the input representation term for a Gaussian cue.
		void pulseStartCue(TaskScores& scores);
		/// @brief Starts delivering the start cue X to the input field.
		void startStartCue();
		/// @brief Stops delivering the start cue; a Gaussian cue also scores the input's bump at X.
		/// @param scores Accumulates the input representation term for a Gaussian cue.
		void endStartCue(TaskScores& scores);
		/// @brief Applies an event stimulus for the pulse duration, keeping the output silent, then scores the input.
		/// @param position Position of the event in the input field.
		/// @param scores Accumulates the input representation and output-silence terms.
		void pulseEvent(double position, TaskScores& scores);
		/// @brief Applies the event stimulus at @p position to the input field.
		/// @param position Position of the event in the input field.
		void startPulse(double position);
		/// @brief Scores the input field's bump at @p position, then removes the event stimulus.
		/// @param position Position of the event in the input field.
		/// @param scores Accumulates the input representation term.
		void endPulse(double position, TaskScores& scores);
		/// @brief Runs for @p duration simulation time, scoring the output field for silence after every step.
		/// @param duration Simulation time to run.
		/// @param scores Accumulates the output-silence term.
		void runWhileOutputSilent(double duration, TaskScores& scores);
		/// @brief How far the output field's lowest activation lies below its resting level, now.
		/// @return The depth below rest; 0 if no activation lies below it.
		[[nodiscard]] double outputDepthBelowRest() const;
		/// @brief Samples how close the input field is to its resting level.
		/// @param scores Accumulates the input-rest term.
		void sampleInputRest(TaskScores& scores) const;

		/// @brief Records, for every event not yet seen, whether its peak has risen in @p bumps since the cue.
		/// @param step Steps elapsed since the start cue.
		/// @param bumps Bumps currently in the output field.
		/// @param observation Onsets recorded so far; updated in place.
		static void detectOnsets(int step, const std::vector<dnf_composer::element::NeuralFieldBump>& bumps,
			RecallObservation& observation);
		/// @brief Scores, for every event whose onset was the settle time before @p step, whether its peak stands alone.
		/// @param step Steps elapsed since the start cue.
		/// @param bumps Bumps currently in the output field.
		/// @param observation Onsets recorded so far; isolation scores are written in place.
		static void scoreIsolation(int step, const std::vector<dnf_composer::element::NeuralFieldBump>& bumps,
			RecallObservation& observation);

		/// @brief Records every output bump that was not up on the previous step.
		/// @param step Steps elapsed since the start cue.
		/// @param previousBumps Bumps in the output field on the previous step.
		/// @param bumps Bumps currently in the output field.
		/// @param observation Rising peaks recorded so far; updated in place.
		static void recordRisingPeaks(int step, const std::vector<dnf_composer::element::NeuralFieldBump>& previousBumps,
			const std::vector<dnf_composer::element::NeuralFieldBump>& bumps, RecallObservation& observation);
		/// @brief Converts one recall's observation into its trace.
		/// @param observation What the output field did during the recall.
		/// @return Onsets, isolation per event and order credit, last-peak decay, and the output peaks that rose.
		[[nodiscard]] static nlohmann::json traceOf(const RecallObservation& observation);
		/// @brief Credit for the output staying near its resting level over one phase.
		/// @param depthSum Per-step depths of the output below rest, summed over the phase.
		/// @param steps Number of steps summed.
		/// @return 1 while the mean depth is within the tolerated depth, then tolerated depth / mean depth.
		[[nodiscard]] static double outputLevelScore(double depthSum, int steps);
		/// @brief Mean output-silence score over the steps run between two snapshots of the scores.
		/// @param from Scores before the phase.
		/// @param to Scores after the phase.
		/// @return Mean silence in [0,1]; 1 if no step was run.
		[[nodiscard]] static double meanOutputSilence(const TaskScores& from, const TaskScores& to);
		/// @brief Share of recalled events whose peak rose in the output field.
		/// @param observation What the output field did during recall.
		/// @return Fitness score in [0,1].
		[[nodiscard]] static double recalledEventsScore(const RecallObservation& observation);
		/// @brief Mean agreement of each recalled onset, measured from the start cue, with the demonstrated one.
		/// @param observation What the output field did during recall.
		/// @return Fitness score in [0,1]; a missing event scores 0.
		[[nodiscard]] static double recalledTimingScore(const RecallObservation& observation);
		/// @brief How far the recalled events rose one after another, in the demonstrated order.
		/// @param observation What the output field did during recall.
		/// @return Fitness score in [0,1]: per consecutive pair, the time between their onsets as a fraction of a
		/// minimum separation, capped at 1; a missing or reversed pair scores 0.
		[[nodiscard]] static double orderScore(const RecallObservation& observation);
		/// @brief The recalled-timing term (f5) for this variant's timing credit.
		/// @param observation What the output field did during recall.
		/// @return Fitness score in [0,1].
		[[nodiscard]] double timingTermScore(const RecallObservation& observation) const;
		/// @brief Mean of the isolation scores of the recalled events and the decay of the last peak.
		/// @param observation What the output field did during recall.
		/// @return Fitness score in [0,1].
		[[nodiscard]] static double oneAtATimeScore(const RecallObservation& observation);
		/// @brief Draws one recall gap from each third of the gap range, shortest first.
		/// @return The gaps for this evaluation's three recall trials.
		[[nodiscard]] static std::array<double, 3> drawRecallGaps();
		/// @brief Number of simulation steps that cover @p duration at the configured deltaT.
		/// @param duration Simulation time.
		/// @return The step count, rounded to the nearest step.
		[[nodiscard]] static int stepsFor(double duration);
	};

	/// @brief SequenceTiming with the start event X delivered as a homogeneous boost of the input
	/// field instead of a Gaussian at position 50.
	///
	/// @details X then carries no position: it can only act through couplings that respond to the
	/// input field's total activity. Everything else -- the events, the trials, the seven partial
	/// fitness terms -- is the same as SequenceTiming, except that input representation (f1) scores
	/// the event pulses only, since the boost is not meant to form a bump.
	class SequenceTimingBoost final : public SequenceTiming
	{
	public:
		/// @brief Builds a task solution whose genome is created from @p topology.
		/// @param topology Fields to start from: the input and the output.
		explicit SequenceTimingBoost(const SolutionTopology& topology);
		/// @brief Builds a task solution whose genome is decoded from an existing phenotype, such as a template.
		/// @param initialTopology Fields the task declares: the input and the output.
		/// @param phenotype Simulation the genome is translated from.
		SequenceTimingBoost(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype);
		/// @brief Creates a solution with a fresh genome built from this one's initial topology and a new id.
		/// @return The new solution.
		SolutionPtr clone() const override;
		/// @brief Creates an independent solution with this one's phenotype and a new id.
		/// @return The new solution.
		SolutionPtr copy() const override;
	};

	/// @brief SequenceTimingBoost whose recalled-timing term (f5) is half order credit and half exact onsets.
	///
	/// @details A stepping stone for evolving the boost task from scratch. An output that ignites at
	/// every event at once earns full recall credit (f4) and some exact-timing credit, but nothing
	/// rewards pulling its peaks apart. Order credit does: each consecutive pair of events earns the
	/// time between their onsets as a fraction of a minimum separation, so simultaneous peaks earn
	/// almost nothing and every step towards a sequence earns more. A solution that recalls the
	/// sequence at the demonstrated times scores the same as in SequenceTimingBoost.
	class SequenceTimingBoostOrder final : public SequenceTiming
	{
	public:
		/// @brief Builds a task solution whose genome is created from @p topology.
		/// @param topology Fields to start from: the input and the output.
		explicit SequenceTimingBoostOrder(const SolutionTopology& topology);
		/// @brief Builds a task solution whose genome is decoded from an existing phenotype, such as a template.
		/// @param initialTopology Fields the task declares: the input and the output.
		/// @param phenotype Simulation the genome is translated from.
		SequenceTimingBoostOrder(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype);
		/// @brief Creates a solution with a fresh genome built from this one's initial topology and a new id.
		/// @return The new solution.
		SolutionPtr clone() const override;
		/// @brief Creates an independent solution with this one's phenotype and a new id.
		/// @return The new solution.
		SolutionPtr copy() const override;
	};

	/// @brief SequenceTimingBoost whose recalled-timing term (f5) is order credit alone.
	///
	/// @details The first stage of a curriculum for evolving the boost task from scratch. In
	/// SequenceTimingBoostOrder, exact timing pulls the first event (D, due 100 after X) out of an
	/// early burst of simultaneous peaks, which reverses the first pair and cancels the order credit.
	/// Without the exact-onset half, every peak that rises after its predecessor earns credit wherever
	/// it lands. Its champions then seed SequenceTimingBoost, where exact onsets are scored.
	class SequenceTimingBoostOrderOnly final : public SequenceTiming
	{
	public:
		/// @brief Builds a task solution whose genome is created from @p topology.
		/// @param topology Fields to start from: the input and the output.
		explicit SequenceTimingBoostOrderOnly(const SolutionTopology& topology);
		/// @brief Builds a task solution whose genome is decoded from an existing phenotype, such as a template.
		/// @param initialTopology Fields the task declares: the input and the output.
		/// @param phenotype Simulation the genome is translated from.
		SequenceTimingBoostOrderOnly(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype);
		/// @brief Creates a solution with a fresh genome built from this one's initial topology and a new id.
		/// @return The new solution.
		SolutionPtr clone() const override;
		/// @brief Creates an independent solution with this one's phenotype and a new id.
		/// @return The new solution.
		SolutionPtr copy() const override;
	};
}
