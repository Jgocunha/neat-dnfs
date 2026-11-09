#pragma once

#include <array>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "solutions/recall_behaviour.h"

namespace neat_dnfs
{
	/// @brief The sequence-timing task, scored by what the output does at every stage: its intended behaviour alone.
	///
	/// @details Inspired by Wojtak et al. (2023), "Adaptive timing in a dynamic field architecture for natural
	/// human-robot interactions": the architecture watches a sequence of events, stores their order and timing, and
	/// replays them on its own once cued.
	///
	/// A trial starts from rest. X, a boost of the input field "nf 1", is given at 0-50; D (180), C (120), A (20) and
	/// B (80) follow as Gaussian pulses at 100, 600, 800 and 1800. A recall trial gives X again after a gap, 1850 plus
	/// the gap, and records the 2300 after it; the probe trial never gives it, and lasts as long as the evaluation's
	/// longest recall trial. Each evaluation runs the probe, then three recall trials, with gaps drawn afresh from each
	/// third of 300-1500 and rounded to the sampling interval: only timing measured from X can be right in all three. The
	/// schedule is laid down as timed stimuli, so a saved solution replays a recall trial with a gap of 700 in
	/// dnf-composer.
	///
	/// Every field is sampled every 10 and scored against what the reference design
	/// (tests/fixtures/sequence-timing-behaviour-reference.dnf) does there, by the rules of RecallBehaviour:
	/// - the input detects each event;
	/// - the output "nf 2" stays below threshold through each demonstration and gap;
	/// - after X the output peaks at D, C, A and B in turn, one at a time, each first peaking in its place;
	/// - each at the demonstrated time after X: 100, 600, 800 and 1800, credited three quarters as a Gaussian 0.2 of that
	///   time wide (scalar timing) and one quarter as the ratio of the delay to that time (the smaller over the larger),
	///   which keeps a slope toward it where the Gaussian is flat, as far as the event peaked in its place;
	/// - once an event's peak ends, it stays released below threshold;
	/// - once the sequence is over, the output has no peak until the recall ends. The sequence is over once every event
	///   has peaked and been released, or 300 after B's time if that never happens.
	///
	/// Each recall trial's credit is scaled by how far the output had no peak before its X, and all of it by how far the
	/// output had no peak in the probe trial after its demonstration, and by how far no event peaked there, each scored as
	/// a recall would score it: an output that is always active, or that replays at a fixed lag after the demonstration,
	/// however briefly, earns no recall credit.
	///
	/// The six partial fitness terms are those stages: f1 the input, f2 the quiet output before X and in the probe, f3
	/// the recall in order, f4 its timing, f5 the release of each recalled event and f6 the output after recall.
	/// SequenceTimingBehaviourGuided adds the pre-shapes and troughs the reference design has.
	///
	/// Every evaluation records a trace in SolutionParameters::evaluationTrace: per trial, its kind, gap and how quiet the
	/// output stayed before X; for the probe trial, how far each event peaked without X; for each recall trial, what each
	/// event did and its timing credit, and snapshots of the input and output at the points the reference peaks.
	class SequenceTimingBehaviour : public RecallBehaviour
	{
	public:
		/// @brief Builds a task solution whose genome is created from @p topology.
		/// @param topology Fields to start from: the input and the output.
		explicit SequenceTimingBehaviour(const SolutionTopology& topology);
		/// @brief Builds a task solution whose genome is decoded from an existing phenotype, such as a template.
		/// @param initialTopology Fields the task declares: the input and the output.
		/// @param phenotype Simulation the genome is translated from.
		SequenceTimingBehaviour(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype);
		/// @brief Creates a solution with a fresh genome built from this one's initial topology and a new id.
		/// @return The new solution.
		SolutionPtr clone() const override;
		/// @brief Creates an independent solution with this one's phenotype and a new id.
		/// @return The new solution.
		SolutionPtr copy() const override;
	protected:
		/// @brief Builds a task variant whose genome is created from @p topology.
		/// @param topology Fields to start from: the input and the output.
		/// @param guides Whether the fitness also pays the guides.
		/// @param taskName Display name, which also names the run's data folder.
		/// @param slug Task slug whose config supplies the fitness weights.
		SequenceTimingBehaviour(const SolutionTopology& topology, Guides guides, const std::string& taskName,
			const std::string& slug);
		/// @brief Builds a task variant whose genome is decoded from an existing phenotype, such as a template.
		/// @param initialTopology Fields the task declares: the input and the output.
		/// @param phenotype Simulation the genome is translated from.
		/// @param guides Whether the fitness also pays the guides.
		/// @param taskName Display name, which also names the run's data folder.
		/// @param slug Task slug whose config supplies the fitness weights.
		SequenceTimingBehaviour(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype,
			Guides guides, const std::string& taskName, const std::string& slug);
	private:
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
		};

		/// @brief Loads the weights of this task's partial fitness terms, and those of the intended behaviour alone.
		/// @param slug Task slug whose config supplies this task's weights.
		void loadAllFitnessWeights(const std::string& slug);
		void testPhenotype() override;
		void createPhenotypeEnvironment() override;
		[[nodiscard]] bool inputSettledAt(double time) const override;

		/// @brief Draws the evaluation's three recall gaps, one from each third of 300-1500, rounded to the sampling
		/// interval so that X falls on a sample.
		/// @return The gaps, shortest third first.
		[[nodiscard]] static std::array<double, 3> drawRecallGaps();
		/// @brief Sets the trial's X: at 0-50 and, for a recall trial, again at @p cue.
		/// @param cue When the recall trial gives X again; none for the probe trial.
		void setCue(std::optional<double> cue);
		/// @brief Runs a trial from rest through its demonstration and up to @p until, scoring the input, the quiet output
		/// and, with the guides scored, its memory.
		/// @param until When to stop: X of a recall trial, or the end of the demonstration in the probe trial.
		/// @param scores Accumulates the input, quiet output and memory terms.
		/// @param snapshots Receives a snapshot before D and as each event is shown.
		/// @return How far the output had no peak over the demonstration and the gap.
		double demonstrate(double until, Scores& scores, nlohmann::json& snapshots);
		/// @brief Runs the probe trial: a demonstration, then the gap and a recall's length without X.
		/// @param gap How long a recall trial would wait before X: the evaluation's longest.
		/// @param scores Accumulates the input and quiet output terms.
		/// @param trials Receives the trial's trace.
		/// @return How far the output had no peak after the demonstration, times the mean, over the events, of how far
		/// each had no real peak there (1 minus its best recall score).
		double runProbeTrial(double gap, Scores& scores, nlohmann::json& trials);
		/// @brief Runs one recall trial and scores its recall.
		/// @param number Number of the recall trial, from 1.
		/// @param gap How long the trial waits after the demonstration before X.
		/// @param scores Accumulates every term; the recall terms are scaled by how far the output had no peak before X.
		/// @param trials Receives the trial's trace.
		void runRecallTrial(size_t number, double gap, Scores& scores, nlohmann::json& trials);
		/// @brief Summarises one recall trial for the evaluation trace.
		/// @param number Number of the recall trial, from 1.
		/// @param recall Each event's record for the recall.
		/// @param after What the output did from the end of the sequence on.
		/// @param cue When X was given.
		/// @return RecallBehaviour::recallSummary(), with each event's first peak after X and its timing credit.
		[[nodiscard]] nlohmann::json timedRecallSummary(size_t number, const CycleRecall& recall, const AfterRecall& after,
			double cue) const;
		/// @brief Scores one recall's timing.
		/// @param recall Each event's record for the recall.
		/// @param cue When X was given.
		/// @return The mean, over the events, of each one's credit in place times how close its first peak came to its
		/// demonstrated time after X; 0 for an event that never peaked.
		[[nodiscard]] static double timingOf(const CycleRecall& recall, double cue);
		/// @brief How close one event's first peak came to its demonstrated time after X.
		/// @param recall Each event's record for the recall.
		/// @param event Index of the event in the sequence.
		/// @param cue When X was given.
		/// @return Three quarters a Gaussian of the error, 0.2 of the demonstrated time wide, and one quarter the smaller
		/// of the delay and the demonstrated time over the larger; 0 if the event never peaked.
		[[nodiscard]] static double onTime(const CycleRecall& recall, size_t event, double cue);

		/// @brief When the current recall trial gives X again; none in the probe trial.
		std::optional<double> trialCue;
		/// @brief Weights of the intended behaviour's six terms, from the unguided task's config: the guided task's fitness
		/// is the larger of its own score and the intended behaviour's.
		std::vector<double> intendedFitnessWeights;
	};

	/// @brief SequenceTimingBehaviour that also pays the guides: what the reference design holds below threshold.
	///
	/// @details On top of the intended behaviour:
	/// - through each demonstration and gap, the output is pre-shaped at every stored event, the earliest highest, as far
	///   as the input is at rest and the output has no peak;
	/// - while they wait, the events not yet recalled stay pre-shaped in order, as far as the input is at rest. A recall
	///   where no event waits is not scored;
	/// - a released event earns half its release for lying below threshold and half for lying below the rest of the
	///   field, in full from 2 below;
	/// - once the sequence is over, half the stop is for every event lying below the rest of the field until the recall
	///   ends, each as far as it was recalled in its place.
	///
	/// The eight partial fitness terms are: f1 the input, f2 the quiet output and f3 its pre-shapes before X, f4 the
	/// recall in order, f5 its timing, f6 the pre-shapes still to come, f7 the release of each recalled event and f8 the
	/// output after recall. The fitness is the larger of their weighted sum and what SequenceTimingBehaviour pays for the
	/// same trials, so a design that recalls on time without the guides is not ranked below one that has them; the
	/// trace's "fitness" records both.
	class SequenceTimingBehaviourGuided final : public SequenceTimingBehaviour
	{
	public:
		/// @brief Builds a task solution whose genome is created from @p topology.
		/// @param topology Fields to start from: the input and the output.
		explicit SequenceTimingBehaviourGuided(const SolutionTopology& topology);
		/// @brief Builds a task solution whose genome is decoded from an existing phenotype, such as a template.
		/// @param initialTopology Fields the task declares: the input and the output.
		/// @param phenotype Simulation the genome is translated from.
		SequenceTimingBehaviourGuided(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype);
		/// @brief Creates a solution with a fresh genome built from this one's initial topology and a new id.
		/// @return The new solution.
		SolutionPtr clone() const override;
		/// @brief Creates an independent solution with this one's phenotype and a new id.
		/// @return The new solution.
		SolutionPtr copy() const override;
	};
}
