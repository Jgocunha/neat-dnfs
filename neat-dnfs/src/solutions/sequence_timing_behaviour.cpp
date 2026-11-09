#include "solutions/sequence_timing_behaviour.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <numeric>

#include "neat_tools/config_loader.h"
#include "neat_tools/utils.h"

namespace neat_dnfs
{
	namespace
	{
		using serial_order_schedule::Event;
		using serial_order_schedule::cueStimulusName;
		using serial_order_schedule::sequence;
		using serial_order_schedule::pulseDuration;
		using serial_order_schedule::sampleInterval;
		using serial_order_schedule::eventsShownBy;
		using serial_order_schedule::positionsOf;
		using tools::utils::mean;

		// When the demonstration's last pulse, B's, ends.
		constexpr double demonstrationEnd = sequence.back().onset + pulseDuration;
		// One gap is drawn from each third of this range, so that a fixed delay after the demonstration can match the
		// recall timing in one trial at most.
		constexpr double shortestRecallGap = 300.0;
		constexpr double longestRecallGap = 1500.0;
		// Recall is recorded until 500 after B is due.
		constexpr double recallDuration = sequence.back().onset + 500.0;
		// If some event is never released, the sequence counts as over 300 after B is due, so the stop is always scored.
		constexpr double sequenceOverAfter = sequence.back().onset + 300.0;
		// Scalar timing: the error tolerated grows with the time since X.
		constexpr double weberFraction = 0.2;
		// Far from its time the Gaussian is flat. This share of the timing credit is the delay over the due time (or the
		// due time over the delay, if later), which keeps rising all the way to it.
		constexpr double timingRatioShare = 0.25;
		// The task that pays the intended behaviour alone: its config supplies the weights the guided task also scores
		// by.
		constexpr auto unguidedSlug = "sequence-timing-behaviour";
		constexpr size_t unguidedTermCount = 6;
		constexpr size_t guidedTermCount = 8;
		// The gap of the trial a saved solution replays: the one in the reference's behaviour table.
		constexpr double replayedGap = 700.0;
		// Points of a recall the trace snapshots: inside the reference's peaks at D, C, A and B.
		constexpr std::array<double, RecallBehaviour::eventCount> recallSnapshotTimes{ 140.0, 620.0, 820.0, 1790.0 };

		double meanOrZero(const std::vector<double>& values)
		{
			return values.empty() ? 0.0 : mean(values);
		}
	}

	SequenceTimingBehaviour::SequenceTimingBehaviour(const SolutionTopology& topology)
		: SequenceTimingBehaviour(topology, Guides::unscored, "Sequence Timing Behaviour", unguidedSlug)
	{
	}

	SequenceTimingBehaviour::SequenceTimingBehaviour(const SolutionTopology& initialTopology,
		const dnf_composer::Simulation& phenotype)
		: SequenceTimingBehaviour(initialTopology, phenotype, Guides::unscored, "Sequence Timing Behaviour", unguidedSlug)
	{
	}

	SequenceTimingBehaviour::SequenceTimingBehaviour(const SolutionTopology& topology, const Guides guides,
		const std::string& taskName, const std::string& slug)
		: RecallBehaviour(topology, guides)
	{
		name = taskName;
		loadAllFitnessWeights(slug);
	}

	SequenceTimingBehaviour::SequenceTimingBehaviour(const SolutionTopology& initialTopology,
		const dnf_composer::Simulation& phenotype, const Guides guides, const std::string& taskName,
		const std::string& slug)
		: RecallBehaviour(initialTopology, phenotype, guides)
	{
		name = taskName;
		loadAllFitnessWeights(slug);
	}

	void SequenceTimingBehaviour::loadAllFitnessWeights(const std::string& slug)
	{
		loadFitnessWeights(slug, guides == Guides::scored ? guidedTermCount : unguidedTermCount);
		intendedFitnessWeights = ConfigLoader::loadFitnessWeights(unguidedSlug, unguidedTermCount);
	}

	SolutionPtr SequenceTimingBehaviour::clone() const
	{
		SequenceTimingBehaviour solution(initialTopology);
		return std::make_shared<SequenceTimingBehaviour>(solution);
	}

	SolutionPtr SequenceTimingBehaviour::copy() const
	{
		SequenceTimingBehaviour solution(initialTopology, phenotype);
		return std::make_shared<SequenceTimingBehaviour>(solution);
	}

	void SequenceTimingBehaviour::testPhenotype()
	{
		parameters.fitness = 0.0;
		parameters.partialFitness.clear();

		Scores scores;
		nlohmann::json trials = nlohmann::json::array();
		createPhenotypeEnvironment();
		const auto gaps = drawRecallGaps();
		// The probe waits as long as the longest recall trial, so whatever fires at a fixed lag after the demonstration
		// before any recall ends fires inside it.
		const double probeQuiet = runProbeTrial(gaps.back(), scores, trials);
		for (size_t trial = 0; trial < gaps.size(); ++trial)
		{
			runRecallTrial(trial + 1, gaps[trial], scores, trials);
		}

		const double input = 0.5 * mean(scores.inputPeaks) + 0.5 * mean(scores.inputRest); // detects, then rests
		const double quiet = mean(scores.outputQuiet);              // no peak before X, nor without it
		const double memory = meanOrZero(scores.outputMemory);      // pre-shaped at the stored events, in order
		// A design that fires without X earns no credit for firing after it.
		const double recalled = probeQuiet * mean(scores.recalled); // the events recalled one at a time, in order
		const double timing = probeQuiet * mean(scores.timing);     // each at its demonstrated time after X
		const double toCome = probeQuiet * meanOrZero(scores.toCome); // events still to recall pre-shaped, in order
		const double released = probeQuiet * mean(scores.released); // each recalled event released
		const double stopped = probeQuiet * mean(scores.stopped);   // no peak once the sequence is over

		const std::vector<double> intended{ input, quiet, recalled, timing,
			probeQuiet * mean(scores.releasedWithoutGuides), probeQuiet * mean(scores.stoppedWithoutGuides) };
		const double intendedFitness = std::inner_product(intended.begin(), intended.end(),
			intendedFitnessWeights.begin(), 0.0);
		nlohmann::json trace{ { "trials", trials } };
		if (guides == Guides::scored)
		{
			parameters.partialFitness = { input, quiet, memory, recalled, timing, toCome, released, stopped };
			const double guidedFitness = std::inner_product(parameters.partialFitness.begin(),
				parameters.partialFitness.end(), fitnessWeights.begin(), 0.0);
			// The guides lead the search toward the reference's way of recalling; they must not rank a design that recalls
			// on time another way below one that does not.
			parameters.fitness = std::max(guidedFitness, intendedFitness);
			// Unrounded, unlike the rest of the trace: the fitness is exactly the larger of the two.
			trace["fitness"] = { { "guided", guidedFitness }, { "intended", intendedFitness } };
		}
		else
		{
			parameters.partialFitness = intended;
			parameters.fitness = intendedFitness;
		}
		parameters.evaluationTrace = trace.dump();
	}

	void SequenceTimingBehaviour::createPhenotypeEnvironment()
	{
		const dnf_composer::element::ElementDimensions dimensions{ DimensionConstants::xSize, DimensionConstants::dx };
		const std::array<double, 2> cues{ 0.0, demonstrationEnd + replayedGap };
		for (const auto& [stimulusName, stimulusParameters] : serial_order_schedule::stimuli(cues))
		{
			addTimedGaussianStimulus(stimulusName, inputField, stimulusParameters, dimensions);
		}
		trialCue = cues.back();
	}

	bool SequenceTimingBehaviour::inputSettledAt(const double time) const
	{
		const auto pulsedRecently = [time](const double onset) {
			return time >= onset && time < onset + pulseDuration + settleTime;
		};
		return !pulsedRecently(0.0) && !(trialCue && pulsedRecently(*trialCue))
			&& std::ranges::none_of(sequence, [&pulsedRecently](const Event& event) { return pulsedRecently(event.onset); });
	}

	std::array<double, 3> SequenceTimingBehaviour::drawRecallGaps()
	{
		std::array<double, 3> gaps{};
		const double third = (longestRecallGap - shortestRecallGap) / static_cast<double>(gaps.size());
		for (size_t i = 0; i < gaps.size(); ++i)
		{
			const double from = shortestRecallGap + third * static_cast<double>(i);
			gaps[i] = std::round(tools::utils::generateRandomDouble(from, from + third) / sampleInterval) * sampleInterval;
		}
		return gaps;
	}

	void SequenceTimingBehaviour::setCue(const std::optional<double> cue)
	{
		const auto stimulus = std::dynamic_pointer_cast<dnf_composer::element::TimedGaussStimulus>(
			phenotype.getElement(cueStimulusName));
		auto stimulusParameters = stimulus->getParameters();
		stimulusParameters.onTimes = { { 0.0, pulseDuration } };
		if (cue)
		{
			stimulusParameters.onTimes.emplace_back(*cue, *cue + pulseDuration);
		}
		stimulus->setParameters(stimulusParameters);
		trialCue = cue;
	}

	double SequenceTimingBehaviour::demonstrate(const double until, Scores& scores, nlohmann::json& snapshots)
	{
		initSimulation();
		std::vector<double> quietSamples;
		const auto observe = [&](const double time, const std::optional<double> inputAtRest) {
			const double noPeak = noBumps(outputField);
			scores.outputQuiet.push_back(noPeak);
			quietSamples.push_back(noPeak);
			// Memory is what the output holds below threshold once the input has let go of each event.
			const size_t stored = eventsShownBy(time);
			if (guides == Guides::scored && inputAtRest && stored > 0)
			{
				scores.outputMemory.push_back(preShapedInOrder(positionsOf(0, stored)) * *inputAtRest * noPeak);
			}
		};

		sampleUntil(sequence.front().onset, scores.inputRest, observe);
		snapshots.push_back(snapshot("before D"));
		for (const Event& event : sequence)
		{
			sampleUntil(event.onset + pulseDuration, scores.inputRest, observe);
			snapshots.push_back(snapshot(std::format("{} shown", event.label)));
			scores.inputPeaks.push_back(inputPeakedAt(event.position));
		}
		sampleUntil(until, scores.inputRest, observe);
		return mean(quietSamples);
	}

	double SequenceTimingBehaviour::runProbeTrial(const double gap, Scores& scores, nlohmann::json& trials)
	{
		setCue(std::nullopt);
		nlohmann::json snapshots = nlohmann::json::array();
		const double quietInDemonstration = demonstrate(demonstrationEnd, scores, snapshots);
		std::vector<double> quietWithoutCue;
		// Each event's peaks without X, scored as a recall's are: a burst too brief to show in the share of quiet samples
		// still peaks at every event.
		CycleRecall withoutCue{};
		std::vector<double> unscoredPreShapes;
		sampleUntil(demonstrationEnd + gap + recallDuration, scores.inputRest,
			[&](const double time, const std::optional<double> inputAtRest) {
				const double noPeak = noBumps(outputField);
				scores.outputQuiet.push_back(noPeak);
				quietWithoutCue.push_back(noPeak);
				observeRecall(time, withoutCue, unscoredPreShapes, inputAtRest.value_or(scores.inputRest.back()));
			});
		snapshots.push_back(snapshot("end"));

		nlohmann::json events = nlohmann::json::object();
		std::vector<double> eventsQuiet;
		for (size_t event = 0; event < withoutCue.size(); ++event)
		{
			eventsQuiet.push_back(1.0 - withoutCue[event].recalled);
			events[sequence[event].label] = { { "peak", rounded(withoutCue[event].recalled) } };
		}
		const double quiet = mean(quietWithoutCue);
		trials.push_back({ { "kind", "probe" }, { "gap", gap }, { "quietInDemonstration", rounded(quietInDemonstration) },
			{ "quietWithoutCue", rounded(quiet) }, { "eventsQuietWithoutCue", rounded(mean(eventsQuiet)) },
			{ "events", events }, { "snapshots", snapshots } });
		return quiet * mean(eventsQuiet);
	}

	void SequenceTimingBehaviour::runRecallTrial(const size_t number, const double gap, Scores& scores,
		nlohmann::json& trials)
	{
		const double cue = demonstrationEnd + gap;
		setCue(cue);
		nlohmann::json snapshots = nlohmann::json::array();
		const double quietBeforeCue = demonstrate(cue, scores, snapshots);
		snapshots.push_back(snapshot("before X"));

		CycleRecall recall{};
		AfterRecall after{};
		std::vector<double> waitingPreShapes;
		size_t snapshotsTaken = 0;
		sampleUntil(cue + recallDuration, scores.inputRest, [&](const double time, const std::optional<double> inputAtRest) {
			// While X still lifts the input, it holds whatever it held before X.
			observeRecall(time, recall, waitingPreShapes, inputAtRest.value_or(scores.inputRest.back()));
			if (sequenceReleased(recall) || time >= cue + sequenceOverAfter)
			{
				observeAfterRecall(time, after);
			}
			if (snapshotsTaken < recallSnapshotTimes.size() && time - cue == recallSnapshotTimes[snapshotsTaken])
			{
				snapshots.push_back(snapshot(std::format("recall {}", sequence[snapshotsTaken].label)));
				++snapshotsTaken;
			}
		});
		snapshots.push_back(snapshot("recall end"));

		// An output active before X sweeps peaks past the events: recall counts only as far as it was quiet.
		scores.recalled.push_back(quietBeforeCue * recallInOrder(recall));
		scores.timing.push_back(quietBeforeCue * timingOf(recall, cue));
		// A recall where every event peaked by the first sample had nothing waiting to score.
		if (!waitingPreShapes.empty())
		{
			scores.toCome.push_back(quietBeforeCue * mean(waitingPreShapes));
		}
		scores.released.push_back(quietBeforeCue * releaseAfterRecall(recall));
		scores.stopped.push_back(quietBeforeCue * stoppedAfterRecall(recall, after));
		scores.releasedWithoutGuides.push_back(quietBeforeCue * releaseAfterRecall(recall, Guides::unscored));
		scores.stoppedWithoutGuides.push_back(quietBeforeCue * stoppedAfterRecall(recall, after, Guides::unscored));

		trials.push_back({ { "kind", "recall" }, { "gap", gap }, { "cue", cue },
			{ "quietBeforeCue", rounded(quietBeforeCue) }, { "recall", timedRecallSummary(number, recall, after, cue) },
			{ "snapshots", snapshots } });
	}

	nlohmann::json SequenceTimingBehaviour::timedRecallSummary(const size_t number, const CycleRecall& recall,
		const AfterRecall& after, const double cue) const
	{
		nlohmann::json summary = recallSummary(number, recall, after);
		for (size_t event = 0; event < recall.size(); ++event)
		{
			const auto& firstPeak = recall[event].firstPeak;
			auto& traced = summary["events"][sequence[event].label];
			traced["afterCue"] = firstPeak ? nlohmann::json(rounded(*firstPeak - cue)) : nlohmann::json(nullptr);
			traced["onTime"] = rounded(onTime(recall, event, cue));
		}
		return summary;
	}

	double SequenceTimingBehaviour::timingOf(const CycleRecall& recall, const double cue)
	{
		double credit = 0.0;
		for (size_t event = 0; event < recall.size(); ++event)
		{
			credit += creditInPlace(recall, event) * onTime(recall, event, cue);
		}
		return credit / static_cast<double>(recall.size());
	}

	double SequenceTimingBehaviour::onTime(const CycleRecall& recall, const size_t event, const double cue)
	{
		const auto& firstPeak = recall[event].firstPeak;
		if (!firstPeak)
		{
			return 0.0;
		}
		const double due = sequence[event].onset;
		const double delay = *firstPeak - cue;
		const double nearDue = tools::utils::normalizeWithGaussian(delay, due, weberFraction * due);
		const double ratio = delay > 0.0 ? std::min(delay, due) / std::max(delay, due) : 0.0;
		return (1.0 - timingRatioShare) * nearDue + timingRatioShare * ratio;
	}

	SequenceTimingBehaviourGuided::SequenceTimingBehaviourGuided(const SolutionTopology& topology)
		: SequenceTimingBehaviour(topology, Guides::scored, "Sequence Timing Behaviour Guided",
			"sequence-timing-behaviour-guided")
	{
	}

	SequenceTimingBehaviourGuided::SequenceTimingBehaviourGuided(const SolutionTopology& initialTopology,
		const dnf_composer::Simulation& phenotype)
		: SequenceTimingBehaviour(initialTopology, phenotype, Guides::scored, "Sequence Timing Behaviour Guided",
			"sequence-timing-behaviour-guided")
	{
	}

	SolutionPtr SequenceTimingBehaviourGuided::clone() const
	{
		SequenceTimingBehaviourGuided solution(initialTopology);
		return std::make_shared<SequenceTimingBehaviourGuided>(solution);
	}

	SolutionPtr SequenceTimingBehaviourGuided::copy() const
	{
		SequenceTimingBehaviourGuided solution(initialTopology, phenotype);
		return std::make_shared<SequenceTimingBehaviourGuided>(solution);
	}
}
