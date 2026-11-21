#include "solutions/sequence_and_timing_recall.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <numeric>
#include <random>

#include <dnf_composer/elements/timed_gauss_stimulus.h>

#include "neat_tools/config_loader.h"
#include "neat_tools/utils.h"

namespace neat_dnfs
{
	namespace
	{
		using tools::utils::mean;

		// One demonstrated event: a Gaussian pulse on the input field, named "tgs <label>".
		struct Event
		{
			const char* label;
			double position;
			double onset;
		};

		// The demonstrated sequence of joint goals: recall must replay it in this order, each event at its onset after the
		// trigger. The onsets are the first six of the ten-event task's, so the pace is as demanding as it was there. On a
		// field of 200, the goals recalled one after another lie 40-80 apart, and the last two 20 apart.
		constexpr std::array<Event, SequenceAndTimingRecall::eventCount> sequence{ { { "BSA", 20.0, 50.0 },
			{ "BLA", 100.0, 150.0 }, { "BSB", 60.0, 350.0 }, { "BLB", 120.0, 450.0 }, { "BS", 160.0, 650.0 },
			{ "BP", 180.0, 750.0 } } };

		const std::string inputField{ "nf 1" };
		const std::string outputField{ "nf 2" };
		// The memory recall trigger: a pulse wide enough to lift the whole input field evenly, as a boost would, as a
		// stimulus a saved file can replay.
		const std::string cueStimulusName{ "tgs memory recall trigger" };
		constexpr double cueWidth = 1e6;
		// How long each pulse, event or trigger, stays on.
		constexpr double pulseDuration = 50.0;
		// How long after a pulse ends the input's peak may take to decay before it must be back at rest: five time
		// constants of the fixed input field.
		constexpr double settleTime = 50.0;
		constexpr double sampleInterval = 10.0;

		// When the demonstration's last pulse, BP's, ends.
		constexpr double demonstrationEnd = sequence.back().onset + pulseDuration;
		// One wait is drawn from each third of this range, so that a fixed delay after the demonstration can match the
		// recall timing in one trial at most.
		constexpr double shortestRecallWait = 300.0;
		constexpr double longestRecallWait = 1500.0;
		// Recall is recorded until 500 after BP is due.
		constexpr double recallDuration = sequence.back().onset + 500.0;
		// After a recall window that no trigger follows, the output is watched this much longer for silence: a peak
		// there is a peak after the sequence.
		constexpr double silenceProbeDuration = 1000.0;
		// If some event is never released, the sequence counts as over 300 after BP is due, so the stop is always
		// scored.
		constexpr double sequenceOverAfter = sequence.back().onset + 300.0;
		// Scalar timing: the error tolerated grows with the time since the trigger. An event within it is on time.
		constexpr double weberFraction = 0.2;
		// Past the window's edge the credit falls off this steeply, as a share of the due time, so that an event moved
		// just out of its window loses most of it.
		constexpr double timingFallOffFraction = 0.05;
		// Far from its window the Gaussian is flat. This share of the timing credit is the delay over the window's edge
		// (or the edge over the delay, if later), which keeps rising all the way to it.
		constexpr double timingRatioShare = 0.25;
		// Full order credit needs each event to peak at least this share of its demonstrated gap after each event due
		// before it: events crowded together in time are not recalled in order.
		constexpr double orderSpacingShare = 0.5;
		// The wait of the trial a saved solution replays: the one in the reference's behaviour table.
		constexpr double replayedWait = 900.0;
		// Each evaluation runs this many consecutive trials, the first wait of each from its own third of the range.
		constexpr size_t consecutiveTrialCount = 3;

		constexpr auto taskSlug = "sequence-and-timing-recall";
		// Holds only the intended behaviour's weights: it is no task of its own.
		constexpr auto intendedSlug = "sequence-and-timing-recall-intended";
		constexpr size_t guidedTermCount = 8;
		constexpr size_t intendedTermCount = 6;

		// What belowThresholdAtPosition() reads at threshold: an event's place is above threshold below this.
		constexpr double atThreshold = 0.5;
		// A bump near an event still counts as its peak from this peak score: about a quarter of full height or more.
		constexpr double peakingScore = 0.25;
		// While an event is up, the bump over its place counts as a peak in full up to this wide, and not at all from the
		// next: the reference's peaks, and those of most evolved designs that recall in order, stay 14 wide or less.
		constexpr double fullCreditWidthWhileUp = 15.0;
		constexpr double noCreditWidthWhileUp = 25.0;
		// A recalled peak stays where it is: its centre may move this much from one sample to the next for full credit,
		constexpr double stillPeakDrift = 1.0;
		// and earns nothing from this much on, where a bump travelling around the field would credit each event it passes.
		constexpr double travellingPeakDrift = 3.0;
		// The nearest bump at two successive samples is the same bump only this close: the events lie 20 apart, so a
		// peak at the next event is a new bump, not a moved one.
		constexpr double sameBumpReach = 10.0;
		// The output's level is read only farther than this from every event. The events lie 20 apart around the whole
		// field, so these are the places midway between two events, 9 and 10 from both: as far as any place lies.
		constexpr double baselineReach = 8.0;
		// Pre-shapes in order are a staircase: each event's step above the next, the last one's above the output's
		// level, counts in full from this height, twenty-five times the fields' noise.
		constexpr double preShapeStep = 0.25;
		// The input's peak at a shown event, in the fixed input field.
		constexpr double inputPeakAmplitude = 7.6;
		constexpr double inputPeakWidth = 7.0;

		double meanOrZero(const std::vector<double>& values)
		{
			return values.empty() ? 0.0 : mean(values);
		}

		double aroundTheField(const double from, const double to)
		{
			const double apart = std::abs(to - from);
			return std::min(apart, static_cast<double>(DimensionConstants::xSize) - apart);
		}

		// Every place on the field, as dnf-composer samples it, farther than baselineReach from every event.
		std::vector<double> farFromEveryEvent()
		{
			const auto samples = static_cast<int>(std::lround(DimensionConstants::xSize / DimensionConstants::dx));
			std::vector<double> positions;
			for (int sample = 1; sample <= samples; ++sample)
			{
				const double position = sample * DimensionConstants::dx;
				if (std::ranges::all_of(sequence,
					[position](const Event& event) { return aroundTheField(position, event.position) > baselineReach; }))
				{
					positions.push_back(position);
				}
			}
			return positions;
		}

		size_t eventsShownBy(const double time)
		{
			return static_cast<size_t>(std::ranges::count_if(sequence, [time](const Event& event) { return event.onset < time; }));
		}

		std::vector<double> positionsOf(const size_t count)
		{
			std::vector<double> positions;
			for (size_t event = 0; event < count; ++event)
			{
				positions.push_back(sequence[event].position);
			}
			return positions;
		}
	}

	SequenceAndTimingRecall::SequenceAndTimingRecall(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "Sequence And Timing Recall";
		loadFitnessWeights(taskSlug, guidedTermCount);
		intendedFitnessWeights = ConfigLoader::loadFitnessWeights(intendedSlug, intendedTermCount);
	}

	SequenceAndTimingRecall::SequenceAndTimingRecall(const SolutionTopology& initialTopology,
		const dnf_composer::Simulation& phenotype)
		: Solution(initialTopology, phenotype)
	{
		name = "Sequence And Timing Recall";
		loadFitnessWeights(taskSlug, guidedTermCount);
		intendedFitnessWeights = ConfigLoader::loadFitnessWeights(intendedSlug, intendedTermCount);
	}

	SolutionPtr SequenceAndTimingRecall::clone() const
	{
		SequenceAndTimingRecall solution(initialTopology);
		return std::make_shared<SequenceAndTimingRecall>(solution);
	}

	SolutionPtr SequenceAndTimingRecall::copy() const
	{
		SequenceAndTimingRecall solution(initialTopology, phenotype);
		return std::make_shared<SequenceAndTimingRecall>(solution);
	}

	void SequenceAndTimingRecall::testPhenotype()
	{
		parameters.fitness = 0.0;
		parameters.partialFitness.clear();

		Scores scores;
		nlohmann::json trials = nlohmann::json::array();
		baselinePositions = farFromEveryEvent();
		createPhenotypeEnvironment();
		// Seeded, every solution of a generation meets the same waits; unseeded, the waits are drawn afresh.
		tools::utils::Xoshiro256pp seededEngine{ evaluationSeed.value_or(0) };
		tools::utils::Xoshiro256pp& waitEngine = evaluationSeed ? seededEngine : tools::utils::engine();
		std::array<std::array<double, 3>, consecutiveTrialCount> consecutiveWaits{};
		for (size_t trial = 0; trial < consecutiveWaits.size(); ++trial)
		{
			consecutiveWaits[trial] = drawConsecutiveWaits(trial, waitEngine);
		}
		// The probe watches as long as the longest consecutive trial, so whatever fires at a fixed lag after the
		// demonstration inside any recall window fires inside it.
		double longestTrial = 0.0;
		for (const auto& waits : consecutiveWaits)
		{
			longestTrial = std::max(longestTrial, std::accumulate(waits.begin(), waits.end(), 0.0)
				+ static_cast<double>(waits.size()) * recallDuration);
		}
		const double probeQuiet = runProbeTrial(consecutiveWaits.back().front(), longestTrial, scores, trials);
		for (const auto& waits : consecutiveWaits)
		{
			runConsecutiveTrial(waits, scores, trials);
		}

		const double input = 0.5 * mean(scores.inputPeaks) + 0.5 * mean(scores.inputRest); // detects, then rests
		const double quiet = mean(scores.outputQuiet);              // no peak before the trigger, nor without it
		const double memory = meanOrZero(scores.outputMemory);      // pre-shaped at the stored events, in order
		// A design that fires without the trigger earns no credit for firing after it. A later recall counts only as far
		// as its trial's first recall went (runConsecutiveTrial).
		const double recalled = probeQuiet * mean(scores.recalled); // the events recalled in order
		const double timing = probeQuiet * mean(scores.timing);     // each at its demonstrated time after the trigger
		const double toCome = probeQuiet * meanOrZero(scores.toCome); // the events still to recall pre-shaped, in order
		const double released = probeQuiet * mean(scores.released); // each recalled event released
		const double stopped = probeQuiet * mean(scores.stopped);   // no peak once the sequence is over

		parameters.partialFitness = { input, quiet, memory, recalled, timing, toCome, released, stopped };
		const double guidedFitness = std::inner_product(parameters.partialFitness.begin(), parameters.partialFitness.end(),
			fitnessWeights.begin(), 0.0);
		const std::vector<double> intended{ input, quiet, recalled, timing,
			probeQuiet * mean(scores.releasedWithoutGuides), probeQuiet * mean(scores.stoppedWithoutGuides) };
		const double intendedFitness = std::inner_product(intended.begin(), intended.end(),
			intendedFitnessWeights.begin(), 0.0);
		// The guides lead the search toward the reference's way of recalling; they must not rank a design that recalls
		// on time another way below one that does not.
		parameters.fitness = std::max(guidedFitness, intendedFitness);
		// Unrounded, unlike the rest of the trace: the fitness is exactly the larger of the two.
		const nlohmann::json trace{ { "fitness", { { "guided", guidedFitness }, { "intended", intendedFitness } } },
			{ "trials", trials } };
		parameters.evaluationTrace = trace.dump();
	}

	void SequenceAndTimingRecall::createPhenotypeEnvironment()
	{
		using dnf_composer::element::TimedGaussStimulusParameters;
		const dnf_composer::element::ElementDimensions dimensions{ DimensionConstants::xSize, DimensionConstants::dx };
		const double cue = demonstrationEnd + replayedWait;
		addTimedGaussianStimulus(cueStimulusName, inputField, TimedGaussStimulusParameters{ cueWidth,
			GaussStimulusConstants::amplitude, DimensionConstants::xSize / 2.0,
			{ { 0.0, pulseDuration }, { cue, cue + pulseDuration } }, false, false }, dimensions);
		for (const Event& event : sequence)
		{
			addTimedGaussianStimulus(std::format("tgs {}", event.label), inputField, TimedGaussStimulusParameters{
				GaussStimulusConstants::width, GaussStimulusConstants::amplitude, event.position,
				{ { event.onset, event.onset + pulseDuration } }, GaussStimulusConstants::circularity,
				GaussStimulusConstants::normalization }, dimensions);
		}
		trialCues = { cue };
	}

	std::array<double, 3> SequenceAndTimingRecall::drawRecallWaits(tools::utils::Xoshiro256pp& engine)
	{
		std::array<double, 3> waits{};
		const double third = (longestRecallWait - shortestRecallWait) / static_cast<double>(waits.size());
		for (size_t i = 0; i < waits.size(); ++i)
		{
			const double from = shortestRecallWait + third * static_cast<double>(i);
			std::uniform_real_distribution<double> withinThird(from, from + third);
			waits[i] = std::round(withinThird(engine) / sampleInterval) * sampleInterval;
		}
		return waits;
	}

	std::array<double, 3> SequenceAndTimingRecall::drawConsecutiveWaits(const size_t firstThird,
		tools::utils::Xoshiro256pp& engine)
	{
		auto waits = drawRecallWaits(engine);
		std::swap(waits.front(), waits[firstThird]);
		// Shuffled, so that no fixed delay after the previous recall matches every wait.
		std::ranges::shuffle(waits.begin() + 1, waits.end(), engine);
		return waits;
	}

	void SequenceAndTimingRecall::setCues(const std::vector<double>& cues)
	{
		const auto stimulus = std::dynamic_pointer_cast<dnf_composer::element::TimedGaussStimulus>(
			phenotype.getElement(cueStimulusName));
		auto stimulusParameters = stimulus->getParameters();
		stimulusParameters.onTimes = { { 0.0, pulseDuration } };
		for (const double cue : cues)
		{
			stimulusParameters.onTimes.emplace_back(cue, cue + pulseDuration);
		}
		stimulus->setParameters(stimulusParameters);
		trialCues = cues;
	}

	bool SequenceAndTimingRecall::inputSettledAt(const double time) const
	{
		const auto pulsedRecently = [time](const double onset) {
			return time >= onset && time < onset + pulseDuration + settleTime;
		};
		return !pulsedRecently(0.0) && std::ranges::none_of(trialCues, pulsedRecently)
			&& std::ranges::none_of(sequence, [&pulsedRecently](const Event& event) { return pulsedRecently(event.onset); });
	}

	void SequenceAndTimingRecall::runUntil(const double time)
	{
		while (phenotype.getT() < time - 1e-9)
		{
			runSimulation(1);
		}
	}

	void SequenceAndTimingRecall::sampleUntil(const double time, std::vector<double>& inputRest,
		const std::function<void(double, std::optional<double>)>& onSample)
	{
		// Samples fall on whole multiples of the interval, so every stage's samples line up with the schedule.
		for (auto sample = static_cast<long>(std::floor(phenotype.getT() / sampleInterval + 1e-9)) + 1;
			static_cast<double>(sample) * sampleInterval <= time + 1e-9; ++sample)
		{
			const double sampleTime = static_cast<double>(sample) * sampleInterval;
			runUntil(sampleTime);
			std::optional<double> inputAtRest;
			if (inputSettledAt(sampleTime))
			{
				inputAtRest = closenessOfMeanActivationToRestingLevel(inputField);
				inputRest.push_back(*inputAtRest);
			}
			onSample(sampleTime, inputAtRest);
		}
		runUntil(time);
	}

	double SequenceAndTimingRecall::demonstrate(const double until, Scores& scores, nlohmann::json& snapshots)
	{
		initSimulation();
		std::vector<double> quietSamples;
		const auto observe = [&](const double time, const std::optional<double> inputAtRest) {
			const double noPeak = noBumps(outputField);
			scores.outputQuiet.push_back(noPeak);
			quietSamples.push_back(noPeak);
			// Memory is what the output holds below threshold once the input has let go of each event.
			const size_t stored = eventsShownBy(time);
			if (inputAtRest && stored > 0)
			{
				scores.outputMemory.push_back(preShapedInOrder(positionsOf(stored), baseline()) * *inputAtRest * noPeak);
			}
		};

		sampleUntil(sequence.front().onset, scores.inputRest, observe);
		snapshots.push_back(snapshot(std::format("before {}", sequence.front().label)));
		for (const Event& event : sequence)
		{
			sampleUntil(event.onset + pulseDuration, scores.inputRest, observe);
			snapshots.push_back(snapshot(std::format("{} shown", event.label)));
			scores.inputPeaks.push_back(oneBumpAtPositionWithAmplitudeAndWidth(inputField, event.position,
				inputPeakAmplitude, inputPeakWidth));
		}
		sampleUntil(until, scores.inputRest, observe);
		return mean(quietSamples);
	}

	double SequenceAndTimingRecall::runProbeTrial(const double wait, const double watchFor, Scores& scores,
		nlohmann::json& trials)
	{
		setCues({});
		nlohmann::json snapshots = nlohmann::json::array();
		const double quietInDemonstration = demonstrate(demonstrationEnd, scores, snapshots);
		std::vector<double> quietWithoutCue;
		// Each event's peaks without the trigger, scored as a recall's are: a burst too brief to show in the share of
		// quiet samples still peaks at every event.
		Recall withoutCue{};
		std::vector<double> unscoredPreShapes;
		sampleUntil(demonstrationEnd + watchFor + silenceProbeDuration, scores.inputRest,
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
			eventsQuiet.push_back(1.0 - withoutCue[event].bestScore);
			events[sequence[event].label] = { { "peak", rounded(withoutCue[event].bestScore) } };
		}
		const double quiet = mean(quietWithoutCue);
		trials.push_back({ { "kind", "probe" }, { "wait", wait }, { "watchedFor", watchFor + silenceProbeDuration },
			{ "quietInDemonstration", rounded(quietInDemonstration) },
			{ "quietWithoutCue", rounded(quiet) }, { "eventsQuietWithoutCue", rounded(mean(eventsQuiet)) },
			{ "events", events }, { "snapshots", snapshots } });
		return quiet * mean(eventsQuiet);
	}

	void SequenceAndTimingRecall::runConsecutiveTrial(const std::array<double, 3>& waits, Scores& scores,
		nlohmann::json& trials)
	{
		std::vector<double> cues;
		double windowEnd = demonstrationEnd;
		for (const double wait : waits)
		{
			cues.push_back(windowEnd + wait);
			windowEnd = cues.back() + recallDuration;
		}
		setCues(cues);
		nlohmann::json snapshots = nlohmann::json::array();
		nlohmann::json recalls = nlohmann::json::array();
		double quietBeforeCue = demonstrate(cues.front(), scores, snapshots);
		RecallTerms firstRecall{};
		for (size_t k = 0; k < cues.size(); ++k)
		{
			if (k > 0)
			{
				std::vector<double> quietSamples;
				sampleUntil(cues[k], scores.inputRest, [&](double, std::optional<double>) {
					const double noPeak = noBumps(outputField);
					scores.outputQuiet.push_back(noPeak);
					quietSamples.push_back(noPeak);
				});
				quietBeforeCue = mean(quietSamples);
			}
			snapshots.push_back(snapshot(std::format("before trigger {}", k + 1)));
			const bool lastRecall = k + 1 == cues.size();
			RecallTerms terms{};
			const nlohmann::json recall = recallAfterCue(cues[k], quietBeforeCue, lastRecall, terms, scores, snapshots);
			if (k == 0)
			{
				firstRecall = terms;
			}
			addRecall(k == 0 ? terms : countedAsFarAsFirst(terms, firstRecall), scores);
			recalls.push_back({ { "wait", waits[k] }, { "cue", cues[k] }, { "quietBeforeCue", rounded(quietBeforeCue) },
				{ "recall", recall } });
		}
		trials.push_back({ { "kind", "consecutive" }, { "recalls", recalls }, { "snapshots", snapshots } });
	}

	nlohmann::json SequenceAndTimingRecall::recallAfterCue(const double cue, const double quietBeforeCue,
		const bool probeSilence, RecallTerms& terms, Scores& scores, nlohmann::json& snapshots)
	{
		Recall recall{};
		AfterRecall after{};
		std::vector<double> waitingPreShapes;
		size_t snapshotsTaken = 0;
		sampleUntil(cue + recallDuration, scores.inputRest, [&](const double time, const std::optional<double> inputAtRest) {
			// While the trigger still lifts the input, it holds whatever it held before.
			observeRecall(time, recall, waitingPreShapes, inputAtRest.value_or(scores.inputRest.back()));
			if (sequenceReleased(recall) || time >= cue + sequenceOverAfter)
			{
				observeAfterRecall(time, after);
			}
			if (snapshotsTaken < sequence.size() && std::abs(time - cue - sequence[snapshotsTaken].onset) < 1e-9)
			{
				snapshots.push_back(snapshot(std::format("recall {}", sequence[snapshotsTaken].label)));
				++snapshotsTaken;
			}
		});
		snapshots.push_back(snapshot("recall end"));
		if (probeSilence)
		{
			// Only silence is watched after the window: an event first peaking there is not recalled.
			sampleUntil(cue + recallDuration + silenceProbeDuration, scores.inputRest,
				[&](double, std::optional<double>) { after.quiet.push_back(noBumps(outputField)); });
		}

		// An output active before the trigger sweeps peaks past the events: recall counts only as far as it was quiet.
		terms = { .recalled = quietBeforeCue * recallInOrder(recall),
			.timing = quietBeforeCue * timingOf(recall, cue),
			.released = quietBeforeCue * releaseAfterRecall(recall, true),
			.stopped = quietBeforeCue * stoppedAfterRecall(recall, after, true),
			.releasedWithoutGuides = quietBeforeCue * releaseAfterRecall(recall, false),
			.stoppedWithoutGuides = quietBeforeCue * stoppedAfterRecall(recall, after, false) };
		// A recall where every event peaked by the first sample had nothing waiting to score.
		if (!waitingPreShapes.empty())
		{
			scores.toCome.push_back(quietBeforeCue * mean(waitingPreShapes));
		}
		return recallSummary(recall, after, cue);
	}

	SequenceAndTimingRecall::RecallTerms SequenceAndTimingRecall::countedAsFarAsFirst(const RecallTerms& later,
		const RecallTerms& first)
	{
		return { .recalled = later.recalled * first.recalled,
			.timing = later.timing * first.timing,
			.released = later.released * first.released,
			.stopped = later.stopped * first.stopped,
			.releasedWithoutGuides = later.releasedWithoutGuides * first.releasedWithoutGuides,
			.stoppedWithoutGuides = later.stoppedWithoutGuides * first.stoppedWithoutGuides };
	}

	void SequenceAndTimingRecall::addRecall(const RecallTerms& terms, Scores& scores)
	{
		scores.recalled.push_back(terms.recalled);
		scores.timing.push_back(terms.timing);
		scores.released.push_back(terms.released);
		scores.stopped.push_back(terms.stopped);
		scores.releasedWithoutGuides.push_back(terms.releasedWithoutGuides);
		scores.stoppedWithoutGuides.push_back(terms.stoppedWithoutGuides);
	}

	double SequenceAndTimingRecall::baseline() const
	{
		return medianActivationAtPositions(outputField, baselinePositions);
	}

	double SequenceAndTimingRecall::preShapedInOrder(const std::vector<double>& positions, const double level) const
	{
		std::vector<double> activations;
		activations.reserve(positions.size());
		for (const double position : positions)
		{
			activations.push_back(activationAtPosition(outputField, position));
		}
		auto steps = tools::utils::stepsDown(activations, level, preShapeStep);
		// An event at or above threshold is peaking, not pre-shaped.
		for (size_t event = 0; event < steps.size(); ++event)
		{
			if (activations[event] >= 0.0)
			{
				steps[event] = 0.0;
			}
		}
		return mean(steps);
	}

	double SequenceAndTimingRecall::peakAt(const double position) const
	{
		// Count a bump only near the event, in proportion to its height, and only as narrow as a peak: a field-wide
		// flash with a gap reads as one bump opposite the gap.
		return closenessOfNearestBumpToPosition(outputField, position)
			* heightCreditOfNearestBumpToPosition(outputField, position)
			* widthCreditOfNearestBumpToPosition(outputField, position);
	}

	double SequenceAndTimingRecall::movedSince(const std::optional<double> lastCentroid, const std::optional<double> centroid)
	{
		if (!lastCentroid || !centroid)
		{
			return 0.0;
		}
		const double moved = aroundTheField(*lastCentroid, *centroid);
		return moved > sameBumpReach ? 0.0 : moved;
	}

	double SequenceAndTimingRecall::stayingInPlace(const double lastMove, const double move)
	{
		return 1.0 - tools::utils::normalize(std::min(lastMove, move), stillPeakDrift, travellingPeakDrift);
	}

	void SequenceAndTimingRecall::observeRecall(const double time, Recall& recall, std::vector<double>& waitingPreShapes,
		const double inputAtRest) const
	{
		const double level = baseline();
		std::vector<double> waiting;
		for (size_t event = 0; event < sequence.size(); ++event)
		{
			const double position = sequence[event].position;
			EventRecall& observed = recall[event];
			const std::optional<double> centroid = centroidOfNearestBumpToPosition(outputField, position);
			const double move = movedSince(observed.lastCentroid, centroid);
			// Overlap is allowed: each event's score is its own, however many peaks are up with it.
			const double score = peakAt(position) * stayingInPlace(observed.lastMove, move);
			observed.lastCentroid = centroid;
			observed.lastMove = move;
			observed.bestScore = std::max(observed.bestScore, score);
			// An event fires when its place first goes above threshold, whatever bump takes it there: a bump over several
			// places fires them all at once, however it splits into peaks centred on each later.
			const bool aboveThreshold = belowThresholdAtPosition(outputField, position) < atThreshold;
			if (aboveThreshold && !observed.firstPeak)
			{
				observed.firstPeak = time;
			}
			// A released peak can split into flanks either side of its place, its centre down first: it has ended only
			// once they are gone too.
			const bool peakEnded = !aboveThreshold && score < peakingScore;
			if (observed.firstPeak && peakEnded && !observed.releasedAt)
			{
				observed.releasedAt = time;
			}
			// A bump over several places is narrow at some moment, as it ignites or once it splits: an event's peak is
			// judged at its widest while it is up.
			if (observed.firstPeak && !observed.releasedAt)
			{
				observed.widthWhileUp = std::min(observed.widthWhileUp, widthCreditOfNearestBumpToPosition(outputField,
					position, fullCreditWidthWhileUp, noCreditWidthWhileUp));
			}
			if (observed.releasedAt)
			{
				observed.belowThreshold.push_back(belowThresholdAtPosition(outputField, position));
				observed.troughs.push_back(troughBelowLevelAtPosition(outputField, position, level));
			}
			if (!observed.firstPeak)
			{
				waiting.push_back(position);
			}
		}
		if (!waiting.empty())
		{
			// The input only detects events: what it holds is not memory.
			waitingPreShapes.push_back(preShapedInOrder(waiting, level) * inputAtRest);
		}
	}

	void SequenceAndTimingRecall::observeAfterRecall(const double time, AfterRecall& after) const
	{
		if (!after.from)
		{
			after.from = time;
		}
		after.quiet.push_back(noBumps(outputField));
		const double level = baseline();
		for (size_t event = 0; event < sequence.size(); ++event)
		{
			after.troughs[event].push_back(troughBelowLevelAtPosition(outputField, sequence[event].position, level));
		}
	}

	bool SequenceAndTimingRecall::sequenceReleased(const Recall& recall)
	{
		return std::ranges::all_of(recall, [](const EventRecall& observed) { return observed.releasedAt.has_value(); });
	}

	double SequenceAndTimingRecall::placeOf(const Recall& recall, const size_t event)
	{
		std::array<std::optional<double>, eventCount> firstPeaks;
		std::ranges::transform(recall, firstPeaks.begin(), &EventRecall::firstPeak);
		std::array<double, eventCount> dueTimes{};
		std::ranges::transform(sequence, dueTimes.begin(), &Event::onset);
		return tools::utils::worstPairInPlace(firstPeaks, dueTimes, event, orderSpacingShare);
	}

	double SequenceAndTimingRecall::creditInPlace(const Recall& recall, const size_t event)
	{
		return recall[event].bestScore * recall[event].widthWhileUp * placeOf(recall, event);
	}

	double SequenceAndTimingRecall::recallInOrder(const Recall& recall)
	{
		double credit = 0.0;
		for (size_t event = 0; event < recall.size(); ++event)
		{
			credit += creditInPlace(recall, event);
		}
		return credit / static_cast<double>(recall.size());
	}

	double SequenceAndTimingRecall::onTime(const Recall& recall, const size_t event, const double cue)
	{
		const auto& firstPeak = recall[event].firstPeak;
		if (!firstPeak)
		{
			return 0.0;
		}
		const double due = sequence[event].onset;
		const double delay = *firstPeak - cue;
		const double tolerance = weberFraction * due;
		const double beyond = std::abs(delay - due) - tolerance;
		if (beyond <= 0.0)
		{
			return 1.0;
		}
		const double edge = delay > due ? due + tolerance : due - tolerance;
		const double nearEdge = tools::utils::normalizeWithGaussian(beyond, 0.0, timingFallOffFraction * due);
		const double ratio = delay > 0.0 ? std::min(delay, edge) / std::max(delay, edge) : 0.0;
		return (1.0 - timingRatioShare) * nearEdge + timingRatioShare * ratio;
	}

	double SequenceAndTimingRecall::timingOf(const Recall& recall, const double cue)
	{
		// Only the events that fired are timed, so a recall that stops early earns timing for what it recalled, while
		// one event late among them still costs it all.
		double worst = 1.0;
		size_t fired = 0;
		for (size_t event = 0; event < recall.size(); ++event)
		{
			if (!recall[event].firstPeak)
			{
				continue;
			}
			++fired;
			worst = std::min(worst, creditInPlace(recall, event) * onTime(recall, event, cue));
		}
		return fired == 0 ? 0.0 : worst * static_cast<double>(fired) / static_cast<double>(recall.size());
	}

	double SequenceAndTimingRecall::releaseAfterRecall(const Recall& recall, const bool withGuides)
	{
		double credit = 0.0;
		for (size_t event = 0; event < recall.size(); ++event)
		{
			const EventRecall& observed = recall[event];
			if (observed.belowThreshold.empty())
			{
				continue;
			}
			const double release = withGuides ? 0.5 * mean(observed.belowThreshold) + 0.5 * mean(observed.troughs)
				: mean(observed.belowThreshold);
			credit += creditInPlace(recall, event) * release;
		}
		return credit / static_cast<double>(recall.size());
	}

	double SequenceAndTimingRecall::stoppedAfterRecall(const Recall& recall, const AfterRecall& after,
		const bool withGuides)
	{
		// An output that never peaked has nothing to stop: its silence counts only as far as the recall was in order.
		const double noPeak = mean(after.quiet) * recallInOrder(recall);
		if (!withGuides)
		{
			return noPeak;
		}
		double troughs = 0.0;
		for (size_t event = 0; event < recall.size(); ++event)
		{
			troughs += creditInPlace(recall, event) * mean(after.troughs[event]);
		}
		return 0.5 * noPeak + 0.5 * troughs / static_cast<double>(recall.size());
	}

	nlohmann::json SequenceAndTimingRecall::recallSummary(const Recall& recall, const AfterRecall& after,
		const double cue) const
	{
		const auto timeOrNull = [](const std::optional<double>& time) {
			return time ? nlohmann::json(rounded(*time)) : nlohmann::json(nullptr);
		};
		nlohmann::json events = nlohmann::json::object();
		for (size_t event = 0; event < recall.size(); ++event)
		{
			const EventRecall& observed = recall[event];
			events[sequence[event].label] = {
				{ "firstPeak", timeOrNull(observed.firstPeak) },
				{ "afterCue", observed.firstPeak ? nlohmann::json(rounded(*observed.firstPeak - cue)) : nlohmann::json(nullptr) },
				{ "releasedAt", timeOrNull(observed.releasedAt) },
				{ "inPlace", rounded(placeOf(recall, event)) },
				{ "bestScore", rounded(observed.bestScore) },
				{ "widthWhileUp", rounded(observed.widthWhileUp) },
				{ "onTime",rounded(onTime(recall, event, cue)) },
				{ "release", observed.belowThreshold.empty() ? nlohmann::json(nullptr)
					: nlohmann::json(rounded(mean(observed.belowThreshold))) }
			};
		}
		const nlohmann::json afterSequence = { { "from", timeOrNull(after.from) }, { "quiet", rounded(mean(after.quiet)) } };
		return { { "events", events }, { "afterSequence", afterSequence } };
	}

	nlohmann::json SequenceAndTimingRecall::snapshot(const std::string& stage) const
	{
		const auto bumpsOf = [this](const std::string& fieldName) {
			nlohmann::json bumps = nlohmann::json::array();
			for (const auto& bump : getNeuralFieldOrThrow(fieldName, "snapshot")->getBumps())
			{
				bumps.push_back({ rounded(bump.centroid), rounded(bump.amplitude), rounded(bump.width) });
			}
			return bumps;
		};
		nlohmann::json atEvents = nlohmann::json::object();
		for (const Event& event : sequence)
		{
			atEvents[event.label] = rounded(activationAtPosition(outputField, event.position));
		}
		return {
			{ "stage", stage },
			{ "t", rounded(phenotype.getT()) },
			{ "input", { { "bumps", bumpsOf(inputField) } } },
			{ "output", { { "bumps", bumpsOf(outputField) }, { "baseline", rounded(baseline()) },
				{ "atEvents", atEvents } } }
		};
	}

	double SequenceAndTimingRecall::rounded(const double value)
	{
		return std::round(value * 100.0) / 100.0;
	}
}
