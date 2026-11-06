#include "solutions/incremental_recall.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <iterator>
#include <numeric>
#include <set>
#include <tuple>

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

		// The demonstrated sequence (the user's table of 2 November 2026): 25 apart around the field, at uneven
		// intervals. Recall must replay it in this order.
		constexpr std::array<Event, IncrementalRecall<RecallStage::singleOrder>::eventCount> sequence{ {
			{ "E1", 40.0, 100.0 }, { "E2", 90.0, 250.0 }, { "E3", 15.0, 600.0 }, { "E4", 65.0, 1200.0 } } };

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

		constexpr double demonstrationEnd = sequence.back().onset + pulseDuration;
		// Recall is recorded until 500 after E4 is due.
		constexpr double recallDuration = sequence.back().onset + 500.0;
		// The probe watches the output without the trigger this long after the demonstration: as long as the longest wait
		// of the later stages (1500) and a recall window, so a design that recalls by itself later than stage 1 waits is
		// seen.
		constexpr double probeHold = 1500.0 + recallDuration;
		// Once a recall is recorded, the output is watched for silence as long after the trigger as the probe watches
		// without it, so activity that comes back long after the recall costs the stop.
		constexpr double silenceWatched = probeHold;
		// If some event is never released, the sequence counts as over 300 after E4 is due, so the stop is always
		// scored.
		constexpr double sequenceOverAfter = sequence.back().onset + 300.0;

		// The user's peak-width rule for this test (2 November 2026): the events lie 25 apart on a field of 100.
		constexpr double fullCreditPeakWidth = 20.0;
		constexpr double noCreditPeakWidth = 40.0;
		// An event counts as peaking from this peak score: one bump near it, about a quarter of full height or more.
		constexpr double peakingScore = 0.25;
		// A recalled peak stays where it is: its centre may move this much from one sample to the next for full credit,
		constexpr double stillPeakDrift = 1.0;
		// and earns nothing from this much on, where a bump travelling around the field would credit each event it passes.
		constexpr double travellingPeakDrift = 3.0;
		// The nearest bump at two successive samples is the same bump only this close: the events lie 25 apart, so a
		// peak at the next event is a new bump, not a moved one.
		constexpr double sameBumpReach = 10.0;
		// From stage 5 a recall earns no one-at-a-time credit from this many samples with two peaks up together (the user's
		// rule: one at a time), each costing an equal share before it.
		constexpr double noCreditSharedSamples = 3.0;
		// From stage 5 a recall earns no release credit from this many samples with an event peaking again after its
		// release (the user's rule, 4 November 2026: an event peaking again after its turn is a failure).
		constexpr double noCreditRefiredSamples = 3.0;
		// The output's level is read only farther than this from every event, where no peak, pre-shape or trough of a
		// recall reaches.
		constexpr double baselineReach = 10.0;
		// The input's peak at a shown event, in the fixed input field.
		constexpr double inputPeakAmplitude = 7.6;
		constexpr double inputPeakWidth = 7.0;

		// Scalar timing: the error tolerated grows with the time since the trigger.
		constexpr double weberFraction = 0.2;
		// Far from its time the Gaussian is flat. This share of the timing credit is the delay over the due time (or the
		// due time over the delay, if later), which keeps rising all the way to it.
		constexpr double timingRatioShare = 0.25;

		// What sets one stage's task apart: its name, the configs of its two weighted sums (the intended one holds only
		// weights: it is no task of its own), whether it scores timing, the range its waits are drawn from, the wait
		// of the trial a saved solution replays, and whether its triggers come in turn in one run.
		struct StageTask
		{
			const char* name;
			const char* slug;
			const char* intendedSlug;
			bool timed;
			double shortestWait;
			double longestWait;
			double replayedWait;
			bool consecutive;
		};

		// Stages 1 and 2 recall after a short wait: their references' memories fade. From stage 3 they must hold through
		// the longest wait the probe covers.
		constexpr double shortWaitFrom = 300.0;
		constexpr double shortWaitTo = 500.0;
		constexpr double longWaitTo = 1500.0;
		// A consecutive stage recalls after six triggers in turn in one run (the user's choice, 5 November 2026: one
		// demonstration, then recall as many times as asked). Its waits are the shortest and the longest twice each, and two
		// middle ones, each drawn on the sampling grid from its own half of 700-1100.
		constexpr double middleWaitFrom = 700.0;
		constexpr double middleWaitSplit = 900.0;
		constexpr double middleWaitTo = 1100.0;
		// A consecutive evaluation's four waits, shortest first.
		constexpr size_t shortWait = 0;
		constexpr size_t earlierMiddleWait = 1;
		constexpr size_t laterMiddleWait = 2;
		constexpr size_t longWait = 3;
		constexpr size_t consecutiveTriggers = IncrementalRecall<RecallStage::singleOrder>::consecutiveTriggerCount;
		// The two sequences of six triggers: each wait once among the first three and once among the last three, and between
		// them every wait follows every other.
		constexpr std::array<std::array<size_t, consecutiveTriggers>, 2> consecutiveSequences{ {
			{ shortWait, earlierMiddleWait, longWait, shortWait, longWait, laterMiddleWait },
			{ longWait, shortWait, earlierMiddleWait, longWait, laterMiddleWait, shortWait } } };
		// A consecutive stage's saved solution replays the first sequence, each middle wait at the middle of its range: each
		// wait from the end of the recall window before it (the first from the end of the demonstration).
		constexpr std::array<double, consecutiveTriggers> consecutiveReplayedWaits{ 300.0, 800.0, 1500.0, 300.0, 1500.0,
			1000.0 };

		constexpr StageTask taskOf(const RecallStage stage)
		{
			switch (stage)
			{
			case RecallStage::consecutiveOrder:
				return { "Incremental Recall 5 - Consecutive Order", "incremental-recall-5-consecutive-order",
					"incremental-recall-5-consecutive-order-intended", false, shortWaitFrom, longWaitTo,
					consecutiveReplayedWaits.front(), true };
			case RecallStage::repeatableOrderAndTiming:
				return { "Incremental Recall 4 - Repeatable Order And Timing",
					"incremental-recall-4-repeatable-order-and-timing",
					"incremental-recall-4-repeatable-order-and-timing-intended", true, shortWaitFrom, longWaitTo, longWaitTo,
					false };
			case RecallStage::repeatableOrder:
				return { "Incremental Recall 3 - Repeatable Order", "incremental-recall-3-repeatable-order",
					"incremental-recall-3-repeatable-order-intended", false, shortWaitFrom, longWaitTo, longWaitTo, false };
			case RecallStage::singleOrderAndTiming:
				return { "Incremental Recall 2 - Single Order And Timing", "incremental-recall-2-single-order-and-timing",
					"incremental-recall-2-single-order-and-timing-intended", true, shortWaitFrom, shortWaitTo, shortWaitFrom,
					false };
			case RecallStage::singleOrder:
			default:
				return { "Incremental Recall 1 - Single Order", "incremental-recall-1-single-order",
					"incremental-recall-1-single-order-intended", false, shortWaitFrom, shortWaitTo, shortWaitFrom, false };
			}
		}

		// When a saved solution's triggers come after the demonstration's.
		std::vector<double> replayedCues(const StageTask& task)
		{
			if (!task.consecutive)
			{
				return { demonstrationEnd + task.replayedWait };
			}
			std::vector<double> cues;
			double waitFrom = demonstrationEnd;
			for (const double wait : consecutiveReplayedWaits)
			{
				cues.push_back(waitFrom + wait);
				waitFrom = cues.back() + recallDuration;
			}
			return cues;
		}

		// Eight guided terms and six intended ones, each with a timing term once a stage scores timing.
		constexpr size_t guidedTermCount(const StageTask& task) { return task.timed ? 9 : 8; }
		constexpr size_t intendedTermCount(const StageTask& task) { return task.timed ? 7 : 6; }

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

		// The positions of the events shown by @p time, in the order shown.
		std::vector<double> positionsShownBy(const double time)
		{
			std::vector<double> positions;
			for (const Event& event : sequence)
			{
				if (event.onset < time)
				{
					positions.push_back(event.position);
				}
			}
			return positions;
		}
	}

	template <RecallStage stage>
	IncrementalRecall<stage>::IncrementalRecall(const SolutionTopology& topology)
		: Solution(topology)
	{
		constexpr StageTask task = taskOf(stage);
		name = task.name;
		loadFitnessWeights(task.slug, guidedTermCount(task));
		intendedFitnessWeights = ConfigLoader::loadFitnessWeights(task.intendedSlug, intendedTermCount(task));
	}

	template <RecallStage stage>
	IncrementalRecall<stage>::IncrementalRecall(const SolutionTopology& initialTopology,
		const dnf_composer::Simulation& phenotype)
		: Solution(initialTopology, phenotype)
	{
		constexpr StageTask task = taskOf(stage);
		name = task.name;
		loadFitnessWeights(task.slug, guidedTermCount(task));
		intendedFitnessWeights = ConfigLoader::loadFitnessWeights(task.intendedSlug, intendedTermCount(task));
	}

	template <RecallStage stage>
	SolutionPtr IncrementalRecall<stage>::clone() const
	{
		IncrementalRecall solution(initialTopology);
		return std::make_shared<IncrementalRecall>(solution);
	}

	template <RecallStage stage>
	SolutionPtr IncrementalRecall<stage>::copy() const
	{
		IncrementalRecall solution(initialTopology, phenotype);
		return std::make_shared<IncrementalRecall>(solution);
	}

	template <RecallStage stage>
	bool IncrementalRecall<stage>::reachedTarget(const double targetFitness) const
	{
		// The fitness itself must pass too (the user, 5 November 2026), whatever the weights add up to.
		return parameters.fitness > targetFitness && !targetTerms.empty() && std::ranges::all_of(targetTerms,
			[targetFitness](const double term) { return term > targetFitness; });
	}

	template <RecallStage stage>
	void IncrementalRecall<stage>::prepareTargetConfirmation(const size_t confirmation)
	{
		this->confirmation = confirmation;
	}

	template <RecallStage stage>
	void IncrementalRecall<stage>::testPhenotype()
	{
		parameters.fitness = 0.0;
		parameters.partialFitness.clear();
		inputPeaks.clear();
		inputRest.clear();
		outputMemory.clear();
		targetTerms.clear();
		baselinePositions = farFromEveryEvent();

		createPhenotypeEnvironment();
		setCue(std::nullopt);
		initSimulation();
		Probe probe;
		std::vector<double> waits;
		if constexpr (taskOf(stage).consecutive)
		{
			waits = consecutiveWaits();
		}
		else
		{
			const Waits drawn = drawRecallWaits();
			waits.assign(drawn.begin(), drawn.end());
		}
		// Every recall branches off the run without the trigger, so the demonstration and the waits are stepped once. In
		// a consecutive stage the one run of triggers branches off at its first: the others follow it in turn.
		const std::span<const double> branches = taskOf(stage).consecutive
			? std::span<const double>(waits).first(1) : std::span<const double>(waits);
		const std::vector<CueMoment> moments = demonstrateAndHold(branches, probe);

		std::vector<double> eventsQuietWithoutCue;
		for (const EventRecall& event : probe.events)
		{
			eventsQuietWithoutCue.push_back(1.0 - event.bestScore);
		}
		const double probeQuiet = mean(probe.quiet) * mean(eventsQuietWithoutCue);
		std::vector<RecallTrial> recalls;
		if constexpr (taskOf(stage).consecutive)
		{
			recalls = recallInTurn(waits, moments.front(), probeQuiet);
		}
		else
		{
			std::ranges::transform(moments, std::back_inserter(recalls),
				[this, probeQuiet](const CueMoment& moment) { return recallFrom(moment, probeQuiet); });
		}

		const double input = 0.5 * mean(inputPeaks) + 0.5 * mean(inputRest); // detects, then rests
		const double memory = meanOrZero(outputMemory);                       // pre-shaped at the stored events, in order
		// The guided and intended terms, each recall term taken over the recalls by @p overValues.
		const auto weightedTerms = [&](const auto& overValues) {
			const auto over = [&](const double RecallTrial::* term) {
				std::vector<double> values;
				std::ranges::transform(recalls, std::back_inserter(values), [term](const RecallTrial& trial) { return trial.*term; });
				return overValues(values);
			};
			const double quiet = over(&RecallTrial::quiet);           // no peak before the trigger
			const double recalled = over(&RecallTrial::recalled);     // the events recalled in order
			const double oneAtATime = over(&RecallTrial::oneAtATime); // each peak alone on the output
			std::vector<double> guided{ input, quiet, memory, recalled, oneAtATime };
			std::vector<double> intended{ input, quiet, recalled, oneAtATime };
			if constexpr (taskOf(stage).timed)
			{
				const double timing = over(&RecallTrial::timing);     // each at its time after the trigger
				guided.push_back(timing);
				intended.push_back(timing);
			}
			guided.insert(guided.end(), { over(&RecallTrial::toCome), // the events still to recall, in order
				over(&RecallTrial::released),                         // each recalled event let go
				over(&RecallTrial::stopped) });                       // no peak once the sequence is over
			intended.insert(intended.end(), { over(&RecallTrial::releasedWithoutGuide),
				over(&RecallTrial::stoppedWithoutGuide) });
			return std::pair{ guided, intended };
		};

		std::vector<double> intended;
		std::tie(parameters.partialFitness, intended) = weightedTerms([](const std::vector<double>& values) { return mean(values); });
		const double guidedFitness = std::inner_product(parameters.partialFitness.begin(),
			parameters.partialFitness.end(), fitnessWeights.begin(), 0.0);
		const double intendedFitness = std::inner_product(intended.begin(), intended.end(),
			intendedFitnessWeights.begin(), 0.0);
		// The guides lead the search toward the reference's way of recalling; they must not rank a design that recalls
		// another way below one that does not, nor keep a run going once it does.
		parameters.fitness = std::max(guidedFitness, intendedFitness);
		targetTerms = intendedFitness > guidedFitness ? intended : parameters.partialFitness;
		// From stage 5 the target asks each term of every recall, so one recall that fails keeps a design short of it
		// however well the others go.
		if constexpr (taskOf(stage).consecutive)
		{
			const auto [worstGuided, worstIntended] = weightedTerms(
				[](const std::vector<double>& values) { return std::ranges::min(values); });
			targetTerms = intendedFitness > guidedFitness ? worstIntended : worstGuided;
		}

		nlohmann::json probeEvents = nlohmann::json::object();
		for (size_t event = 0; event < sequence.size(); ++event)
		{
			probeEvents[sequence[event].label] = { { "peak", rounded(probe.events[event].bestScore) } };
		}
		nlohmann::json trials = nlohmann::json::array();
		for (const RecallTrial& recall : recalls)
		{
			trials.push_back(recall.trace);
		}
		trials.push_back({ { "kind", "probe" }, { "watchedFrom", demonstrationEnd }, { "length", probeHold },
			{ "quietWithoutCue", rounded(mean(probe.quiet)) },
			{ "eventsQuietWithoutCue", rounded(mean(eventsQuietWithoutCue)) }, { "events", probeEvents } });
		// Unrounded, unlike the rest of the trace: the fitness is exactly the larger of the two.
		const nlohmann::json trace{ { "fitness", { { "guided", guidedFitness }, { "intended", intendedFitness } } },
			{ "trials", trials } };
		parameters.evaluationTrace = trace.dump();
	}

	template <RecallStage stage>
	void IncrementalRecall<stage>::createPhenotypeEnvironment()
	{
		using dnf_composer::element::TimedGaussStimulusParameters;
		const dnf_composer::element::ElementDimensions dimensions{ DimensionConstants::xSize, DimensionConstants::dx };
		const std::vector<double> cues = replayedCues(taskOf(stage));
		cue = cues.front();
		std::vector<std::pair<double, double>> onTimes{ { 0.0, pulseDuration } };
		for (const double replayedCue : cues)
		{
			onTimes.emplace_back(replayedCue, replayedCue + pulseDuration);
		}
		addTimedGaussianStimulus(cueStimulusName, inputField, TimedGaussStimulusParameters{ cueWidth,
			GaussStimulusConstants::amplitude, DimensionConstants::xSize / 2.0, onTimes, false, false }, dimensions);
		for (const Event& event : sequence)
		{
			addTimedGaussianStimulus(std::format("tgs {} {:.0f}", event.label, event.position), inputField,
				TimedGaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude,
				event.position, { { event.onset, event.onset + pulseDuration } }, GaussStimulusConstants::circularity,
				GaussStimulusConstants::normalization }, dimensions);
		}
	}

	template <RecallStage stage>
	auto IncrementalRecall<stage>::drawRecallWaits() const -> Waits
	{
		// One wait from each equal share of the range, so the mean over the recalls spans it every time.
		constexpr StageTask task = taskOf(stage);
		constexpr double share = (task.longestWait - task.shortestWait) / static_cast<double>(recallTrialCount);
		Waits waits{};
		for (size_t trial = 0; trial < recallTrialCount; ++trial)
		{
			const double from = task.shortestWait + share * static_cast<double>(trial);
			waits[trial] = std::round(tools::utils::generateRandomDouble(from, from + share) / sampleInterval)
				* sampleInterval;
		}
		return waits;
	}

	template <RecallStage stage>
	std::vector<double> IncrementalRecall<stage>::consecutiveWaits() const
	{
		constexpr StageTask task = taskOf(stage);
		const auto onGrid = [](const double from, const double to) {
			return std::round(tools::utils::generateRandomDouble(from, to) / sampleInterval) * sampleInterval;
		};
		const std::array<double, 4> waits{ task.shortestWait, onGrid(middleWaitFrom, middleWaitSplit),
			onGrid(middleWaitSplit, middleWaitTo), task.longestWait };
		// Either sequence serves, drawn afresh each evaluation; confirmations of a reached target take them in turn, so any
		// two together recall after both.
		const size_t chosen = confirmation ? *confirmation % consecutiveSequences.size()
			: static_cast<size_t>(tools::utils::generateRandomInt(0, static_cast<int>(consecutiveSequences.size()) - 1));
		std::vector<double> inTurn;
		std::ranges::transform(consecutiveSequences[chosen], std::back_inserter(inTurn),
			[&waits](const size_t wait) { return waits[wait]; });
		return inTurn;
	}

	template <RecallStage stage>
	void IncrementalRecall<stage>::setCue(const std::optional<double> cueTime)
	{
		const auto stimulus = std::dynamic_pointer_cast<dnf_composer::element::TimedGaussStimulus>(
			phenotype.getElement(cueStimulusName));
		auto stimulusParameters = stimulus->getParameters();
		stimulusParameters.onTimes = { { 0.0, pulseDuration } };
		if (cueTime)
		{
			stimulusParameters.onTimes.emplace_back(*cueTime, *cueTime + pulseDuration);
		}
		stimulus->setParameters(stimulusParameters);
		cue = cueTime;
	}

	template <RecallStage stage>
	void IncrementalRecall<stage>::triggerAt(const double cueTime)
	{
		// The state was saved as the clock reached the trigger's moment in the run without it. The step that reaches the
		// trigger switches it on after the fields have stepped, so it is switched on here, and the fields feel it from the
		// next step, as in a run that had it all along.
		phenotype.t = cueTime;
		setCue(cueTime);
		phenotype.getElement(cueStimulusName)->step(cueTime, phenotype.getDeltaT());
	}

	template <RecallStage stage>
	bool IncrementalRecall<stage>::inputSettledAt(const double time) const
	{
		const auto pulsedRecently = [time](const double onset) {
			return time >= onset && time < onset + pulseDuration + settleTime;
		};
		return !pulsedRecently(0.0) && !(cue && pulsedRecently(*cue))
			&& std::ranges::none_of(sequence, [&pulsedRecently](const Event& event) { return pulsedRecently(event.onset); });
	}

	template <RecallStage stage>
	void IncrementalRecall<stage>::runUntil(const double time)
	{
		while (phenotype.getT() < time - 1e-9)
		{
			runSimulation(1);
		}
	}

	template <RecallStage stage>
	void IncrementalRecall<stage>::sampleUntil(const double time,
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

	template <RecallStage stage>
	auto IncrementalRecall<stage>::demonstrateAndHold(const std::span<const double> waits, Probe& probe)
		-> std::vector<CueMoment>
	{
		std::vector<double> quietSamples;
		// A consecutive stage branches off at its first trigger only, yet its memory is read through the longest wait.
		const double lastCue = demonstrationEnd
			+ (taskOf(stage).consecutive ? taskOf(stage).longestWait : std::ranges::max(waits));
		const auto observe = [&](const double time, const std::optional<double> inputAtRest) {
			const double noPeak = noBumps(outputField);
			quietSamples.push_back(noPeak);
			// Memory is what the output holds below threshold once the input has let go of each event, until a trigger.
			const std::vector<double> stored = positionsShownBy(time);
			if (inputAtRest && !stored.empty() && time <= lastCue)
			{
				outputMemory.push_back(preShapedInOrder(stored, baseline()) * *inputAtRest * noPeak);
			}
			// From the end of the demonstration the run is the probe: whatever fires without the trigger counts against it.
			if (time > demonstrationEnd)
			{
				probe.quiet.push_back(noPeak);
				observeRecall(time, probe.events);
			}
		};
		for (const Event& event : sequence)
		{
			sampleUntil(event.onset + pulseDuration, observe);
			inputPeaks.push_back(oneBumpAtPositionWithAmplitudeAndWidth(inputField, event.position,
				inputPeakAmplitude, inputPeakWidth));
		}
		std::vector<CueMoment> moments;
		std::ranges::transform(waits, std::back_inserter(moments), [&](const double wait) {
			const double cueTime = demonstrationEnd + wait;
			sampleUntil(cueTime, observe);
			return CueMoment{ cueTime, mean(quietSamples), simulationState() };
		});
		sampleUntil(demonstrationEnd + probeHold, observe);
		return moments;
	}

	template <RecallStage stage>
	auto IncrementalRecall<stage>::recallFrom(const CueMoment& moment, const double probeQuiet) -> RecallTrial
	{
		restoreSimulationState(moment.state);
		return recallAt(moment.cue, moment.cue - demonstrationEnd, moment.quietBeforeCue, probeQuiet,
			moment.cue + silenceWatched);
	}

	template <RecallStage stage>
	auto IncrementalRecall<stage>::recallInTurn(const std::span<const double> waits, const CueMoment& first,
		const double probeQuiet) -> std::vector<RecallTrial>
	{
		restoreSimulationState(first.state);
		std::vector<RecallTrial> recalls;
		double cueTime = first.cue;
		double quietBeforeCue = first.quietBeforeCue;
		for (size_t trigger = 0; trigger < waits.size(); ++trigger)
		{
			// The silence after a recall is watched until the next trigger, and after the last as after any recall.
			const bool last = trigger + 1 == waits.size();
			const double nextCue = last ? cueTime + silenceWatched : cueTime + recallDuration + waits[trigger + 1];
			recalls.push_back(recallAt(cueTime, waits[trigger], quietBeforeCue, probeQuiet, nextCue));
			quietBeforeCue = recalls.back().quietPastWindow;
			cueTime = nextCue;
		}
		return recalls;
	}

	template <RecallStage stage>
	auto IncrementalRecall<stage>::recallAt(const double cueTime, const double wait, const double quietBeforeCue,
		const double probeQuiet, const double watchedUntil) -> RecallTrial
	{
		triggerAt(cueTime);
		Recall recall{};
		AfterRecall after{};
		std::vector<double> stillAtEvents;
		const std::vector<double> waitingPreShapes = recordRecall(recall, after, stillAtEvents, watchedUntil);

		// A design that peaks before the trigger, or fires without it, earns no credit for firing after it.
		const double recallScale = quietBeforeCue * probeQuiet;
		RecallTrial trial;
		trial.quiet = quietBeforeCue;
		trial.recalled = recallScale * meanCreditInPlace(recall, [](const EventRecall&) { return 1.0; });
		trial.oneAtATime = recallScale * meanCreditInPlace(recall,
			[](const EventRecall& event) { return meanOrZero(event.alone); });
		trial.timing = recallScale * timingOf(recall, cueTime);
		trial.toCome = recallScale * meanOrZero(waitingPreShapes);
		trial.released = recallScale * releaseAfterRecall(recall, true);
		trial.stopped = recallScale * stoppedAfterRecall(recall, after, true);
		trial.releasedWithoutGuide = recallScale * releaseAfterRecall(recall, false);
		trial.stoppedWithoutGuide = recallScale * stoppedAfterRecall(recall, after, false);
		// From stage 5 a bump that stands at no event, between the peaks or rolling past them, costs one at a time too,
		// each sample two peaks are up together costs a share of the recall's one-at-a-time credit, not only of the
		// peak's samples, and each sample an event peaks again after its release costs a share of its release credit.
		const size_t sharedSamples = distinctSamples(recall, &EventRecall::sharedAt);
		const size_t refiredSamples = distinctSamples(recall, &EventRecall::refiredAt);
		if constexpr (taskOf(stage).consecutive)
		{
			trial.oneAtATime *= meanOrZero(stillAtEvents)
				* (1.0 - tools::utils::normalize(static_cast<double>(sharedSamples), 0.0, noCreditSharedSamples));
			const double neverRefired = 1.0
				- tools::utils::normalize(static_cast<double>(refiredSamples), 0.0, noCreditRefiredSamples);
			trial.released *= neverRefired;
			trial.releasedWithoutGuide *= neverRefired;
		}
		trial.quietPastWindow = meanOrZero(after.quietPastWindow);
		const nlohmann::json terms{ { "quiet", rounded(trial.quiet) }, { "recalled", rounded(trial.recalled) },
			{ "oneAtATime", rounded(trial.oneAtATime) }, { "timing", rounded(trial.timing) },
			{ "toCome", rounded(trial.toCome) }, { "released", rounded(trial.released) },
			{ "stopped", rounded(trial.stopped) }, { "releasedWithoutGuide", rounded(trial.releasedWithoutGuide) },
			{ "stoppedWithoutGuide", rounded(trial.stoppedWithoutGuide) } };
		trial.trace = { { "kind", "recall" }, { "wait", wait }, { "cue", cueTime },
			{ "quietBeforeCue", rounded(quietBeforeCue) }, { "stillAtEvents", rounded(meanOrZero(stillAtEvents)) },
			{ "sharedSamples", sharedSamples }, { "refiredSamples", refiredSamples }, { "terms", terms },
			{ "recall", recallSummary(recall, after) } };
		return trial;
	}

	template <RecallStage stage>
	std::vector<double> IncrementalRecall<stage>::recordRecall(Recall& recall, AfterRecall& after,
		std::vector<double>& stillAtEvents, const double watchedUntil)
	{
		std::vector<double> waitingPreShapes;
		sampleUntil(*cue + recallDuration, [&](const double time, const std::optional<double> inputAtRest) {
			// While the trigger still lifts the input, it holds whatever it held before.
			const double restingInput = inputAtRest.value_or(inputRest.back());
			if (const std::optional<double> waiting = observeRecall(time, recall))
			{
				// The input only detects events: what it holds is not memory.
				waitingPreShapes.push_back(*waiting * restingInput);
			}
			if (!sequenceReleased(recall))
			{
				if (const std::optional<double> still = bumpsStillAtEvents(recall))
				{
					stillAtEvents.push_back(*still);
				}
			}
			if (sequenceReleased(recall) || time >= *cue + sequenceOverAfter)
			{
				observeAfterRecall(time, after);
			}
		});
		sampleUntil(watchedUntil, [&](const double time, std::optional<double>) {
			const double noPeak = noBumps(outputField);
			after.quiet.push_back(noPeak);
			after.quietPastWindow.push_back(noPeak);
			observeRefiring(time, recall);
		});
		return waitingPreShapes;
	}

	template <RecallStage stage>
	void IncrementalRecall<stage>::observeRefiring(const double time, Recall& recall) const
	{
		for (size_t event = 0; event < sequence.size(); ++event)
		{
			EventRecall& observed = recall[event];
			if (observed.releasedAt && peakAt(sequence[event].position) >= peakingScore)
			{
				observed.refiredAt.push_back(time);
			}
		}
	}

	template <RecallStage stage>
	double IncrementalRecall<stage>::baseline() const
	{
		return medianActivationAtPositions(outputField, baselinePositions);
	}

	template <RecallStage stage>
	double IncrementalRecall<stage>::preShapedInOrder(const std::vector<double>& positions, const double level) const
	{
		std::vector<double> preShapes;
		for (const double position : positions)
		{
			preShapes.push_back(preShapeAboveLevelAtPosition(outputField, position, level));
		}
		const double order = positions.size() > 1 ? activationsDescendAcrossPositions(outputField, positions) : 1.0;
		// Order only counts among real pre-shapes: noise alone puts a flat field's samples in some order.
		return mean(preShapes) * (0.5 + 0.5 * order);
	}

	template <RecallStage stage>
	double IncrementalRecall<stage>::peakAt(const double position) const
	{
		// Count a bump only near the event, in proportion to its height, and only as narrow as a peak: a field-wide
		// flash with a gap reads as one bump opposite the gap.
		return closenessOfNearestBumpToPosition(outputField, position)
			* heightCreditOfNearestBumpToPosition(outputField, position)
			* widthCreditOfNearestBumpToPosition(outputField, position, fullCreditPeakWidth, noCreditPeakWidth);
	}

	template <RecallStage stage>
	double IncrementalRecall<stage>::movedSince(const std::optional<double> lastCentroid,
		const std::optional<double> centroid)
	{
		if (!lastCentroid || !centroid)
		{
			return 0.0;
		}
		const double moved = aroundTheField(*lastCentroid, *centroid);
		return moved > sameBumpReach ? 0.0 : moved;
	}

	template <RecallStage stage>
	double IncrementalRecall<stage>::stayingInPlace(const std::array<double, driftSamples>& lastMoves)
	{
		// From stage 5 the drift is the bump's path over all the moves watched but its largest, as a peak may split for a
		// sample as it is released: a bump left rolling round the field after one recall moves about a unit a sample,
		// little from one sample to the next but three or more over three.
		const double drift = taskOf(stage).consecutive
			? std::accumulate(lastMoves.begin(), lastMoves.end(), 0.0) - std::ranges::max(lastMoves)
			: std::min(lastMoves[driftSamples - 2], lastMoves.back());
		return 1.0 - tools::utils::normalize(drift, stillPeakDrift, travellingPeakDrift);
	}

	template <RecallStage stage>
	std::optional<double> IncrementalRecall<stage>::bumpsStillAtEvents(const Recall& recall) const
	{
		const auto& bumps = getNeuralFieldOrThrow(outputField, "bumpsStillAtEvents")->getBumps();
		if (bumps.empty())
		{
			return std::nullopt;
		}
		double leastStill = 1.0;
		for (const dnf_composer::element::NeuralFieldBump& bump : bumps)
		{
			// Each event follows the bump nearest it, so a bump within an event's reach moved as that event's did.
			double still = 0.0;
			for (size_t event = 0; event < sequence.size(); ++event)
			{
				if (aroundTheField(bump.centroid, sequence[event].position) <= sameBumpReach)
				{
					still = std::max(still, stayingInPlace(recall[event].lastMoves));
				}
			}
			leastStill = std::min(leastStill, still);
		}
		return leastStill;
	}

	template <RecallStage stage>
	size_t IncrementalRecall<stage>::distinctSamples(const Recall& recall, const std::vector<double> EventRecall::* times)
	{
		std::set<double> samples;
		for (const EventRecall& observed : recall)
		{
			samples.insert((observed.*times).begin(), (observed.*times).end());
		}
		return samples.size();
	}

	template <RecallStage stage>
	std::optional<double> IncrementalRecall<stage>::observeRecall(const double time, Recall& recall) const
	{
		const auto& bumps = getNeuralFieldOrThrow(outputField, "observeRecall")->getBumps();
		const double level = baseline();
		std::vector<double> waiting;
		for (size_t event = 0; event < sequence.size(); ++event)
		{
			const double position = sequence[event].position;
			EventRecall& observed = recall[event];
			const std::optional<double> centroid = centroidOfNearestBumpToPosition(outputField, position);
			std::ranges::rotate(observed.lastMoves, observed.lastMoves.begin() + 1);
			observed.lastMoves.back() = movedSince(observed.lastCentroid, centroid);
			const double score = peakAt(position) * stayingInPlace(observed.lastMoves);
			observed.lastCentroid = centroid;
			// A peak earns its best samples running: a flash up for a single sample earns a share of a peak.
			std::ranges::rotate(observed.lastScores, observed.lastScores.begin() + 1);
			observed.lastScores.back() = score;
			observed.bestScore = std::max(observed.bestScore, mean(observed.lastScores));
			const bool peaking = score >= peakingScore;
			if (peaking && !observed.firstPeak)
			{
				observed.firstPeak = time;
			}
			// The order is the order the peaks come in, so they must come one at a time. A peak may split for a sample as it
			// is released: only a bump away from the event is another peak.
			if (peaking)
			{
				const bool alone = std::ranges::none_of(bumps, [position](const dnf_composer::element::NeuralFieldBump& bump) {
					return aroundTheField(bump.centroid, position) > sameBumpReach;
				});
				observed.alone.push_back(alone ? 1.0 : 0.0);
				if (!alone)
				{
					observed.sharedAt.push_back(time);
				}
				if (observed.releasedAt)
				{
					observed.refiredAt.push_back(time);
				}
			}
			if (observed.firstPeak && !peaking && !observed.releasedAt)
			{
				observed.releasedAt = time;
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
		if (waiting.empty())
		{
			return std::nullopt;
		}
		return preShapedInOrder(waiting, level);
	}

	template <RecallStage stage>
	void IncrementalRecall<stage>::observeAfterRecall(const double time, AfterRecall& after) const
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

	template <RecallStage stage>
	bool IncrementalRecall<stage>::sequenceReleased(const Recall& recall)
	{
		return std::ranges::all_of(recall, [](const EventRecall& observed) { return observed.releasedAt.has_value(); });
	}

	template <RecallStage stage>
	double IncrementalRecall<stage>::placeOf(const Recall& recall, const size_t event)
	{
		std::array<std::optional<double>, eventCount> firstPeaks;
		std::ranges::transform(recall, firstPeaks.begin(), &EventRecall::firstPeak);
		// From stage 5 an event is placed among all the others, so both events of a swapped pair lose their place.
		if constexpr (taskOf(stage).consecutive)
		{
			return tools::utils::shareInPlaceAmongAll(firstPeaks, event);
		}
		return tools::utils::shareInPlace(firstPeaks, event);
	}

	template <RecallStage stage>
	double IncrementalRecall<stage>::creditInPlace(const Recall& recall, const size_t event)
	{
		return recall[event].bestScore * placeOf(recall, event);
	}

	template <RecallStage stage>
	double IncrementalRecall<stage>::meanCreditInPlace(const Recall& recall,
		const std::function<double(const EventRecall&)>& perEvent)
	{
		double credit = 0.0;
		for (size_t event = 0; event < recall.size(); ++event)
		{
			credit += creditInPlace(recall, event) * perEvent(recall[event]);
		}
		return credit / static_cast<double>(recall.size());
	}

	template <RecallStage stage>
	double IncrementalRecall<stage>::onTime(const Recall& recall, const size_t event, const double cueTime)
	{
		const auto& firstPeak = recall[event].firstPeak;
		if (!firstPeak)
		{
			return 0.0;
		}
		const double due = sequence[event].onset;
		const double delay = *firstPeak - cueTime;
		const double nearDue = tools::utils::normalizeWithGaussian(delay, due, weberFraction * due);
		const double ratio = delay > 0.0 ? std::min(delay, due) / std::max(delay, due) : 0.0;
		return (1.0 - timingRatioShare) * nearDue + timingRatioShare * ratio;
	}

	template <RecallStage stage>
	double IncrementalRecall<stage>::timingOf(const Recall& recall, const double cueTime)
	{
		double credit = 0.0;
		for (size_t event = 0; event < recall.size(); ++event)
		{
			credit += creditInPlace(recall, event) * onTime(recall, event, cueTime);
		}
		return credit / static_cast<double>(recall.size());
	}

	template <RecallStage stage>
	double IncrementalRecall<stage>::releaseAfterRecall(const Recall& recall, const bool withGuides)
	{
		return meanCreditInPlace(recall, [withGuides](const EventRecall& event) {
			if (event.belowThreshold.empty())
			{
				return 0.0;
			}
			return withGuides ? 0.5 * mean(event.belowThreshold) + 0.5 * mean(event.troughs) : mean(event.belowThreshold);
		});
	}

	template <RecallStage stage>
	double IncrementalRecall<stage>::stoppedAfterRecall(const Recall& recall, const AfterRecall& after,
		const bool withGuides)
	{
		// An output that never peaked has nothing to stop: its silence counts only as far as the recall was in order.
		const double noPeak = meanOrZero(after.quiet) * meanCreditInPlace(recall, [](const EventRecall&) { return 1.0; });
		if (!withGuides)
		{
			return noPeak;
		}
		double troughs = 0.0;
		for (size_t event = 0; event < recall.size(); ++event)
		{
			troughs += creditInPlace(recall, event) * meanOrZero(after.troughs[event]);
		}
		return 0.5 * noPeak + 0.5 * troughs / static_cast<double>(recall.size());
	}

	template <RecallStage stage>
	nlohmann::json IncrementalRecall<stage>::recallSummary(const Recall& recall, const AfterRecall& after) const
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
				{ "afterCue", observed.firstPeak ? nlohmann::json(rounded(*observed.firstPeak - *cue)) : nlohmann::json(nullptr) },
				{ "releasedAt", timeOrNull(observed.releasedAt) },
				{ "inPlace", rounded(placeOf(recall, event)) },
				{ "bestScore", rounded(observed.bestScore) },
				{ "alone", observed.alone.empty() ? nlohmann::json(nullptr) : nlohmann::json(rounded(mean(observed.alone))) },
				{ "onTime", rounded(onTime(recall, event, *cue)) },
				{ "release", observed.belowThreshold.empty() ? nlohmann::json(nullptr)
					: nlohmann::json(rounded(mean(observed.belowThreshold))) }
			};
		}
		const nlohmann::json afterSequence = { { "from", timeOrNull(after.from) },
			{ "quiet", rounded(meanOrZero(after.quiet)) } };
		return { { "events", events }, { "afterSequence", afterSequence } };
	}

	template <RecallStage stage>
	double IncrementalRecall<stage>::rounded(const double value)
	{
		return std::round(value * 100.0) / 100.0;
	}

	template class IncrementalRecall<RecallStage::singleOrder>;
	template class IncrementalRecall<RecallStage::singleOrderAndTiming>;
	template class IncrementalRecall<RecallStage::repeatableOrder>;
	template class IncrementalRecall<RecallStage::repeatableOrderAndTiming>;
	template class IncrementalRecall<RecallStage::consecutiveOrder>;
}
