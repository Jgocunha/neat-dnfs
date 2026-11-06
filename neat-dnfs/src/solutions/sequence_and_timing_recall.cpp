#include "solutions/sequence_and_timing_recall.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <iterator>
#include <limits>
#include <numeric>
#include <ranges>
#include <set>
#include <span>

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

		// The demonstrated sequence: recall must replay it in this order, each event at its onset after the trigger.
		constexpr std::array<Event, SequenceAndTimingRecall::eventCount> sequence{ { { "LA2B", 40.0, 50.0 },
			{ "RA3A", 160.0, 150.0 }, { "LA1A", 60.0, 350.0 }, { "RA1B", 100.0, 450.0 }, { "RA2B", 120.0, 650.0 },
			{ "LA3A", 80.0, 750.0 }, { "RA1A", 140.0, 950.0 }, { "LA1B", 20.0, 1050.0 }, { "HOL", 200.0, 1900.0 },
			{ "HOS", 220.0, 2800.0 } } };

		const std::string inputField{ "nf 1" };
		const std::string outputField{ "nf 2" };
		// The memory recall trigger: a pulse wide enough to lift the whole input field evenly, as a boost would, as a
		// stimulus a saved file can replay.
		const std::string cueStimulusName{ "tgs memory recall trigger" };
		constexpr double cueWidth = 1e6;
		// How long each pulse, event or trigger, stays on.
		constexpr double pulseDuration = 50.0;
		// How long after a pulse ends the input's peak may take to decay before it must be settled.
		constexpr double settleTime = 50.0;
		constexpr double sampleInterval = 10.0;

		// When the demonstration's last pulse, HOS's, ends.
		constexpr double demonstrationEnd = sequence.back().onset + pulseDuration;
		// Each wait before a trigger of an evaluation's run is drawn afresh from this range, on the sampling grid (the
		// user, 6 November 2026: a design must recall after short and long waits alike).
		constexpr int shortestRecallWait = 500;
		constexpr int longestRecallWait = 1500;
		// How long the output is watched without a trigger: after the probe's demonstration, and after the last recall
		// window (the user, 6 November 2026).
		constexpr double unpromptedWatch = 1500.0;
		// Recall is recorded until 500 after HOS is due.
		constexpr double recallDuration = sequence.back().onset + 500.0;
		// Scalar timing: the error tolerated grows with the time since the trigger.
		constexpr double weberFraction = 0.2;
		// If some event never ends its first peak, the sequence counts as over once HOS is past its time by that error.
		constexpr double sequenceOverAfter = sequence.back().onset * (1.0 + weberFraction);
		// Far from its time the Gaussian is flat. This share of the timing credit is the delay over the due time (or the
		// due time over the delay, if later), which keeps rising all the way to it.
		constexpr double timingRatioShare = 0.25;
		// The waits a saved solution replays: the judge's first schedule, short and long waits in turn.
		constexpr std::array<double, SequenceAndTimingRecall::recallTriggerCount> replayedWaits{ 500.0, 1500.0, 1000.0,
			500.0, 1500.0 };

		constexpr auto taskSlug = "sequence-and-timing-recall";
		// Holds only the intended behaviour's weights: it is no task of its own.
		constexpr auto intendedSlug = "sequence-and-timing-recall-intended";
		constexpr size_t guidedTermCount = 8;
		constexpr size_t intendedTermCount = 5;

		// An event counts as peaking from this peak score: one bump near it, about a quarter of full height or more.
		constexpr double peakingScore = 0.25;
		// A recalled peak stays where it is: its centre may move this much from one sample to the next for full credit,
		constexpr double stillPeakDrift = 1.0;
		// and earns nothing from this much on, where a bump travelling around the field would credit each event it passes.
		constexpr double travellingPeakDrift = 3.0;
		// The nearest bump at two successive samples is the same bump only this close: the events lie 20 apart, so a
		// peak at the next event is a new bump, not a moved one.
		constexpr double sameBumpReach = 10.0;
		// The output's level is read only farther than this from every event, where no peak, pre-shape or trough of a
		// recall reaches.
		constexpr double baselineReach = 10.0;
		// A bump lies at an event this close to it, as dnf-composer's checks read a first peak,
		constexpr double eventReach = 8.0;
		// counts from this height,
		constexpr double bumpHeight = 0.5;
		// and is no wider than a peak up to this width, where a peak's width credit is still full.
		constexpr double peakWidth = 30.0;
		// At most this many events may be up at once, or first peak at one sample (the user's tie rule of 5 November
		// 2026: a pair, never a burst).
		constexpr size_t mostEventsTogether = 2;
		// A level counts as below the one before it from this much lower, above the noise between neighbouring places
		// (in the reference, a standard deviation of 0.013, at most 0.07),
		constexpr double levelNoise = 0.03;
		// and fully from this much further.
		constexpr double levelStep = 0.05;
		// The memory a recall starts from is read over this long before its trigger.
		constexpr double memoryReading = 100.0;
		// A released event's trough is read for at least this long.
		constexpr double shortestTroughReading = 100.0;
		// The input must be above threshold during the trigger: read from this long after it starts to its end.
		constexpr double triggerRise = 10.0;
		// From this long after a trigger starts, the input must be quiet until the next.
		constexpr double triggerSettled = pulseDuration + settleTime;

		double meanOrZero(const std::vector<double>& values)
		{
			return values.empty() ? 0.0 : mean(values);
		}

		double worstOrZero(const std::vector<double>& values)
		{
			return values.empty() ? 0.0 : std::ranges::min(values);
		}

		// The mean of the samples timed from @p from (inclusive) to @p to (exclusive), if there are any.
		std::optional<double> meanBetween(const std::vector<std::pair<double, double>>& samples, const double from,
			const double to)
		{
			std::vector<double> values;
			for (const auto& [time, value] : samples)
			{
				if (time >= from - 1e-9 && time < to - 1e-9)
				{
					values.push_back(value);
				}
			}
			return values.empty() ? std::nullopt : std::optional<double>(mean(values));
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

		// The first @p count events, in order.
		std::vector<size_t> firstEvents(const size_t count)
		{
			std::vector<size_t> events(count);
			std::iota(events.begin(), events.end(), size_t{ 0 });
			return events;
		}

		size_t eventsShownBy(const double time)
		{
			return static_cast<size_t>(std::ranges::count_if(sequence, [time](const Event& event) { return event.onset < time; }));
		}

		// The event a bump centred at @p centroid lies at, if any.
		std::optional<size_t> eventAt(const double centroid)
		{
			for (size_t event = 0; event < sequence.size(); ++event)
			{
				if (aroundTheField(centroid, sequence[event].position) <= eventReach)
				{
					return event;
				}
			}
			return std::nullopt;
		}

		// When each trigger of a run comes: the first its wait after the demonstration, each later one its wait after the
		// recall window before it.
		std::vector<double> cuesAfter(const std::span<const double> waits)
		{
			std::vector<double> cues;
			double windowEnd = demonstrationEnd;
			for (const double wait : waits)
			{
				cues.push_back(windowEnd + wait);
				windowEnd = cues.back() + recallDuration;
			}
			return cues;
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

	bool SequenceAndTimingRecall::reachedTarget(const double targetFitness) const
	{
		// The fitness itself must pass too (the user, 5 November 2026), whatever the weights add up to.
		return parameters.fitness > targetFitness && !targetTerms.empty() && std::ranges::all_of(targetTerms,
			[targetFitness](const double term) { return term > targetFitness; });
	}

	void SequenceAndTimingRecall::testPhenotype()
	{
		parameters.fitness = 0.0;
		parameters.partialFitness.clear();
		targetTerms.clear();
		baselinePositions = farFromEveryEvent();
		createPhenotypeEnvironment();

		nlohmann::json trials = nlohmann::json::array();
		const double probeQuiet = runProbe(trials);

		const std::vector<double> waits = recallWaits();
		const std::vector<double> cues = cuesAfter(waits);
		setCues(cues);
		nlohmann::json snapshots = nlohmann::json::array();
		const DemonstrationScores demonstration = demonstrate(cues.front(), snapshots);
		const double tailEnd = cues.back() + recallDuration + unpromptedWatch;
		std::vector<RecallTerms> perRecall;
		nlohmann::json recalls = nlohmann::json::array();
		double memoryBeforeCue = meanOrZero(demonstration.memoryAtEnd);
		double tailQuiet = 0.0;
		for (size_t k = 0; k < cues.size(); ++k)
		{
			snapshots.push_back(snapshot(std::format("before trigger {}", k + 1)));
			const double following = k + 1 < cues.size() ? cues[k + 1] : tailEnd;
			const RecallSamples samples = sampleRecall(cues[k], following, snapshots);
			const RecallTerms terms = scoreRecall(samples, cues[k]);
			perRecall.push_back(terms);
			recalls.push_back({ { "wait", waits[k] }, { "cue", cues[k] }, { "memoryBeforeCue", rounded(memoryBeforeCue) },
				{ "clean", rounded(terms.clean) }, { "refiredSamples", refiredSamplesOf(samples.events) },
				{ "terms", { { "input", rounded(terms.input) }, { "order", rounded(terms.order) },
					{ "timing", rounded(terms.timing) }, { "toCome", rounded(terms.toCome) },
					{ "passed", rounded(terms.passed) }, { "afterIntended", rounded(terms.afterIntended) },
					{ "afterGuided", rounded(terms.afterGuided) } } },
				{ "recall", recallSummary(samples, cues[k]) } });
			memoryBeforeCue = meanOrZero(samples.memoryBeforeNext);
			if (k + 1 == cues.size())
			{
				tailQuiet = meanBetween(samples.quiet, cues[k] + recallDuration, tailEnd).value_or(0.0);
			}
		}
		trials.push_back({ { "kind", "consecutive" }, { "recalls", recalls },
			{ "tail", { { "watched", unpromptedWatch }, { "quiet", rounded(tailQuiet) } } }, { "snapshots", snapshots } });

		// Whatever fires without the trigger, before it or once the run is over, earns no credit for firing after it.
		const double demonstrationQuiet = mean(demonstration.quiet);
		const double gate = mean(std::vector<double>{ demonstrationQuiet, probeQuiet, tailQuiet });
		const double input = 0.5 * mean(demonstration.inputPulses) + 0.5 * meanOrZero(demonstration.inputBetweenPulses);
		const double memory = meanOrZero(demonstration.memory);
		// The guided and intended terms, each recall term taken over the recalls by @p overRecalls.
		const auto weightedTerms = [&](const auto& overRecalls) {
			const auto over = [&](double RecallTerms::* term, const double scale) {
				std::vector<double> values;
				for (const RecallTerms& terms : perRecall)
				{
					values.push_back(terms.*term);
				}
				return scale * overRecalls(values);
			};
			const double recallInput = over(&RecallTerms::input, 1.0);
			const double order = over(&RecallTerms::order, gate);
			const double timing = over(&RecallTerms::timing, gate);
			const std::vector<double> guided{ input, memory, recallInput, order, timing, over(&RecallTerms::toCome, gate),
				over(&RecallTerms::passed, gate), over(&RecallTerms::afterGuided, gate) };
			const std::vector<double> intended{ input, recallInput, order, timing, over(&RecallTerms::afterIntended, gate) };
			return std::pair{ guided, intended };
		};

		std::vector<double> intended;
		std::tie(parameters.partialFitness, intended) = weightedTerms(meanOrZero);
		const double guidedFitness = std::inner_product(parameters.partialFitness.begin(), parameters.partialFitness.end(),
			fitnessWeights.begin(), 0.0);
		const double intendedFitness = std::inner_product(intended.begin(), intended.end(),
			intendedFitnessWeights.begin(), 0.0);
		// The guides lead the search toward the reference's way of recalling; they must not rank a design that recalls
		// on time another way below one that does not, nor keep a run going once it does.
		parameters.fitness = std::max(guidedFitness, intendedFitness);
		// The target asks each term of every recall, so one recall that fails keeps a design short of it however well the
		// others go.
		const auto [worstGuided, worstIntended] = weightedTerms(worstOrZero);
		targetTerms = intendedFitness > guidedFitness ? worstIntended : worstGuided;
		// Unrounded, unlike the rest of the trace: the fitness is exactly the larger of the two.
		const nlohmann::json trace{ { "fitness", { { "guided", guidedFitness }, { "intended", intendedFitness } } },
			{ "gate", { { "gate", gate }, { "demonstration", rounded(demonstrationQuiet) }, { "probe", rounded(probeQuiet) },
				{ "tail", rounded(tailQuiet) } } },
			{ "trials", trials } };
		parameters.evaluationTrace = trace.dump();
	}

	void SequenceAndTimingRecall::createPhenotypeEnvironment()
	{
		using dnf_composer::element::TimedGaussStimulusParameters;
		const dnf_composer::element::ElementDimensions dimensions{ DimensionConstants::xSize, DimensionConstants::dx };
		const std::vector<double> cues = cuesAfter(replayedWaits);
		TimedGaussStimulusParameters trigger{ cueWidth, GaussStimulusConstants::amplitude, DimensionConstants::xSize / 2.0,
			{ { 0.0, pulseDuration } }, false, false };
		for (const double cue : cues)
		{
			trigger.onTimes.emplace_back(cue, cue + pulseDuration);
		}
		addTimedGaussianStimulus(cueStimulusName, inputField, trigger, dimensions);
		for (const Event& event : sequence)
		{
			addTimedGaussianStimulus(std::format("tgs {}", event.label), inputField, TimedGaussStimulusParameters{
				GaussStimulusConstants::width, GaussStimulusConstants::amplitude, event.position,
				{ { event.onset, event.onset + pulseDuration } }, GaussStimulusConstants::circularity,
				GaussStimulusConstants::normalization }, dimensions);
		}
		trialCues = cues;
	}

	std::vector<double> SequenceAndTimingRecall::recallWaits() const
	{
		// On the sampling grid, so that each trigger falls on a sample.
		const auto steps = [](const int wait) { return static_cast<int>(wait / static_cast<int>(sampleInterval)); };
		std::vector<double> waits;
		for (size_t trigger = 0; trigger < recallTriggerCount; ++trigger)
		{
			waits.push_back(sampleInterval * tools::utils::generateRandomInt(steps(shortestRecallWait), steps(longestRecallWait)));
		}
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

	void SequenceAndTimingRecall::sampleUntil(const double time, const std::function<void(double)>& onSample)
	{
		// Samples fall on whole multiples of the interval, so every stage's samples line up with the schedule.
		for (auto sample = static_cast<long>(std::floor(phenotype.getT() / sampleInterval + 1e-9)) + 1;
			static_cast<double>(sample) * sampleInterval <= time + 1e-9; ++sample)
		{
			const double sampleTime = static_cast<double>(sample) * sampleInterval;
			runUntil(sampleTime);
			onSample(sampleTime);
		}
		runUntil(time);
	}

	SequenceAndTimingRecall::DemonstrationScores SequenceAndTimingRecall::demonstrate(const double until,
		nlohmann::json& snapshots)
	{
		initSimulation();
		DemonstrationScores scores;
		const auto inputBumps = [this] { return getNeuralFieldOrThrow(inputField, "demonstrate")->getBumps(); };
		const std::vector<size_t> allEvents = firstEvents(sequence.size());
		const auto observe = [&](const double time) {
			const double noPeak = noBumps(outputField);
			scores.quiet.push_back(noPeak);
			// Memory is what the output holds below threshold once the input has let go of each event.
			if (inputSettledAt(time))
			{
				scores.inputBetweenPulses.push_back(inputBumps().empty() ? 1.0 : 0.0);
				if (const size_t shown = eventsShownBy(time); shown >= 2)
				{
					scores.memory.push_back(levelsFallInOrder(firstEvents(shown)) * noPeak);
				}
			}
			if (time > until - memoryReading + 1e-9)
			{
				scores.memoryAtEnd.push_back(levelsFallInOrder(allEvents) * noPeak);
			}
		};

		sampleUntil(sequence.front().onset, observe);
		snapshots.push_back(snapshot(std::format("before {}", sequence.front().label)));
		for (const Event& event : sequence)
		{
			sampleUntil(event.onset + pulseDuration, observe);
			snapshots.push_back(snapshot(std::format("{} shown", event.label)));
			const bool elsewhere = std::ranges::any_of(inputBumps(), [&event](const auto& bump) {
				return aroundTheField(bump.centroid, event.position) > eventReach;
			});
			scores.inputPulses.push_back(0.5 * bumpAt(inputField, event.position) + 0.5 * !elsewhere);
		}
		sampleUntil(until, observe);
		return scores;
	}

	double SequenceAndTimingRecall::runProbe(nlohmann::json& trials)
	{
		setCues({});
		nlohmann::json snapshots = nlohmann::json::array();
		demonstrate(demonstrationEnd, snapshots);
		std::vector<double> quiet;
		sampleUntil(demonstrationEnd + unpromptedWatch, [&](double) { quiet.push_back(noBumps(outputField)); });
		snapshots.push_back(snapshot("end"));
		const double quietWithoutCue = mean(quiet);
		trials.push_back({ { "kind", "probe" }, { "watched", unpromptedWatch },
			{ "quietWithoutCue", rounded(quietWithoutCue) }, { "snapshots", snapshots } });
		return quietWithoutCue;
	}

	SequenceAndTimingRecall::RecallSamples SequenceAndTimingRecall::sampleRecall(const double cue, const double following,
		nlohmann::json& snapshots)
	{
		RecallSamples samples;
		const double windowEnd = cue + recallDuration;
		const std::vector<size_t> allEvents = firstEvents(sequence.size());
		size_t snapshotsTaken = 0;
		sampleUntil(following, [&](const double time) {
			// The sample at the next trigger belongs to the next recall.
			if (time >= following - 1e-9)
			{
				return;
			}
			const bool inWindow = time < windowEnd - 1e-9;
			observeRecall(time, samples.events, inWindow);
			const double noPeak = noBumps(outputField);
			samples.quiet.emplace_back(time, noPeak);
			if (inWindow)
			{
				const bool refired = std::ranges::any_of(samples.events, [time](const EventRecall& observed) {
					return !observed.refiredAt.empty() && std::abs(observed.refiredAt.back() - time) < 1e-9;
				});
				samples.clean.emplace_back(time, bumpsClean() && !refired);
				std::vector<size_t> waiting;
				std::ranges::copy_if(allEvents, std::back_inserter(waiting),
					[&samples](const size_t event) { return !samples.events[event].firstPeak; });
				if (waiting.size() >= 2)
				{
					samples.toCome.emplace_back(time, levelsFallInOrder(waiting));
				}
			}
			if (time > following - memoryReading + 1e-9)
			{
				samples.memoryBeforeNext.push_back(levelsFallInOrder(allEvents) * noPeak);
			}
			const bool inputUp = !getNeuralFieldOrThrow(inputField, "sampleRecall")->getBumps().empty();
			if (time >= cue + triggerRise - 1e-9 && time <= cue + pulseDuration + 1e-9 && inputUp)
			{
				samples.inputTriggered = true;
			}
			if (time >= cue + triggerSettled - 1e-9)
			{
				samples.inputQuiet.push_back(inputUp ? 0.0 : 1.0);
			}
			if (snapshotsTaken < sequence.size() && std::abs(time - cue - sequence[snapshotsTaken].onset) < 1e-9)
			{
				snapshots.push_back(snapshot(std::format("recall {}", sequence[snapshotsTaken].label)));
				++snapshotsTaken;
			}
			if (std::abs(time - windowEnd) < 1e-9)
			{
				snapshots.push_back(snapshot("recall end"));
			}
		});
		return samples;
	}

	SequenceAndTimingRecall::RecallTerms SequenceAndTimingRecall::scoreRecall(const RecallSamples& samples, const double cue)
	{
		const Recall& recall = samples.events;
		const double sequenceEnd = sequenceEndOf(recall, cue);
		RecallTerms terms;
		std::vector<double> clean;
		for (const auto& [time, sampleClean] : samples.clean)
		{
			if (time < sequenceEnd - 1e-9)
			{
				clean.push_back(sampleClean ? 1.0 : 0.0);
			}
		}
		terms.clean = clean.empty() ? 1.0 : mean(clean);
		double credit = 0.0;
		double timing = 0.0;
		double passed = 0.0;
		for (size_t event = 0; event < recall.size(); ++event)
		{
			const double eventCredit = creditOf(recall, event);
			credit += eventCredit;
			timing += eventCredit * onTime(recall, event, cue);
			passed += eventCredit * passedOf(recall[event], sequenceEnd);
		}
		const auto events = static_cast<double>(recall.size());
		terms.order = terms.clean * credit / events;
		terms.timing = timing / events;
		terms.passed = passed / events;
		terms.toCome = terms.order * meanBetween(samples.toCome, cue, sequenceEnd).value_or(0.0);
		// An output that never peaked has nothing to stop: its silence counts only as far as the recall was in order.
		const double quietAfter = meanBetween(samples.quiet, sequenceEnd, std::numeric_limits<double>::infinity()).value_or(1.0);
		terms.afterIntended = terms.order * quietAfter;
		terms.afterGuided = terms.order * (0.5 * quietAfter + 0.5 * meanOrZero(samples.memoryBeforeNext));
		terms.input = 0.5 * (samples.inputTriggered ? 1.0 : 0.0) + 0.5 * meanOrZero(samples.inputQuiet);
		return terms;
	}

	double SequenceAndTimingRecall::baseline() const
	{
		return medianActivationAtPositions(outputField, baselinePositions);
	}

	double SequenceAndTimingRecall::levelsFallInOrder(const std::vector<size_t>& events) const
	{
		std::vector<double> levels;
		for (const size_t event : events)
		{
			levels.push_back(activationAtPosition(outputField, sequence[event].position));
		}
		return tools::utils::descendingShare(levels, levelNoise, levelStep);
	}

	bool SequenceAndTimingRecall::bumpAt(const std::string& fieldName, const double position) const
	{
		return std::ranges::any_of(getNeuralFieldOrThrow(fieldName, "bumpAt")->getBumps(), [position](const auto& bump) {
			return aroundTheField(bump.centroid, position) <= eventReach && bump.amplitude >= bumpHeight;
		});
	}

	bool SequenceAndTimingRecall::bumpsClean() const
	{
		std::set<size_t> eventsUp;
		for (const auto& bump : getNeuralFieldOrThrow(outputField, "bumpsClean")->getBumps())
		{
			const std::optional<size_t> event = eventAt(bump.centroid);
			if (!event || bump.width > peakWidth)
			{
				return false;
			}
			eventsUp.insert(*event);
		}
		return eventsUp.size() <= mostEventsTogether;
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

	void SequenceAndTimingRecall::observeRecall(const double time, Recall& recall, const bool inWindow) const
	{
		const double level = baseline();
		for (size_t event = 0; event < sequence.size(); ++event)
		{
			const double position = sequence[event].position;
			EventRecall& observed = recall[event];
			if (inWindow)
			{
				const std::optional<double> centroid = centroidOfNearestBumpToPosition(outputField, position);
				const double move = movedSince(observed.lastCentroid, centroid);
				// Overlap is allowed: each event's score is its own, however many peaks are up with it.
				const double score = peakAt(position) * stayingInPlace(observed.lastMove, move);
				observed.lastCentroid = centroid;
				observed.lastMove = move;
				// A peak earns its best samples running: a flash up for a single sample earns a share of a peak.
				std::ranges::rotate(observed.lastScores, observed.lastScores.begin() + 1);
				observed.lastScores.back() = score;
				observed.bestScore = std::max(observed.bestScore, mean(observed.lastScores));
				const bool peaking = score >= peakingScore;
				if (peaking && !observed.firstPeak)
				{
					observed.firstPeak = time;
				}
				if (peaking && observed.releasedAt)
				{
					observed.refiredAt.push_back(time);
				}
				if (observed.firstPeak && !peaking && !observed.releasedAt)
				{
					observed.releasedAt = time;
				}
			}
			else if (observed.releasedAt && peakAt(position) >= peakingScore)
			{
				observed.refiredAt.push_back(time);
			}
			if (observed.releasedAt)
			{
				observed.troughs.emplace_back(time, troughBelowLevelAtPosition(outputField, position, level));
			}
		}
	}

	double SequenceAndTimingRecall::sequenceEndOf(const Recall& recall, const double cue)
	{
		if (!std::ranges::all_of(recall, [](const EventRecall& observed) { return observed.releasedAt.has_value(); }))
		{
			return cue + sequenceOverAfter;
		}
		return std::ranges::max(recall | std::views::transform([](const EventRecall& observed) { return *observed.releasedAt; }));
	}

	bool SequenceAndTimingRecall::inPositionOf(const Recall& recall, const size_t event)
	{
		std::array<std::optional<double>, eventCount> firstPeaks;
		std::ranges::transform(recall, firstPeaks.begin(), &EventRecall::firstPeak);
		return tools::utils::cameInPosition(firstPeaks, event, mostEventsTogether);
	}

	double SequenceAndTimingRecall::creditOf(const Recall& recall, const size_t event)
	{
		return inPositionOf(recall, event) ? recall[event].bestScore : 0.0;
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
		const double nearDue = tools::utils::normalizeWithGaussian(delay, due, weberFraction * due);
		const double ratio = delay > 0.0 ? std::min(delay, due) / std::max(delay, due) : 0.0;
		return (1.0 - timingRatioShare) * nearDue + timingRatioShare * ratio;
	}

	double SequenceAndTimingRecall::passedOf(const EventRecall& observed, const double sequenceEnd)
	{
		if (!observed.releasedAt)
		{
			return 0.0;
		}
		const double until = std::max(sequenceEnd, *observed.releasedAt + shortestTroughReading);
		return meanBetween(observed.troughs, *observed.releasedAt, until + sampleInterval).value_or(0.0);
	}

	size_t SequenceAndTimingRecall::refiredSamplesOf(const Recall& recall)
	{
		std::set<double> samples;
		for (const EventRecall& observed : recall)
		{
			samples.insert(observed.refiredAt.begin(), observed.refiredAt.end());
		}
		return samples.size();
	}

	nlohmann::json SequenceAndTimingRecall::recallSummary(const RecallSamples& samples, const double cue)
	{
		const Recall& recall = samples.events;
		const double sequenceEnd = sequenceEndOf(recall, cue);
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
				{ "inPosition", inPositionOf(recall, event) },
				{ "bestScore", rounded(observed.bestScore) },
				{ "onTime", rounded(onTime(recall, event, cue)) },
				{ "passed", observed.releasedAt ? nlohmann::json(rounded(passedOf(observed, sequenceEnd))) : nlohmann::json(nullptr) }
			};
		}
		const double quietAfter = meanBetween(samples.quiet, sequenceEnd, std::numeric_limits<double>::infinity()).value_or(1.0);
		const nlohmann::json afterSequence = { { "from", rounded(sequenceEnd) }, { "quiet", rounded(quietAfter) } };
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
