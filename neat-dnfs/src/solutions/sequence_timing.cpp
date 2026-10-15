#include "solutions/sequence_timing.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace neat_dnfs
{
	namespace
	{
		const std::string inputField = "nf 1";
		const std::string outputField = "nf 2";

		// Every duration below is simulation time, converted to steps at the configured deltaT.
		struct TaskEvent
		{
			const char* label;
			double position;
			double onset; // after the start event X
		};

		constexpr TaskEvent startEvent{ "X", 50.0, 0.0 };
		// Serial order D, C, A, B: intervals 100, 500, 200 and 1000.
		constexpr std::array<TaskEvent, SequenceTiming::recalledEventCount> recalledEvents{ {
			{ "D", 180.0, 100.0 },
			{ "C", 120.0, 600.0 },
			{ "A", 20.0, 800.0 },
			{ "B", 80.0, 1800.0 }
		} };
		constexpr std::array<TaskEvent, SequenceTiming::recalledEventCount + 1> demonstratedEvents{ {
			startEvent, recalledEvents[0], recalledEvents[1], recalledEvents[2], recalledEvents[3]
		} };

		constexpr double pulseDuration = 50.0;
		// Three gaps, one drawn from each third of this range on every evaluation, so that a fixed delay after the
		// demonstration can match the recall timing in one trial at most and no fixed set of gaps can be learned.
		constexpr double shortestRecallGap = 300.0;
		constexpr double longestRecallGap = 1500.0;
		constexpr double recallDuration = recalledEvents.back().onset + 500.0;

		// Scalar timing: the tolerated error grows with the time elapsed since the cue.
		constexpr double weberFraction = 0.2;
		// Consecutive onsets at least this far apart earn full order credit; the closest demonstrated pair is 200 apart.
		constexpr double minimumOnsetSeparation = 100.0;
		constexpr double orderCreditShare = 0.5;
		// Events are at least 30 apart, so a bump within 10 belongs to one event only.
		constexpr double onsetPositionTolerance = 10.0;
		constexpr double isolationSettleTime = 50.0;
		// A bump this close to one on the previous step is the same peak, not a new one.
		constexpr double samePeakDistance = 5.0;
		constexpr size_t maxTracedRisingPeaks = 16;

		constexpr double inputBumpAmplitude = 5.0;
		constexpr double inputBumpWidth = 5.0;

		bool hasBumpAt(const std::vector<dnf_composer::element::NeuralFieldBump>& bumps, const double position)
		{
			return std::ranges::any_of(bumps, [position](const auto& bump) {
				return std::abs(bump.centroid - position) <= onsetPositionTolerance;
			});
		}
	}

	SequenceTiming::SequenceTiming(const SolutionTopology& topology)
		: SequenceTiming(topology, StartCue::gaussianPulse, TimingCredit::exactOnsets, "Sequence Timing", "sequence-timing")
	{
	}

	SequenceTiming::SequenceTiming(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype)
		: SequenceTiming(initialTopology, phenotype, StartCue::gaussianPulse, TimingCredit::exactOnsets, "Sequence Timing",
			"sequence-timing")
	{
	}

	SequenceTiming::SequenceTiming(const SolutionTopology& topology, const StartCue startCue, const TimingCredit timingCredit,
		const std::string& taskName, const std::string& slug)
		: Solution(topology), startCue(startCue), timingCredit(timingCredit)
	{
		name = taskName;
		loadFitnessWeights(slug, 6);
	}

	SequenceTiming::SequenceTiming(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype,
		const StartCue startCue, const TimingCredit timingCredit, const std::string& taskName, const std::string& slug)
		: Solution(initialTopology, phenotype), startCue(startCue), timingCredit(timingCredit)
	{
		name = taskName;
		loadFitnessWeights(slug, 6);
	}

	SolutionPtr SequenceTiming::clone() const
	{
		SequenceTiming solution(initialTopology);
		return std::make_shared<SequenceTiming>(solution);
	}

	SolutionPtr SequenceTiming::copy() const
	{
		SequenceTiming solution(initialTopology, phenotype);
		return std::make_shared<SequenceTiming>(solution);
	}

	void SequenceTiming::testPhenotype()
	{
		parameters.fitness = 0.0;
		parameters.partialFitness.clear();

		TaskScores scores;
		nlohmann::json trials = nlohmann::json::array();
		const auto recallGaps = drawRecallGaps();
		for (const double gap : recallGaps)
		{
			trials.push_back(runRecallTrial(gap, scores));
		}
		trials.push_back(runProbeTrial(recallGaps.front(), scores));

		parameters.partialFitness = {
			scores.inputRepresentation / scores.pulses,
			scores.inputRest / scores.restSamples,
			scores.outputSilence / scores.silentSteps,
			scores.recalledEvents / scores.recallTrials,
			scores.recalledTiming / scores.recallTrials,
			scores.oneAtATime / scores.recallTrials
		};

		parameters.fitness = std::inner_product(parameters.partialFitness.begin(), parameters.partialFitness.end(),
			fitnessWeights.begin(), 0.0);
		parameters.evaluationTrace = nlohmann::json{ { "trials", trials } }.dump();
	}

	nlohmann::json SequenceTiming::runRecallTrial(const double gap, TaskScores& scores)
	{
		const TaskScores trialStart = scores;
		demonstrateSequence(scores);
		const TaskScores gapStart = scores;
		runWhileOutputSilent(gap, scores);
		const double gapSilence = meanOutputSilence(gapStart, scores);
		// Recall credit only counts as far as the output was silent until the cue.
		const double recallCreditGate = meanOutputSilence(trialStart, scores);
		sampleInputRest(scores);

		const RecallObservation recall = recallSequence(scores);
		scores.recalledEvents += recallCreditGate * recalledEventsScore(recall);
		scores.recalledTiming += recallCreditGate * timingTermScore(recall);
		scores.oneAtATime += recallCreditGate * oneAtATimeScore(recall);
		++scores.recallTrials;

		return {
			{ "kind", "recall" },
			{ "gap", gap },
			{ "outputSilence", { { "demonstration", meanOutputSilence(trialStart, gapStart) }, { "gap", gapSilence } } },
			{ "recallCreditGate", recallCreditGate },
			{ "recall", traceOf(recall) }
		};
	}

	nlohmann::json SequenceTiming::runProbeTrial(const double gap, TaskScores& scores)
	{
		const TaskScores trialStart = scores;
		demonstrateSequence(scores);
		const TaskScores waitStart = scores;
		runWhileOutputSilent(gap + recallDuration, scores);
		sampleInputRest(scores);

		return {
			{ "kind", "probe" },
			{ "outputSilence", { { "demonstration", meanOutputSilence(trialStart, waitStart) },
				{ "withoutCue", meanOutputSilence(waitStart, scores) } } }
		};
	}

	void SequenceTiming::demonstrateSequence(TaskScores& scores)
	{
		initSimulation();
		for (size_t i = 0; i < demonstratedEvents.size(); ++i)
		{
			if (i == 0)
			{
				pulseStartCue(scores);
			}
			else
			{
				pulseEvent(demonstratedEvents[i].position, scores);
			}
			const bool isLastEvent = i + 1 == demonstratedEvents.size();
			if (!isLastEvent)
			{
				runWhileOutputSilent(demonstratedEvents[i + 1].onset - demonstratedEvents[i].onset - pulseDuration, scores);
				sampleInputRest(scores);
			}
		}
	}

	SequenceTiming::RecallObservation SequenceTiming::recallSequence(TaskScores& scores)
	{
		const auto output = getNeuralFieldOrThrow(outputField, "recallSequence");
		const int pulseSteps = stepsFor(pulseDuration);
		RecallObservation observation;
		std::vector<dnf_composer::element::NeuralFieldBump> previousBumps;

		startStartCue();
		for (int step = 1; step <= stepsFor(recallDuration); ++step)
		{
			runSimulation(1);
			if (step == pulseSteps)
			{
				endStartCue(scores);
			}
			const auto bumps = output->getBumps();
			detectOnsets(step, bumps, observation);
			scoreIsolation(step, bumps, observation);
			recordRisingPeaks(step, previousBumps, bumps, observation);
			previousBumps = bumps;
		}
		sampleInputRest(scores);

		// The last peak has no successor to push it out, so it must die out on its own.
		// Only a peak that actually rose can earn that credit.
		if (observation.onsetSteps.back())
		{
			observation.lastPeakDecay = noBumps(outputField);
		}
		return observation;
	}

	void SequenceTiming::pulseStartCue(TaskScores& scores)
	{
		startStartCue();
		runWhileOutputSilent(pulseDuration, scores);
		endStartCue(scores);
	}

	void SequenceTiming::startStartCue()
	{
		if (startCue == StartCue::boost)
		{
			addBoostStimulus(inputField, GaussStimulusConstants::amplitude,
				dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
			return;
		}
		startPulse(startEvent.position);
	}

	void SequenceTiming::endStartCue(TaskScores& scores)
	{
		if (startCue == StartCue::boost)
		{
			removeBoostStimuli();
			return;
		}
		endPulse(startEvent.position, scores);
	}

	void SequenceTiming::pulseEvent(const double position, TaskScores& scores)
	{
		startPulse(position);
		runWhileOutputSilent(pulseDuration, scores);
		endPulse(position, scores);
	}

	void SequenceTiming::startPulse(const double position)
	{
		addGaussianStimulus(inputField,
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude, position,
				GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
	}

	void SequenceTiming::endPulse(const double position, TaskScores& scores)
	{
		scores.inputRepresentation += oneBumpAtPositionWithAmplitudeAndWidth(inputField, position, inputBumpAmplitude, inputBumpWidth);
		++scores.pulses;
		removeGaussianStimuli();
	}

	void SequenceTiming::runWhileOutputSilent(const double duration, TaskScores& scores)
	{
		for (int step = 0; step < stepsFor(duration); ++step)
		{
			runSimulation(1);
			scores.outputSilence += noBumps(outputField);
			++scores.silentSteps;
		}
	}

	void SequenceTiming::sampleInputRest(TaskScores& scores) const
	{
		scores.inputRest += closenessOfMeanActivationToRestingLevel(inputField);
		++scores.restSamples;
	}

	void SequenceTiming::detectOnsets(const int step, const std::vector<dnf_composer::element::NeuralFieldBump>& bumps,
		RecallObservation& observation)
	{
		for (size_t i = 0; i < recalledEventCount; ++i)
		{
			if (observation.onsetSteps[i])
			{
				continue;
			}
			if (!hasBumpAt(bumps, recalledEvents[i].position))
			{
				observation.absentSinceCue[i] = true;
			}
			else if (observation.absentSinceCue[i])
			{
				observation.onsetSteps[i] = step;
			}
		}
	}

	void SequenceTiming::scoreIsolation(const int step, const std::vector<dnf_composer::element::NeuralFieldBump>& bumps,
		RecallObservation& observation)
	{
		for (size_t i = 0; i < recalledEventCount; ++i)
		{
			const auto& onset = observation.onsetSteps[i];
			if (onset && *onset + stepsFor(isolationSettleTime) == step)
			{
				const double bumpCountScore = 1.0 / (1.0 + std::abs(static_cast<double>(bumps.size()) - 1.0));
				observation.isolation[i] = hasBumpAt(bumps, recalledEvents[i].position) ? bumpCountScore : 0.0;
			}
		}
	}

	void SequenceTiming::recordRisingPeaks(const int step, const std::vector<dnf_composer::element::NeuralFieldBump>& previousBumps,
		const std::vector<dnf_composer::element::NeuralFieldBump>& bumps, RecallObservation& observation)
	{
		for (const auto& bump : bumps)
		{
			const bool wasAlreadyUp = std::ranges::any_of(previousBumps, [&bump](const auto& previous) {
				return std::abs(previous.centroid - bump.centroid) <= samePeakDistance;
			});
			if (wasAlreadyUp)
			{
				continue;
			}
			++observation.risingPeakCount;
			if (observation.risingPeaks.size() < maxTracedRisingPeaks)
			{
				observation.risingPeaks.emplace_back(step * SimulationConstants::deltaT, bump.centroid);
			}
		}
	}

	nlohmann::json SequenceTiming::traceOf(const RecallObservation& observation)
	{
		nlohmann::json onsets;
		nlohmann::json isolation;
		for (size_t i = 0; i < recalledEventCount; ++i)
		{
			const auto& onsetStep = observation.onsetSteps[i];
			onsets[recalledEvents[i].label] = onsetStep ? nlohmann::json(*onsetStep * SimulationConstants::deltaT) : nlohmann::json(nullptr);
			isolation[recalledEvents[i].label] = observation.isolation[i];
		}
		return {
			{ "onsets", onsets },
			{ "isolation", isolation },
			{ "order", orderScore(observation) },
			{ "lastPeakDecay", observation.lastPeakDecay },
			{ "risingPeakCount", observation.risingPeakCount },
			{ "risingPeaks", observation.risingPeaks }
		};
	}

	double SequenceTiming::meanOutputSilence(const TaskScores& from, const TaskScores& to)
	{
		const int steps = to.silentSteps - from.silentSteps;
		return steps > 0 ? (to.outputSilence - from.outputSilence) / steps : 1.0;
	}

	double SequenceTiming::recalledEventsScore(const RecallObservation& observation)
	{
		const auto recalled = std::ranges::count_if(observation.onsetSteps, [](const auto& onset) { return onset.has_value(); });
		return static_cast<double>(recalled) / recalledEventCount;
	}

	double SequenceTiming::recalledTimingScore(const RecallObservation& observation)
	{
		double total = 0.0;
		for (size_t i = 0; i < recalledEventCount; ++i)
		{
			const auto& onsetStep = observation.onsetSteps[i];
			if (onsetStep)
			{
				const double expectedOnset = recalledEvents[i].onset;
				total += tools::utils::normalizeWithGaussian(*onsetStep * SimulationConstants::deltaT, expectedOnset, weberFraction * expectedOnset);
			}
		}
		return total / recalledEventCount;
	}

	double SequenceTiming::orderScore(const RecallObservation& observation)
	{
		double total = 0.0;
		for (size_t i = 0; i + 1 < recalledEventCount; ++i)
		{
			const auto& earlier = observation.onsetSteps[i];
			const auto& later = observation.onsetSteps[i + 1];
			if (earlier && later)
			{
				const double separation = (*later - *earlier) * SimulationConstants::deltaT;
				total += std::clamp(separation / minimumOnsetSeparation, 0.0, 1.0);
			}
		}
		return total / (recalledEventCount - 1);
	}

	double SequenceTiming::timingTermScore(const RecallObservation& observation) const
	{
		switch (timingCredit)
		{
		case TimingCredit::orderAndOnsets:
			return orderCreditShare * orderScore(observation) + (1.0 - orderCreditShare) * recalledTimingScore(observation);
		case TimingCredit::orderOnly:
			return orderScore(observation);
		case TimingCredit::exactOnsets:
			break;
		}
		return recalledTimingScore(observation);
	}

	double SequenceTiming::oneAtATimeScore(const RecallObservation& observation)
	{
		const double isolation = std::accumulate(observation.isolation.begin(), observation.isolation.end(), 0.0);
		return (isolation + observation.lastPeakDecay) / (recalledEventCount + 1);
	}

	std::array<double, 3> SequenceTiming::drawRecallGaps()
	{
		std::array<double, 3> gaps{};
		const double third = (longestRecallGap - shortestRecallGap) / static_cast<double>(gaps.size());
		for (size_t i = 0; i < gaps.size(); ++i)
		{
			const double from = shortestRecallGap + third * static_cast<double>(i);
			gaps[i] = tools::utils::generateRandomDouble(from, from + third);
		}
		return gaps;
	}

	int SequenceTiming::stepsFor(const double duration)
	{
		return static_cast<int>(std::lround(duration / SimulationConstants::deltaT));
	}

	void SequenceTiming::createPhenotypeEnvironment()
	{
		// The start cue at full strength, as recall begins; the other events at zero
		// amplitude, so a saved solution carries every stimulus ready to switch on.
		// A boost cue is not saved: it is switched on and off around each delivery.
		if (startCue == StartCue::gaussianPulse)
		{
			startPulse(startEvent.position);
		}
		for (const auto& event : recalledEvents)
		{
			addGaussianStimulus(inputField,
				dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, 0.0, event.position,
					GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
				dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		}
	}

	SequenceTimingBoost::SequenceTimingBoost(const SolutionTopology& topology)
		: SequenceTiming(topology, StartCue::boost, TimingCredit::exactOnsets, "Sequence Timing Boost", "sequence-timing-boost")
	{
	}

	SequenceTimingBoost::SequenceTimingBoost(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype)
		: SequenceTiming(initialTopology, phenotype, StartCue::boost, TimingCredit::exactOnsets, "Sequence Timing Boost",
			"sequence-timing-boost")
	{
	}

	SolutionPtr SequenceTimingBoost::clone() const
	{
		SequenceTimingBoost solution(initialTopology);
		return std::make_shared<SequenceTimingBoost>(solution);
	}

	SolutionPtr SequenceTimingBoost::copy() const
	{
		SequenceTimingBoost solution(initialTopology, phenotype);
		return std::make_shared<SequenceTimingBoost>(solution);
	}

	SequenceTimingBoostOrder::SequenceTimingBoostOrder(const SolutionTopology& topology)
		: SequenceTiming(topology, StartCue::boost, TimingCredit::orderAndOnsets, "Sequence Timing Boost Order",
			"sequence-timing-boost-order")
	{
	}

	SequenceTimingBoostOrder::SequenceTimingBoostOrder(const SolutionTopology& initialTopology,
		const dnf_composer::Simulation& phenotype)
		: SequenceTiming(initialTopology, phenotype, StartCue::boost, TimingCredit::orderAndOnsets,
			"Sequence Timing Boost Order", "sequence-timing-boost-order")
	{
	}

	SolutionPtr SequenceTimingBoostOrder::clone() const
	{
		SequenceTimingBoostOrder solution(initialTopology);
		return std::make_shared<SequenceTimingBoostOrder>(solution);
	}

	SolutionPtr SequenceTimingBoostOrder::copy() const
	{
		SequenceTimingBoostOrder solution(initialTopology, phenotype);
		return std::make_shared<SequenceTimingBoostOrder>(solution);
	}

	SequenceTimingBoostOrderOnly::SequenceTimingBoostOrderOnly(const SolutionTopology& topology)
		: SequenceTiming(topology, StartCue::boost, TimingCredit::orderOnly, "Sequence Timing Boost Order Only",
			"sequence-timing-boost-order-only")
	{
	}

	SequenceTimingBoostOrderOnly::SequenceTimingBoostOrderOnly(const SolutionTopology& initialTopology,
		const dnf_composer::Simulation& phenotype)
		: SequenceTiming(initialTopology, phenotype, StartCue::boost, TimingCredit::orderOnly,
			"Sequence Timing Boost Order Only", "sequence-timing-boost-order-only")
	{
	}

	SolutionPtr SequenceTimingBoostOrderOnly::clone() const
	{
		SequenceTimingBoostOrderOnly solution(initialTopology);
		return std::make_shared<SequenceTimingBoostOrderOnly>(solution);
	}

	SolutionPtr SequenceTimingBoostOrderOnly::copy() const
	{
		SequenceTimingBoostOrderOnly solution(initialTopology, phenotype);
		return std::make_shared<SequenceTimingBoostOrderOnly>(solution);
	}
}
