#include "solutions/serial_order.h"

#include <numeric>

namespace neat_dnfs
{
	namespace
	{
		// X and the four events 200 apart: the order is all the demonstration carries, and it stays short.
		constexpr SequenceTiming::EventOnsets evenlySpacedOnsets{ 200.0, 400.0, 600.0, 800.0 };
		constexpr double shortestGap = 300.0;
		constexpr double longestGap = 1500.0;
		// Long enough for four peaks one after another; varied so no fixed period between cues can be learned.
		constexpr double shortestCycle = 800.0;
		constexpr double longestCycle = 1500.0;
	}

	SerialOrder::SerialOrder(const SolutionTopology& topology)
		: SequenceTiming(topology, StartCue::boost, TimingCredit::orderOnly, "Serial Order", "serial-order")
	{
	}

	SerialOrder::SerialOrder(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype)
		: SequenceTiming(initialTopology, phenotype, StartCue::boost, TimingCredit::orderOnly, "Serial Order", "serial-order")
	{
	}

	SolutionPtr SerialOrder::clone() const
	{
		SerialOrder solution(initialTopology);
		return std::make_shared<SerialOrder>(solution);
	}

	SolutionPtr SerialOrder::copy() const
	{
		SerialOrder solution(initialTopology, phenotype);
		return std::make_shared<SerialOrder>(solution);
	}

	void SerialOrder::testPhenotype()
	{
		parameters.fitness = 0.0;
		parameters.partialFitness.clear();

		TaskScores scores;
		const double gap = drawGap();
		const auto cycleLengths = drawCycleLengths();
		nlohmann::json trials = nlohmann::json::array();
		trials.push_back(runCuedTrial(gap, cycleLengths, scores));
		trials.push_back(runUncuedTrial(gap + std::accumulate(cycleLengths.begin(), cycleLengths.end(), 0.0), scores));

		recordEvaluation(scores, trials);
	}

	nlohmann::json SerialOrder::runCuedTrial(const double gap, const std::array<double, recallCycleCount>& cycleLengths,
		TaskScores& scores)
	{
		const TaskScores trialStart = scores;
		const nlohmann::json demonstratedOnsets = demonstrateSequence(scores, evenlySpacedOnsets);
		const TaskScores gapStart = scores;
		runWhileOutputSilent(gap, scores);
		const double gapSilence = meanOutputSilence(gapStart, scores);
		// Recall credit only counts as far as the output was silent until the first cue.
		const double recallCreditGate = meanOutputSilence(trialStart, scores);
		sampleInputRest(scores);

		nlohmann::json cycles = nlohmann::json::array();
		for (const double length : cycleLengths)
		{
			const RecallObservation recall = recallSequence(scores, length);
			// Activity away from the events would let output sweeping across the field earn credit by chance.
			const double cycleCreditGate = recallCreditGate * activityAtEvents(recall);
			scores.recalledEvents += cycleCreditGate * recalledEventsScore(recall);
			scores.recalledTiming += cycleCreditGate * orderScore(recall);
			scores.oneAtATime += cycleCreditGate * oneAtATimeFromRiseScore(recall);
			++scores.recallTrials;
			cycles.push_back({ { "length", length }, { "recall", traceOf(recall) } });
		}

		return {
			{ "kind", "recall" },
			{ "demonstratedOnsets", demonstratedOnsets },
			{ "gap", gap },
			{ "outputSilence", { { "demonstration", meanOutputSilence(trialStart, gapStart) }, { "gap", gapSilence } } },
			{ "recallCreditGate", recallCreditGate },
			{ "cycles", cycles }
		};
	}

	nlohmann::json SerialOrder::runUncuedTrial(const double waitingTime, TaskScores& scores)
	{
		const TaskScores trialStart = scores;
		const nlohmann::json demonstratedOnsets = demonstrateSequence(scores, evenlySpacedOnsets);
		const TaskScores waitStart = scores;
		runWhileOutputSilent(waitingTime, scores);
		scores.probeSilence = meanOutputSilence(waitStart, scores);
		sampleInputRest(scores);

		return {
			{ "kind", "probe" },
			{ "demonstratedOnsets", demonstratedOnsets },
			{ "outputSilence", { { "demonstration", meanOutputSilence(trialStart, waitStart) },
				{ "withoutCue", scores.probeSilence } } }
		};
	}

	double SerialOrder::drawGap()
	{
		return tools::utils::generateRandomDouble(shortestGap, longestGap);
	}

	std::array<double, SerialOrder::recallCycleCount> SerialOrder::drawCycleLengths()
	{
		std::array<double, recallCycleCount> lengths{};
		for (double& length : lengths)
		{
			length = tools::utils::generateRandomDouble(shortestCycle, longestCycle);
		}
		return lengths;
	}
}
