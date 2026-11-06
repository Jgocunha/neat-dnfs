#include "solutions/recall_behaviour.h"

#include <algorithm>
#include <cmath>

#include "neat_tools/utils.h"

namespace neat_dnfs
{
	namespace
	{
		using serial_order_schedule::sequence;
		using serial_order_schedule::sampleInterval;
		using tools::utils::mean;
		// An event counts as peaking from this recall score: one bump near it, about a quarter of full height or more.
		constexpr double peakingScore = 0.25;
		// A recalled peak stays where it is: its centre may move this much from one sample to the next for full credit,
		constexpr double stillPeakDrift = 1.0;
		// and earns nothing from this much on, where a bump travelling around the field would credit each event it passes.
		constexpr double travellingPeakDrift = 3.0;
		// The nearest bump at two successive samples is the same bump only this close: the events lie at least 40 apart,
		// so a peak handed over to the next event is a new bump, not a moved one.
		constexpr double sameBumpReach = 10.0;

		// The reference design's peaks; amplitude and width carry little of a peak's credit.
		constexpr double inputPeakAmplitude = 7.5;
		constexpr double inputPeakWidth = 6.0;
		constexpr double outputPeakAmplitude = 9.0;
		constexpr double outputPeakWidth = 7.0;
	}

	RecallBehaviour::RecallBehaviour(const SolutionTopology& topology, const Guides guides)
		: Solution(topology), guides(guides)
	{
	}

	RecallBehaviour::RecallBehaviour(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype,
		const Guides guides)
		: Solution(initialTopology, phenotype), guides(guides)
	{
	}

	void RecallBehaviour::runUntil(const double time)
	{
		while (phenotype.getT() < time - 1e-9)
		{
			runSimulation(1);
		}
	}

	void RecallBehaviour::sampleUntil(const double time, std::vector<double>& inputRest,
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

	double RecallBehaviour::inputPeakedAt(const double position) const
	{
		return oneBumpAtPositionWithAmplitudeAndWidth(inputField, position, inputPeakAmplitude, inputPeakWidth);
	}

	double RecallBehaviour::preShapedInOrder(const std::vector<double>& positions) const
	{
		std::vector<double> preShapes;
		for (const double position : positions)
		{
			preShapes.push_back(preShapeAboveFieldAtPosition(outputField, position));
		}
		const double order = positions.size() > 1 ? activationsDescendAcrossPositions(outputField, positions) : 1.0;
		// Order only counts among real pre-shapes: noise alone puts a flat field's samples in some order.
		return mean(preShapes) * (0.5 + 0.5 * order);
	}

	double RecallBehaviour::recalledAt(const double position) const
	{
		// A bump score pays a lone peak anywhere, and one barely above threshold, almost in full: count it only near the
		// event, and in proportion to its height. A field-wide flash with a gap reads as one bump opposite the gap: count
		// only a bump as narrow as a peak.
		return closenessOfNearestBumpToPosition(outputField, position) * heightCreditOfNearestBumpToPosition(outputField, position)
			* widthCreditOfNearestBumpToPosition(outputField, position);
	}

	double RecallBehaviour::movedSince(const std::optional<double> lastCentroid, const std::optional<double> centroid)
	{
		if (!lastCentroid || !centroid)
		{
			return 0.0;
		}
		const double apart = std::abs(*centroid - *lastCentroid);
		const double moved = std::min(apart, static_cast<double>(DimensionConstants::xSize) - apart);
		return moved > sameBumpReach ? 0.0 : moved;
	}

	double RecallBehaviour::stayingInPlace(const double lastMove, const double move)
	{
		return 1.0 - tools::utils::normalize(std::min(lastMove, move), stillPeakDrift, travellingPeakDrift);
	}

	void RecallBehaviour::observeRecall(const double time, CycleRecall& recall,
		std::vector<double>& waitingPreShapes, const double inputAtRest) const
	{
		// Recall is one peak at a time: a sample's credit is shared among the peaks up together.
		const auto peaksUp = static_cast<double>(getNeuralFieldOrThrow(outputField, "observeRecall")->getBumps().size());
		std::vector<double> waiting;
		for (size_t event = 0; event < sequence.size(); ++event)
		{
			const double position = sequence[event].position;
			EventRecall& observed = recall[event];
			const std::optional<double> centroid = centroidOfNearestBumpToPosition(outputField, position);
			const double move = movedSince(observed.lastCentroid, centroid);
			const double recalled = recalledAt(position) * stayingInPlace(observed.lastMove, move);
			observed.lastCentroid = centroid;
			observed.lastMove = move;
			const double score = recalled * oneBumpAtPositionWithAmplitudeAndWidth(outputField, position,
				outputPeakAmplitude, outputPeakWidth);
			const bool peaking = score >= peakingScore;
			observed.bestScore = std::max(observed.bestScore, score / std::max(peaksUp, 1.0));
			observed.recalled = std::max(observed.recalled, recalled);
			if (peaking && !observed.firstPeak)
			{
				observed.firstPeak = time;
			}
			observed.released = observed.released || (observed.firstPeak && !peaking);
			if (observed.released)
			{
				observed.belowThreshold.push_back(belowThresholdAtPosition(outputField, position));
				observed.belowField.push_back(troughBelowFieldAtPosition(outputField, position));
			}
			if (!observed.firstPeak)
			{
				waiting.push_back(position);
			}
		}
		if (!waiting.empty())
		{
			// The input only detects events: what it holds is not memory.
			waitingPreShapes.push_back(preShapedInOrder(waiting) * inputAtRest);
		}
	}

	bool RecallBehaviour::sequenceReleased(const CycleRecall& recall)
	{
		return std::ranges::all_of(recall, &EventRecall::released);
	}

	void RecallBehaviour::observeAfterRecall(const double time, AfterRecall& after) const
	{
		if (!after.from)
		{
			after.from = time;
		}
		after.quiet.push_back(noBumps(outputField));
		if (guides == Guides::scored)
		{
			for (size_t event = 0; event < sequence.size(); ++event)
			{
				after.troughs[event].push_back(troughBelowFieldAtPosition(outputField, sequence[event].position));
			}
		}
	}

	double RecallBehaviour::placeOf(const CycleRecall& recall, const size_t event)
	{
		std::array<std::optional<double>, eventCount> firstPeaks;
		std::ranges::transform(recall, firstPeaks.begin(), &EventRecall::firstPeak);
		return tools::utils::shareInPlace(firstPeaks, event);
	}

	double RecallBehaviour::creditInPlace(const CycleRecall& recall, const size_t event)
	{
		return recall[event].bestScore * placeOf(recall, event);
	}

	double RecallBehaviour::recallInOrder(const CycleRecall& recall)
	{
		double credit = 0.0;
		for (size_t event = 0; event < recall.size(); ++event)
		{
			credit += creditInPlace(recall, event);
		}
		return credit / static_cast<double>(recall.size());
	}

	double RecallBehaviour::releaseScore(const EventRecall& observed) const
	{
		return releaseScore(observed, guides);
	}

	double RecallBehaviour::releaseScore(const EventRecall& observed, const Guides scoring)
	{
		if (scoring == Guides::unscored)
		{
			return mean(observed.belowThreshold);
		}
		return 0.5 * mean(observed.belowThreshold) + 0.5 * mean(observed.belowField);
	}

	double RecallBehaviour::releaseAfterRecall(const CycleRecall& recall) const
	{
		return releaseAfterRecall(recall, guides);
	}

	double RecallBehaviour::releaseAfterRecall(const CycleRecall& recall, const Guides scoring)
	{
		double credit = 0.0;
		for (size_t event = 0; event < recall.size(); ++event)
		{
			if (!recall[event].belowThreshold.empty())
			{
				credit += creditInPlace(recall, event) * releaseScore(recall[event], scoring);
			}
		}
		return credit / static_cast<double>(recall.size());
	}

	double RecallBehaviour::stoppedAfterRecall(const CycleRecall& recall, const AfterRecall& after) const
	{
		return stoppedAfterRecall(recall, after, guides);
	}

	double RecallBehaviour::stoppedAfterRecall(const CycleRecall& recall, const AfterRecall& after, const Guides scoring)
	{
		// An output that never peaked has nothing to stop: its silence counts only as far as the recall was in order.
		const double noPeak = mean(after.quiet) * recallInOrder(recall);
		if (scoring == Guides::unscored)
		{
			return noPeak;
		}
		return 0.5 * noPeak + 0.5 * troughsAfterRecall(recall, after);
	}

	double RecallBehaviour::troughsAfterRecall(const CycleRecall& recall, const AfterRecall& after)
	{
		std::vector<double> troughs;
		for (size_t event = 0; event < recall.size(); ++event)
		{
			troughs.push_back(creditInPlace(recall, event) * mean(after.troughs[event]));
		}
		return mean(troughs);
	}

	nlohmann::json RecallBehaviour::recallSummary(const size_t cycle, const CycleRecall& recall,
		const AfterRecall& after) const
	{
		nlohmann::json events = nlohmann::json::object();
		for (size_t event = 0; event < recall.size(); ++event)
		{
			const auto& observed = recall[event];
			events[sequence[event].label] = {
				{ "firstPeak", observed.firstPeak ? nlohmann::json(rounded(*observed.firstPeak)) : nlohmann::json(nullptr) },
				{ "inPlace", rounded(placeOf(recall, event)) },
				{ "bestScore", rounded(observed.bestScore) },
				{ "recalled", rounded(observed.recalled) },
				{ "release", observed.belowThreshold.empty() ? nlohmann::json(nullptr)
					: nlohmann::json(rounded(releaseScore(observed))) }
			};
		}
		const nlohmann::json afterSequence = {
			{ "from", after.from ? nlohmann::json(rounded(*after.from)) : nlohmann::json(nullptr) },
			{ "quiet", rounded(mean(after.quiet)) }
		};
		return { { "cycle", cycle }, { "events", events }, { "afterSequence", afterSequence } };
	}

	nlohmann::json RecallBehaviour::snapshot(const std::string& stage) const
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
		for (const auto& event : sequence)
		{
			atEvents[event.label] = rounded(activationAtPosition(outputField, event.position));
		}
		return {
			{ "stage", stage },
			{ "t", rounded(phenotype.getT()) },
			{ "input", { { "bumps", bumpsOf(inputField) } } },
			{ "output", { { "bumps", bumpsOf(outputField) }, { "median", rounded(medianActivation(outputField)) },
				{ "atEvents", atEvents } } }
		};
	}

	double RecallBehaviour::rounded(const double value)
	{
		return std::round(value * 100.0) / 100.0;
	}
}
