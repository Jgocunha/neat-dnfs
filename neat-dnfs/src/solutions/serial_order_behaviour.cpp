#include "solutions/serial_order_behaviour.h"

#include <algorithm>
#include <array>
#include <format>
#include <numeric>

#include "neat_tools/utils.h"

namespace neat_dnfs
{
	namespace
	{
		using serial_order_schedule::Event;
		using serial_order_schedule::sequence;
		using serial_order_schedule::pulseDuration;
		using serial_order_schedule::cueTimes;
		using serial_order_schedule::endTime;
		using serial_order_schedule::eventsShownBy;
		using serial_order_schedule::positionsOf;
		using tools::utils::mean;
		// By then the reference design has recalled and suppressed every event on every cycle.
		constexpr double recallEndTime = 600.0;
		// Points of the recall window the trace snapshots: inside the reference design's peak at D, C, A and B on every
		// cycle.
		constexpr std::array<double, SerialOrderBehaviour::eventCount> recallSnapshotTimes{ 150.0, 290.0, 400.0, 500.0 };
	}

	SerialOrderBehaviour::SerialOrderBehaviour(const SolutionTopology& topology)
		: SerialOrderBehaviour(topology, Guides::unscored, "Serial Order Behaviour", "serial-order-behaviour")
	{
	}

	SerialOrderBehaviour::SerialOrderBehaviour(const SolutionTopology& initialTopology,
		const dnf_composer::Simulation& phenotype)
		: SerialOrderBehaviour(initialTopology, phenotype, Guides::unscored, "Serial Order Behaviour",
			"serial-order-behaviour")
	{
	}

	SerialOrderBehaviour::SerialOrderBehaviour(const SolutionTopology& topology, const Guides guides,
		const std::string& taskName, const std::string& slug)
		: RecallBehaviour(topology, guides)
	{
		name = taskName;
		loadFitnessWeights(slug, guides == Guides::scored ? 7 : 5);
	}

	SerialOrderBehaviour::SerialOrderBehaviour(const SolutionTopology& initialTopology,
		const dnf_composer::Simulation& phenotype, const Guides guides, const std::string& taskName,
		const std::string& slug)
		: RecallBehaviour(initialTopology, phenotype, guides)
	{
		name = taskName;
		loadFitnessWeights(slug, guides == Guides::scored ? 7 : 5);
	}

	SolutionPtr SerialOrderBehaviour::clone() const
	{
		SerialOrderBehaviour solution(initialTopology);
		return std::make_shared<SerialOrderBehaviour>(solution);
	}

	SolutionPtr SerialOrderBehaviour::copy() const
	{
		SerialOrderBehaviour solution(initialTopology, phenotype);
		return std::make_shared<SerialOrderBehaviour>(solution);
	}

	void SerialOrderBehaviour::testPhenotype()
	{
		parameters.fitness = 0.0;
		parameters.partialFitness.clear();

		std::vector<double> inputPeaks, inputRest, outputQuiet, outputMemory;
		std::vector<double> recallPeaks, recallPreShapes, recallReleases, afterRecall;
		nlohmann::json trace = nlohmann::json::array();
		const auto observeDemonstration = [&](const double time, const std::optional<double> inputAtRest) {
			const double quiet = noBumps(outputField);
			outputQuiet.push_back(quiet);
			// Memory is what the output holds below threshold once the input has let go of each event.
			const size_t stored = eventsShownBy(time);
			if (inputAtRest && stored > 0)
			{
				outputMemory.push_back(preShapedInOrder(positionsOf(0, stored)) * *inputAtRest * quiet);
			}
		};

		// the schedule a saved solution carries, so its file replays this evaluation
		createPhenotypeEnvironment();
		initSimulation();

		// demonstration: X, then D, C, A and B
		sampleUntil(sequence.front().onset, inputRest, observeDemonstration);
		trace.push_back(snapshot("before D"));
		for (size_t event = 0; event < sequence.size(); ++event)
		{
			sampleUntil(sequence[event].onset + pulseDuration, inputRest, observeDemonstration);
			trace.push_back(snapshot(std::format("{} shown", sequence[event].label)));
			inputPeaks.push_back(inputPeakedAt(sequence[event].position));

			sampleUntil(event + 1 < sequence.size() ? sequence[event + 1].onset : cueTimes[1], inputRest,
				observeDemonstration);
			trace.push_back(snapshot(std::format("after {}", sequence[event].label)));
		}

		// recall: each X should replay D, C, A and B one peak at a time, in that order, at any pace within the window
		nlohmann::json recallTrace = nlohmann::json::array();
		for (size_t cycle = 1; cycle < cueTimes.size(); ++cycle)
		{
			const double cue = cueTimes[cycle];
			CycleRecall recall{};
			AfterRecall after{};
			std::vector<double> waitingPreShapes;
			size_t snapshotsTaken = 0;
			sampleUntil(cue + recallEndTime, inputRest, [&](const double time, const std::optional<double> inputAtRest) {
				// While X still lifts the input, it holds whatever it held before X.
				observeRecall(time, recall, waitingPreShapes, inputAtRest.value_or(inputRest.back()));
				if (sequenceReleased(recall))
				{
					observeAfterRecall(time, after);
				}
				if (snapshotsTaken < recallSnapshotTimes.size() && time - cue == recallSnapshotTimes[snapshotsTaken])
				{
					trace.push_back(snapshot(std::format("recall {} {}", cycle, sequence[snapshotsTaken].label)));
					++snapshotsTaken;
				}
			});
			trace.push_back(snapshot(std::format("recall {} end", cycle)));
			recallPeaks.push_back(recallInOrder(recall));
			// A cycle where every event peaked by the first sample had nothing waiting to score.
			if (!waitingPreShapes.empty())
			{
				recallPreShapes.push_back(mean(waitingPreShapes));
			}
			recallReleases.push_back(releaseAfterRecall(recall));
			sampleUntil(cycle + 1 < cueTimes.size() ? cueTimes[cycle + 1] : endTime, inputRest,
				[&](const double time, std::optional<double>) { observeAfterRecall(time, after); });
			trace.push_back(snapshot(std::format("recall {} rest", cycle)));
			afterRecall.push_back(stoppedAfterRecall(recall, after));
			recallTrace.push_back(recallSummary(cycle, recall, after));
		}

		const double input = 0.5 * mean(inputPeaks) + 0.5 * mean(inputRest); // input peaks at the events, rests otherwise
		const double quiet = mean(outputQuiet);        // output below threshold through the demonstration
		const double memory = mean(outputMemory);      // output pre-shaped at the stored events, in order, held by itself
		const double recalled = mean(recallPeaks);     // the events recalled one at a time, in order
		const double toCome = recallPreShapes.empty() ? 0.0 : mean(recallPreShapes); // events still to recall pre-shaped, in order
		const double released = mean(recallReleases);  // each recalled event released
		const double stopped = mean(afterRecall);      // from the end of the sequence to the next X: no peak

		if (guides == Guides::scored)
		{
			parameters.partialFitness = { input, quiet, memory, recalled, toCome, released, stopped };
		}
		else
		{
			parameters.partialFitness = { input, quiet, recalled, released, stopped };
		}
		parameters.fitness = std::inner_product(parameters.partialFitness.begin(), parameters.partialFitness.end(),
			fitnessWeights.begin(), 0.0);
		parameters.evaluationTrace = nlohmann::json{ { "snapshots", trace }, { "recall", recallTrace } }.dump();
	}

	void SerialOrderBehaviour::createPhenotypeEnvironment()
	{
		const dnf_composer::element::ElementDimensions dimensions{ DimensionConstants::xSize, DimensionConstants::dx };
		for (const auto& [stimulusName, stimulusParameters] : serial_order_schedule::stimuli())
		{
			addTimedGaussianStimulus(stimulusName, inputField, stimulusParameters, dimensions);
		}
	}

	bool SerialOrderBehaviour::inputSettledAt(const double time) const
	{
		const auto pulsedRecently = [time](const double onset) {
			return time >= onset && time < onset + pulseDuration + settleTime;
		};
		return std::ranges::none_of(cueTimes, pulsedRecently)
			&& std::ranges::none_of(sequence, [&pulsedRecently](const Event& event) { return pulsedRecently(event.onset); });
	}

	SerialOrderBehaviourGuided::SerialOrderBehaviourGuided(const SolutionTopology& topology)
		: SerialOrderBehaviour(topology, Guides::scored, "Serial Order Behaviour Guided", "serial-order-behaviour-guided")
	{
	}

	SerialOrderBehaviourGuided::SerialOrderBehaviourGuided(const SolutionTopology& initialTopology,
		const dnf_composer::Simulation& phenotype)
		: SerialOrderBehaviour(initialTopology, phenotype, Guides::scored, "Serial Order Behaviour Guided",
			"serial-order-behaviour-guided")
	{
	}

	SolutionPtr SerialOrderBehaviourGuided::clone() const
	{
		SerialOrderBehaviourGuided solution(initialTopology);
		return std::make_shared<SerialOrderBehaviourGuided>(solution);
	}

	SolutionPtr SerialOrderBehaviourGuided::copy() const
	{
		SerialOrderBehaviourGuided solution(initialTopology, phenotype);
		return std::make_shared<SerialOrderBehaviourGuided>(solution);
	}
}
