#include "solutions/serial_order_fitness_test.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>

namespace neat_dnfs
{
	namespace
	{
		const std::string inputField = "nf 1";
		const std::string outputField = "nf 2";

		struct Event
		{
			const char* label;
			double position;
			double onset; // after the demonstration's cue X
		};

		// D, C, A and B: recall must replay them in this order.
		constexpr std::array<Event, SerialOrderFitnessTest::eventCount> sequence{ { { "D", 180.0, 100.0 }, { "C", 120.0, 600.0 },
			{ "A", 20.0, 800.0 }, { "B", 80.0, 1800.0 } } };
		constexpr double pulseDuration = 50.0;
		// The demonstration's X, then one X starting each recall cycle: 1000, 800 and 1500 long.
		constexpr std::array<double, 4> cueTimes{ 0.0, 2550.0, 3550.0, 4350.0 };
		constexpr double endTime = 5850.0;
		// X lifts the whole input field evenly, as a boost would, as a stimulus a saved file can replay.
		constexpr double cueWidth = 1e6;
		// Every stage is sampled this often.
		constexpr double sampleInterval = 10.0;
		// After a pulse ends, a detection field's peak has this long to decay before the input must be back at rest: five
		// time constants of the reference's input.
		constexpr double settleTime = 50.0;
		// By then the reference design has recalled and suppressed every event on every cycle.
		constexpr double recallEndTime = 600.0;
		// Points of the recall window the trace snapshots: inside the reference design's peak at D, C, A and B on every
		// cycle.
		constexpr std::array<double, SerialOrderFitnessTest::eventCount> recallSnapshotTimes{ 150.0, 290.0, 400.0, 500.0 };
		// An event counts as peaking from this recall score: one bump near it, about a quarter of full height or more.
		constexpr double peakingScore = 0.25;

		// The reference design's peaks; amplitude and width carry little of a peak's credit.
		constexpr double inputPeakAmplitude = 7.5;
		constexpr double inputPeakWidth = 6.0;
		constexpr double outputPeakAmplitude = 9.0;
		constexpr double outputPeakWidth = 7.0;

		// Whether the input should be back at rest: no pulse, X or event, started within a pulse and its settling time.
		bool inputSettledAt(const double time)
		{
			const auto pulsedRecently = [time](const double onset) {
				return time >= onset && time < onset + pulseDuration + settleTime;
			};
			return std::ranges::none_of(cueTimes, pulsedRecently)
				&& std::ranges::none_of(sequence, [&pulsedRecently](const Event& event) { return pulsedRecently(event.onset); });
		}

		size_t eventsShownBy(const double time)
		{
			return static_cast<size_t>(std::ranges::count_if(sequence, [time](const Event& event) { return event.onset < time; }));
		}

		std::vector<double> positionsOf(const size_t first, const size_t last)
		{
			std::vector<double> positions;
			for (size_t i = first; i < last; ++i)
			{
				positions.push_back(sequence[i].position);
			}
			return positions;
		}

		double mean(const std::vector<double>& values)
		{
			return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
		}

		// Two decimals keep a trace small and are finer than anything it is read for.
		double rounded(const double value)
		{
			return std::round(value * 100.0) / 100.0;
		}
	}

	SerialOrderFitnessTest::SerialOrderFitnessTest(const SolutionTopology& topology)
		: Solution(topology)
	{
		name = "Serial Order Fitness Test";
		loadFitnessWeights("serial-order-fitness-test", 7);
	}

	SerialOrderFitnessTest::SerialOrderFitnessTest(const SolutionTopology& initialTopology,
		const dnf_composer::Simulation& phenotype)
		: Solution(initialTopology, phenotype)
	{
		name = "Serial Order Fitness Test";
		loadFitnessWeights("serial-order-fitness-test", 7);
	}

	SolutionPtr SerialOrderFitnessTest::clone() const
	{
		SerialOrderFitnessTest solution(initialTopology);
		return std::make_shared<SerialOrderFitnessTest>(solution);
	}

	SolutionPtr SerialOrderFitnessTest::copy() const
	{
		SerialOrderFitnessTest solution(initialTopology, phenotype);
		return std::make_shared<SerialOrderFitnessTest>(solution);
	}

	void SerialOrderFitnessTest::testPhenotype()
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
			inputPeaks.push_back(oneBumpAtPositionWithAmplitudeAndWidth(inputField, sequence[event].position,
				inputPeakAmplitude, inputPeakWidth));

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
			std::vector<double> waitingPreShapes;
			size_t snapshotsTaken = 0;
			sampleUntil(cue + recallEndTime, inputRest, [&](const double time, const std::optional<double> inputAtRest) {
				// While X still lifts the input, it holds whatever it held before X.
				observeRecall(time, recall, waitingPreShapes, inputAtRest.value_or(inputRest.back()));
				if (snapshotsTaken < recallSnapshotTimes.size() && time - cue == recallSnapshotTimes[snapshotsTaken])
				{
					trace.push_back(snapshot(std::format("recall {} {}", cycle, sequence[snapshotsTaken].label)));
					++snapshotsTaken;
				}
			});
			trace.push_back(snapshot(std::format("recall {} end", cycle)));
			recallPeaks.push_back(recallInOrder(recall));
			recallPreShapes.push_back(waitingPreShapes.empty() ? 0.0 : mean(waitingPreShapes));
			recallReleases.push_back(releaseAfterRecall(recall));
			afterRecall.push_back(0.5 * noBumps(outputField) + 0.5 * troughsAfterRecall(recall));
			sampleUntil(cycle + 1 < cueTimes.size() ? cueTimes[cycle + 1] : endTime, inputRest,
				[](double, std::optional<double>) {});
			trace.push_back(snapshot(std::format("recall {} rest", cycle)));
			afterRecall.push_back(0.5 * noBumps(outputField) + 0.5 * troughsAfterRecall(recall));
			recallTrace.push_back(recallSummary(cycle, recall));
		}

		const double f1 = 0.5 * mean(inputPeaks) + 0.5 * mean(inputRest); // input peaks at the events, rests otherwise
		const double f2 = mean(outputQuiet);     // output below threshold through the demonstration
		const double f3 = mean(outputMemory);    // output pre-shaped at the stored events, in order, held by itself
		const double f4 = mean(recallPeaks);     // the events recalled one at a time, in order
		const double f5 = mean(recallPreShapes); // events still to recall pre-shaped, in order
		const double f6 = mean(recallReleases);  // each recalled event released, below the field
		const double f7 = mean(afterRecall);     // after recall: no peak, every event below the field

		parameters.partialFitness = { f1, f2, f3, f4, f5, f6, f7 };

		const auto& w = fitnessWeights;
		parameters.fitness = w[0] * f1 + w[1] * f2 + w[2] * f3 + w[3] * f4 + w[4] * f5 + w[5] * f6 + w[6] * f7;
		parameters.evaluationTrace = nlohmann::json{ { "snapshots", trace }, { "recall", recallTrace } }.dump();
	}

	void SerialOrderFitnessTest::createPhenotypeEnvironment()
	{
		using namespace dnf_composer::element;
		const ElementDimensions dimensions{ DimensionConstants::xSize, DimensionConstants::dx };

		std::vector<std::pair<double, double>> cueWindows;
		for (const double cue : cueTimes)
		{
			cueWindows.emplace_back(cue, cue + pulseDuration);
		}
		addTimedGaussianStimulus("tgs X", inputField, TimedGaussStimulusParameters{ cueWidth,
			GaussStimulusConstants::amplitude, DimensionConstants::xSize / 2.0, cueWindows, false, false }, dimensions);

		for (const auto& event : sequence)
		{
			addTimedGaussianStimulus(std::format("tgs {}", event.label), inputField,
				TimedGaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude,
					event.position, { { event.onset, event.onset + pulseDuration } },
					GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
				dimensions);
		}
	}

	void SerialOrderFitnessTest::runUntil(const double time)
	{
		while (phenotype.getT() < time - 1e-9)
		{
			runSimulation(1);
		}
	}

	void SerialOrderFitnessTest::sampleUntil(const double time, std::vector<double>& inputRest,
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

	double SerialOrderFitnessTest::preShapedInOrder(const std::vector<double>& positions) const
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

	double SerialOrderFitnessTest::recalledAt(const double position) const
	{
		// A bump score pays a lone peak anywhere, and one barely above threshold, almost in full: count it only near the
		// event, and in proportion to its height.
		return closenessOfNearestBumpToPosition(outputField, position) * heightCreditOfNearestBumpToPosition(outputField, position);
	}

	void SerialOrderFitnessTest::observeRecall(const double time, CycleRecall& recall,
		std::vector<double>& waitingPreShapes, const double inputAtRest) const
	{
		// Recall is one peak at a time: a sample's credit is shared among the peaks up together.
		const auto peaksUp = static_cast<double>(getNeuralFieldOrThrow(outputField, "observeRecall")->getBumps().size());
		std::vector<double> waiting;
		for (size_t event = 0; event < sequence.size(); ++event)
		{
			const double position = sequence[event].position;
			EventRecall& observed = recall[event];
			const double recalled = recalledAt(position);
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
				const bool belowThreshold = activationAtPosition(outputField, position) < 0.0;
				observed.releaseScores.push_back(belowThreshold
					? 0.5 + 0.5 * troughBelowFieldAtPosition(outputField, position) : 0.0);
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

	double SerialOrderFitnessTest::recallInOrder(const CycleRecall& recall)
	{
		double credit = 0.0;
		for (size_t event = 0; event < recall.size(); ++event)
		{
			const auto& firstPeak = recall[event].firstPeak;
			if (!firstPeak)
			{
				continue;
			}
			// D must peak before any other event; every later event after the one before it.
			const bool inOrder = event == 0
				? std::ranges::none_of(recall, [&firstPeak](const EventRecall& other)
					{ return other.firstPeak && *other.firstPeak < *firstPeak; })
				: recall[event - 1].firstPeak && *recall[event - 1].firstPeak < *firstPeak;
			if (inOrder)
			{
				credit += recall[event].bestScore;
			}
		}
		return credit / static_cast<double>(recall.size());
	}

	double SerialOrderFitnessTest::releaseAfterRecall(const CycleRecall& recall)
	{
		double credit = 0.0;
		for (const auto& event : recall)
		{
			if (!event.releaseScores.empty())
			{
				credit += event.recalled * mean(event.releaseScores);
			}
		}
		return credit / static_cast<double>(recall.size());
	}

	double SerialOrderFitnessTest::troughsAfterRecall(const CycleRecall& recall) const
	{
		std::vector<double> troughs;
		for (size_t event = 0; event < recall.size(); ++event)
		{
			troughs.push_back(recall[event].recalled * troughBelowFieldAtPosition(outputField, sequence[event].position));
		}
		return mean(troughs);
	}

	nlohmann::json SerialOrderFitnessTest::recallSummary(const size_t cycle, const CycleRecall& recall)
	{
		nlohmann::json events = nlohmann::json::object();
		for (size_t event = 0; event < recall.size(); ++event)
		{
			const auto& observed = recall[event];
			events[sequence[event].label] = {
				{ "firstPeak", observed.firstPeak ? nlohmann::json(rounded(*observed.firstPeak)) : nlohmann::json(nullptr) },
				{ "bestScore", rounded(observed.bestScore) },
				{ "recalled", rounded(observed.recalled) },
				{ "release", observed.releaseScores.empty() ? nlohmann::json(nullptr)
					: nlohmann::json(rounded(mean(observed.releaseScores))) }
			};
		}
		return { { "cycle", cycle }, { "events", events } };
	}

	nlohmann::json SerialOrderFitnessTest::snapshot(const std::string& stage) const
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
}
