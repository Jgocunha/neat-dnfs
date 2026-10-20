#include "solutions/serial_order_fitness_test.h"

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
		constexpr std::array<Event, 4> sequence{ { { "D", 180.0, 100.0 }, { "C", 120.0, 600.0 },
			{ "A", 20.0, 800.0 }, { "B", 80.0, 1800.0 } } };
		constexpr double pulseDuration = 50.0;
		// The demonstration's X, then one X starting each recall cycle: 1000, 800 and 1500 long.
		constexpr std::array<double, 4> cueTimes{ 0.0, 2550.0, 3550.0, 4350.0 };
		constexpr double endTime = 5850.0;
		// X lifts the whole input field evenly, as a boost would, as a stimulus a saved file can replay.
		constexpr double cueWidth = 1e6;
		// When each event's recalled peak is scored, after the cue: inside the reference 
		// design's peak on every cycle.
		constexpr std::array<double, 4> recallPeakTimes{ 150.0, 290.0, 400.0, 500.0 };
		// By then the reference design has recalled and suppressed every event on every cycle.
		constexpr double recallEndTime = 600.0;

		// The reference design's peaks; amplitude and width carry little of a peak's credit.
		constexpr double inputPeakAmplitude = 7.5;
		constexpr double inputPeakWidth = 6.0;
		constexpr double outputPeakAmplitude = 9.0;
		constexpr double outputPeakWidth = 7.0;

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
		std::vector<double> recallPeaks, recallPreShapes, recallTroughs, afterRecall;
		nlohmann::json trace = nlohmann::json::array();
		const auto scoreDemonstratedOutput = [&](const size_t storedCount) {
			outputQuiet.push_back(noBumps(outputField));
			if (storedCount > 0)
			{
				outputMemory.push_back(preShapedInOrder(positionsOf(0, storedCount)));
			}
		};

		// the schedule a saved solution carries, so its file replays this evaluation
		createPhenotypeEnvironment();
		initSimulation();

		// demonstration: X, then D, C, A and B
		runUntil(sequence.front().onset);
		trace.push_back(snapshot("before D"));
		inputRest.push_back(closenessOfMeanActivationToRestingLevel(inputField));
		scoreDemonstratedOutput(0);
		for (size_t event = 0; event < sequence.size(); ++event)
		{
			runUntil(sequence[event].onset + pulseDuration);
			trace.push_back(snapshot(std::format("{} shown", sequence[event].label)));
			inputPeaks.push_back(oneBumpAtPositionWithAmplitudeAndWidth(inputField, sequence[event].position,
				inputPeakAmplitude, inputPeakWidth));
			scoreDemonstratedOutput(event + 1);

			runUntil(event + 1 < sequence.size() ? sequence[event + 1].onset : cueTimes[1]);
			trace.push_back(snapshot(std::format("after {}", sequence[event].label)));
			inputRest.push_back(closenessOfMeanActivationToRestingLevel(inputField));
			scoreDemonstratedOutput(event + 1);
		}

		// recall: each X replays D, C, A and B one peak at a time
		for (size_t cycle = 1; cycle < cueTimes.size(); ++cycle)
		{
			const double cue = cueTimes[cycle];
			for (size_t event = 0; event < sequence.size(); ++event)
			{
				runUntil(cue + recallPeakTimes[event]);
				trace.push_back(snapshot(std::format("recall {} {}", cycle, sequence[event].label)));
				inputRest.push_back(closenessOfMeanActivationToRestingLevel(inputField));
				recallPeaks.push_back(recallPeakAt(sequence[event].position));
				if (event + 1 < sequence.size())
				{
					recallPreShapes.push_back(preShapedInOrder(positionsOf(event + 1, sequence.size())));
				}
				if (event > 0)
				{
					recallTroughs.push_back(troughsAt(positionsOf(0, event)));
				}
			}
			runUntil(cue + recallEndTime);
			trace.push_back(snapshot(std::format("recall {} end", cycle)));
			afterRecall.push_back(0.5 * noBumps(outputField) + 0.5 * troughsAt(positionsOf(0, sequence.size())));
			runUntil(cycle + 1 < cueTimes.size() ? cueTimes[cycle + 1] : endTime);
			trace.push_back(snapshot(std::format("recall {} rest", cycle)));
			afterRecall.push_back(0.5 * noBumps(outputField) + 0.5 * troughsAt(positionsOf(0, sequence.size())));
		}

		const double f1 = 0.5 * mean(inputPeaks) + 0.5 * mean(inputRest); // input peaks at the events, rests otherwise
		const double f2 = mean(outputQuiet);     // output below threshold through the demonstration
		const double f3 = mean(outputMemory);    // output pre-shaped at the stored events, in order
		const double f4 = mean(recallPeaks);     // the right peak at each point of recall
		const double f5 = mean(recallPreShapes); // events still to recall pre-shaped, in order
		const double f6 = mean(recallTroughs);   // recalled events pushed below the field
		const double f7 = mean(afterRecall);     // after recall: no peak, every event below the field

		parameters.partialFitness = { f1, f2, f3, f4, f5, f6, f7 };

		const auto& w = fitnessWeights;
		parameters.fitness = w[0] * f1 + w[1] * f2 + w[2] * f3 + w[3] * f4 + w[4] * f5 + w[5] * f6 + w[6] * f7;
		parameters.evaluationTrace = nlohmann::json{ { "snapshots", trace } }.dump();
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

	double SerialOrderFitnessTest::recallPeakAt(const double position) const
	{
		// A lone peak anywhere would still earn the bump score's count, height and width shares: paid only near the event.
		return oneBumpAtPositionWithAmplitudeAndWidth(outputField, position, outputPeakAmplitude, outputPeakWidth)
			* closenessOfNearestBumpToPosition(outputField, position);
	}

	double SerialOrderFitnessTest::troughsAt(const std::vector<double>& positions) const
	{
		std::vector<double> troughs;
		for (const double position : positions)
		{
			troughs.push_back(troughBelowFieldAtPosition(outputField, position));
		}
		return mean(troughs);
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
