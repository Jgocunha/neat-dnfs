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
			double position;
			double onset; // after the demonstration's cue X
		};

		// D, C, A and B: recall must replay them in this order.
		constexpr std::array<Event, 4> sequence{ { { 180.0, 100.0 }, { 120.0, 600.0 }, 
			{ 20.0, 800.0 }, { 80.0, 1800.0 } } };
		constexpr double pulseDuration = 50.0;
		constexpr double demonstrationDuration = 2550.0;
		constexpr std::array<double, 3> recallCycleDurations{ 1000.0, 800.0, 1500.0 };
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
		const auto scoreDemonstratedOutput = [&](const size_t storedCount) {
			outputQuiet.push_back(noBumps(outputField));
			if (storedCount > 0)
			{
				outputMemory.push_back(preShapedInOrder(positionsOf(0, storedCount)));
			}
		};

		// demonstration: X, then D, C, A and B
		initSimulation();
		cue();
		runFor(sequence.front().onset - pulseDuration);
		inputRest.push_back(closenessOfMeanActivationToRestingLevel(inputField));
		scoreDemonstratedOutput(0);
		for (size_t event = 0; event < sequence.size(); ++event)
		{
			presentEvent(sequence[event].position);
			inputPeaks.push_back(oneBumpAtPositionWithAmplitudeAndWidth(inputField, sequence[event].position,
				inputPeakAmplitude, inputPeakWidth));
			scoreDemonstratedOutput(event + 1);
			removeGaussianStimuli();

			const double nextOnset = event + 1 < sequence.size() ? sequence[event + 1].onset : demonstrationDuration;
			runFor(nextOnset - sequence[event].onset - pulseDuration);
			inputRest.push_back(closenessOfMeanActivationToRestingLevel(inputField));
			scoreDemonstratedOutput(event + 1);
		}

		// recall: each X replays D, C, A and B one peak at a time
		for (const double cycleDuration : recallCycleDurations)
		{
			cue();
			double elapsed = pulseDuration;
			for (size_t event = 0; event < sequence.size(); ++event)
			{
				runFor(recallPeakTimes[event] - elapsed);
				elapsed = recallPeakTimes[event];
				inputRest.push_back(closenessOfMeanActivationToRestingLevel(inputField));
				recallPeaks.push_back(oneBumpAtPositionWithAmplitudeAndWidth(outputField, sequence[event].position,
					outputPeakAmplitude, outputPeakWidth));
				if (event + 1 < sequence.size())
				{
					recallPreShapes.push_back(preShapedInOrder(positionsOf(event + 1, sequence.size())));
				}
				if (event > 0)
				{
					recallTroughs.push_back(troughsAt(positionsOf(0, event)));
				}
			}
			runFor(recallEndTime - elapsed);
			afterRecall.push_back(0.5 * noBumps(outputField) + 0.5 * troughsAt(positionsOf(0, sequence.size())));
			runFor(cycleDuration - recallEndTime);
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
	}

	void SerialOrderFitnessTest::createPhenotypeEnvironment()
	{
	}

	void SerialOrderFitnessTest::cue()
	{
		addBoostStimulus(inputField, GaussStimulusConstants::amplitude,
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runFor(pulseDuration);
		removeBoostStimuli();
	}

	void SerialOrderFitnessTest::presentEvent(const double position)
	{
		addGaussianStimulus(inputField,
			dnf_composer::element::GaussStimulusParameters{ GaussStimulusConstants::width, GaussStimulusConstants::amplitude,
				position, GaussStimulusConstants::circularity, GaussStimulusConstants::normalization },
			dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		runFor(pulseDuration);
	}

	void SerialOrderFitnessTest::runFor(const double duration)
	{
		runSimulation(static_cast<int>(std::lround(duration / SimulationConstants::deltaT)));
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

	double SerialOrderFitnessTest::troughsAt(const std::vector<double>& positions) const
	{
		std::vector<double> troughs;
		for (const double position : positions)
		{
			troughs.push_back(troughBelowFieldAtPosition(outputField, position));
		}
		return mean(troughs);
	}
}
