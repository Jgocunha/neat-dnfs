#pragma once

#include <nlohmann/json.hpp>

#include "neat/solution.h"

namespace neat_dnfs
{
	/// @brief The serial-order task, scored by how the fields respond at every stage, after a reference design.
	///
	/// @details The demonstration is a start cue X (a boost of the input field "nf 1"), then D (180), C (120), A (20)
	/// and B (80) as Gaussian pulses at 100, 600, 800 and 1800 after it. After a gap of 700, X is given three times,
	/// starting recall cycles of 1000, 800 and 1500. createPhenotypeEnvironment() lays that schedule down as timed
	/// stimuli, which the evaluation runs on and every saved solution carries, so a saved file replays the evaluation
	/// in dnf-composer. Every field is scored at the end of each interval of that
	/// schedule, against what the hand-built design of 18 October 2026 (tests/fixtures/
	/// serial-order-fitness-test-reference.dnf) does there:
	/// - the input peaks at each event and is at rest otherwise;
	/// - the output "nf 2" stays below threshold through the demonstration, pre-shaped at every stored event, the
	///   earliest highest;
	/// - after each X the output peaks at D, C, A and B in turn. The events still to come stay pre-shaped in order,
	///   and each recalled one is pushed below the rest of the field;
	/// - once recall ends, the output has no peak and every event lies below the rest of the field.
	///
	/// The seven partial fitness terms are those stages: f1 the input, f2 the quiet output and f3 its pre-shapes
	/// during the demonstration, f4 the recalled peaks, f5 the pre-shapes still to come, f6 the troughs during recall
	/// and f7 the output after recall.
	///
	/// Every evaluation also records a trace in SolutionParameters::evaluationTrace. At each point it is scored, the
	/// trace gives:
	/// - the input's bumps;
	/// - the output's bumps, its median activation and its activation at each event.
	class SerialOrderFitnessTest final : public Solution
	{
	public:
		/// @brief Builds a task solution whose genome is created from @p topology.
		/// @param topology Fields to start from: the input and the output.
		explicit SerialOrderFitnessTest(const SolutionTopology& topology);
		/// @brief Builds a task solution whose genome is decoded from an existing phenotype, such as a template.
		/// @param initialTopology Fields the task declares: the input and the output.
		/// @param phenotype Simulation the genome is translated from.
		SerialOrderFitnessTest(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype);
		/// @brief Creates a solution with a fresh genome built from this one's initial topology and a new id.
		/// @return The new solution.
		SolutionPtr clone() const override;
		/// @brief Creates an independent solution with this one's phenotype and a new id.
		/// @return The new solution.
		SolutionPtr copy() const override;
	private:
		void testPhenotype() override;
		void createPhenotypeEnvironment() override;

		/// @brief Simulates until the simulation time reaches @p time.
		/// @param time Simulation time to stop at.
		void runUntil(double time);
		/// @brief Scores the output pre-shaped at @p positions, above the rest of the field and in that order.
		/// @param positions Event positions, highest pre-shape expected first.
		/// @return The mean pre-shape, with half of it kept only as far as the pre-shapes are in order.
		[[nodiscard]] double preShapedInOrder(const std::vector<double>& positions) const;
		/// @brief Scores the output pushed below the rest of the field at every one of @p positions.
		/// @param positions Event positions.
		/// @return The mean trough score over @p positions.
		[[nodiscard]] double troughsAt(const std::vector<double>& positions) const;
		/// @brief Records what the input and output look like now, for the evaluation trace.
		/// @param stage Name of the point in the schedule being scored, such as "C shown" or "recall 2 A".
		/// @return The time, the input's bumps, and the output's bumps, median and activation at each event.
		[[nodiscard]] nlohmann::json snapshot(const std::string& stage) const;
	};
}
