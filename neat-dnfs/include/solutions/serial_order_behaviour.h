#pragma once

#include <string>

#include "solutions/recall_behaviour.h"

namespace neat_dnfs
{
	/// @brief The serial-order task, scored by what the output does at every stage: its intended behaviour alone.
	///
	/// @details The demonstration is a start cue X (a boost of the input field "nf 1"), then D (180), C (120), A (20)
	/// and B (80) as Gaussian pulses at 100, 600, 800 and 1800 after it. After a gap of 700, X is given three times,
	/// starting recall cycles of 1000, 800 and 1500. createPhenotypeEnvironment() lays that schedule down as timed
	/// stimuli, which the evaluation runs on and every saved solution carries, so a saved file replays the evaluation
	/// in dnf-composer. Every field is sampled every 10 through that schedule, and scored against what the hand-built
	/// design of 18 October 2026 (tests/fixtures/serial-order-fitness-test-reference.dnf) does there, by the rules of
	/// RecallBehaviour:
	/// - the input detects each event;
	/// - the output "nf 2" stays below threshold through the demonstration;
	/// - after each X the output peaks at D, C, A and B in turn, one at a time. Recall is sampled through the 600 after X
	///   and scored by order, not by when each peak comes.
	/// - Once an event's peak ends, it stays released below threshold.
	/// - Once the sequence is over, the output has no peak at any sample until the next X. The sequence is over once
	///   every event has peaked and been released, or 600 after X if that never happens.
	///
	/// The five partial fitness terms are those stages: f1 the input, f2 the quiet output during the demonstration, f3
	/// the recall in order, f4 the release of each recalled event and f5 the output after recall. Nothing below
	/// threshold earns credit: SerialOrderBehaviourGuided adds the pre-shapes and troughs the reference design has.
	///
	/// Every evaluation also records a trace in SolutionParameters::evaluationTrace. At each point it is scored, the
	/// trace gives:
	/// - the input's bumps;
	/// - the output's bumps, its median activation and its activation at each event.
	///
	/// For each recall cycle it also gives each event's first peak, best recall score, recall and release.
	class SerialOrderBehaviour : public RecallBehaviour
	{
	public:
		/// @brief Builds a task solution whose genome is created from @p topology.
		/// @param topology Fields to start from: the input and the output.
		explicit SerialOrderBehaviour(const SolutionTopology& topology);
		/// @brief Builds a task solution whose genome is decoded from an existing phenotype, such as a template.
		/// @param initialTopology Fields the task declares: the input and the output.
		/// @param phenotype Simulation the genome is translated from.
		SerialOrderBehaviour(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype);
		/// @brief Creates a solution with a fresh genome built from this one's initial topology and a new id.
		/// @return The new solution.
		SolutionPtr clone() const override;
		/// @brief Creates an independent solution with this one's phenotype and a new id.
		/// @return The new solution.
		SolutionPtr copy() const override;
	protected:
		/// @brief Builds a task variant whose genome is created from @p topology.
		/// @param topology Fields to start from: the input and the output.
		/// @param guides Whether the fitness also pays the guides.
		/// @param taskName Display name, which also names the run's data folder.
		/// @param slug Task slug whose config supplies the fitness weights.
		SerialOrderBehaviour(const SolutionTopology& topology, Guides guides, const std::string& taskName,
			const std::string& slug);
		/// @brief Builds a task variant whose genome is decoded from an existing phenotype, such as a template.
		/// @param initialTopology Fields the task declares: the input and the output.
		/// @param phenotype Simulation the genome is translated from.
		/// @param guides Whether the fitness also pays the guides.
		/// @param taskName Display name, which also names the run's data folder.
		/// @param slug Task slug whose config supplies the fitness weights.
		SerialOrderBehaviour(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype,
			Guides guides, const std::string& taskName, const std::string& slug);
	private:
		void testPhenotype() override;
		void createPhenotypeEnvironment() override;
		[[nodiscard]] bool inputSettledAt(double time) const override;
	};

	/// @brief SerialOrderBehaviour that also pays the guides: what the reference design holds below threshold.
	///
	/// @details On top of the intended behaviour:
	/// - during the demonstration, the output is pre-shaped at every stored event, the earliest highest. Pre-shapes count
	///   as memory only as far as the input is at rest and the output has no peak.
	/// - While they wait, the events not yet recalled stay pre-shaped in order, counted as far as the input is at rest.
	///   A cycle where no event waits is not scored.
	/// - A released event earns half its release for lying below threshold and half for lying below the rest of the
	///   field, in full from 2 below.
	/// - Once the sequence is over, half the stop is for every event lying below the rest of the field until the next X,
	///   each as far as it was recalled in its place.
	///
	/// The seven partial fitness terms are: f1 the input, f2 the quiet output and f3 its pre-shapes during the
	/// demonstration, f4 the recall in order, f5 the pre-shapes still to come, f6 the release of each recalled event and
	/// f7 the output after recall.
	class SerialOrderBehaviourGuided final : public SerialOrderBehaviour
	{
	public:
		/// @brief Builds a task solution whose genome is created from @p topology.
		/// @param topology Fields to start from: the input and the output.
		explicit SerialOrderBehaviourGuided(const SolutionTopology& topology);
		/// @brief Builds a task solution whose genome is decoded from an existing phenotype, such as a template.
		/// @param initialTopology Fields the task declares: the input and the output.
		/// @param phenotype Simulation the genome is translated from.
		SerialOrderBehaviourGuided(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype);
		/// @brief Creates a solution with a fresh genome built from this one's initial topology and a new id.
		/// @return The new solution.
		SolutionPtr clone() const override;
		/// @brief Creates an independent solution with this one's phenotype and a new id.
		/// @return The new solution.
		SolutionPtr copy() const override;
	};
}
