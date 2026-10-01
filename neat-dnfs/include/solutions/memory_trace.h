#pragma once

#include "neat/solution.h"
#include "neat_tools/utils.h"

namespace neat_dnfs
{
	/// @brief Evolves a three-field architecture in which one field must become a memory trace.
	///
	/// @details In Dynamic Field Theory a memory trace is a layer whose activation
	/// evolves on a much slower time scale than the activation fields it supports: it builds up
	/// where a peak has been active, decays once that peak is gone, and in the meantime biases
	/// the field it feeds by lifting activation at the remembered locations closer to the
	/// threshold. The whole point
	/// of this task is to ask whether NEAT can discover such a slow field on its own: nothing
	/// in the task tells evolution to add a hidden field or to slow its time constant, so a
	/// solution can only score well if the topology it grows behaves like a memory trace.
	///
	/// The task uses two input fields and one output field. One input carries the encoding
	/// stimulus, the other the probe, and the output is where the remembered location must
	/// show itself. Position B is encoded; position A never is, and serves as the control
	/// that separates a genuine trace from a field that simply relays its input.
	///
	/// The run has three phases, and the eight partial fitness terms follow them in order:
	///
	/// Phase A -- the probe is applied at A with nothing encoded anywhere. The output must show
	/// only subthreshold preactivation at A (f1): input alone must not drive a detection
	/// instability, because an unremembered location has no trace supporting it. Removing the
	/// stimulus must then return the probe input field to its resting level (f2), so that later
	/// phases start from rest rather than from lingering activation.
	///
	/// Phase B -- the encoding stimulus is applied at B and held for a long interval. The output
	/// must again stay subthreshold, now at B (f3). This is the trace being laid down: the
	/// remembered location is marked by a hill of activation below threshold, not by a peak.
	///
	/// Phase C -- the encoding stimulus is removed and the probe is applied at A and B at once.
	/// The encoding input field must be back at its resting level (f4), so whatever survives is
	/// held internally rather than by the stimulus that created it. The probe input field must
	/// carry peaks at both A and B (f5), proving both locations are being probed equally. The
	/// output must nevertheless peak only at B (f6): only there does the accumulated trace add
	/// to the probe strongly enough to cross the threshold, which is the detection instability
	/// that the trace has biased. That peak must still be present after a further interval (f7),
	/// so it is self-stabilized rather than a transient, and must finally decay away (f8), so
	/// the trace dissipates instead of becoming permanent working memory.
	///
	/// Taken together the terms pin down the slow time scale from both sides: f7 forbids a
	/// trace that fades too quickly to be read out, and f8 forbids one that never fades. The
	/// A/B contrast in f1 and f6 forbids the degenerate solution of a field that peaks for any
	/// input, and f2 and f4 forbid one that is merely echoing a stimulus still being applied.
	class MemoryTrace final : public Solution
	{
	public:
		explicit MemoryTrace(const SolutionTopology& topology);
		MemoryTrace(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype);
		SolutionPtr clone() const override;
		SolutionPtr copy() const override;
	private:
		void testPhenotype() override;
		void createPhenotypeEnvironment() override;
	};
}
