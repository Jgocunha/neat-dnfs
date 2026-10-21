#pragma once

#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "neat/solution.h"

namespace neat_dnfs
{
	/// @brief The activation of some fields through a trial: each field's activation at every position, at each sample.
	struct ActivationProfile
	{
		/// @brief Simulation time of each sample.
		std::vector<double> times;
		/// @brief Each recorded field's activation, by field name: one row per sample, one value per position.
		std::map<std::string, std::vector<std::vector<double>>> fields;

		/// @brief Writes the profile as JSON, every activation rounded to three decimals.
		/// @return An object with the sample "times" and, under "fields", each field's rows by name.
		[[nodiscard]] nlohmann::json toJson() const;
		/// @brief Reads a profile written by toJson().
		/// @param json An object with the sample "times" and, under "fields", each field's rows by name.
		/// @return The profile.
		[[nodiscard]] static ActivationProfile fromJson(const nlohmann::json& json);
	};

	/// @brief The serial-order task, scored by how closely the activation of the input and output fields through the whole
	/// trial matches that of a reference design.
	///
	/// @details The trial is SerialOrderFitnessTest's (serial_order_schedule.h): a start cue X boosts the input field
	/// "nf 1", then D, C, A and B are shown as Gaussian pulses; after a hold, X is given three times, each starting a
	/// recall cycle. The evaluation records the activation of the input "nf 1" and the output "nf 2" at every position,
	/// every 10, from the start of the trial to its end, and compares it with referenceProfile(): the same recording of
	/// the Gauss-only reference design (tests/fixtures/serial-order-fitness-test-reference-gauss.dnf), saved in
	/// config/references/serial-order-profile.json.
	///
	/// At each moment, each field's activation across its positions is compared with the reference's at that moment by
	/// their concordance (tools::utils::concordance), which counts shape, height and level, each against how much the
	/// fields vary at that moment, with a tolerance of 1 for a flat field. The moment's score is (1 + concordance) / 2,
	/// averaged over the two fields: 1 for the reference's activation, 0.5 for an unrelated one, 0 for its mirror image.
	/// The fitness is the mean over every moment of the trial. Each partial fitness term is that mean over one of
	/// stages(), in order.
	class SerialOrderProfile final : public Solution
	{
	public:
		/// @brief A stretch of the trial whose score is one partial fitness term.
		struct Stage
		{
			/// @brief What happens in it, such as "event C" or "2 recall A".
			std::string name;
			/// @brief When it starts.
			double start;
			/// @brief When the next stage starts; for the last, the end of the trial.
			double end;
		};

		/// @brief Builds a task solution whose genome is created from @p topology.
		/// @param topology Fields to start from: the input and the output.
		explicit SerialOrderProfile(const SolutionTopology& topology);
		/// @brief Builds a task solution whose genome is decoded from an existing phenotype, such as a template.
		/// @param initialTopology Fields the task declares: the input and the output.
		/// @param phenotype Simulation the genome is translated from.
		SerialOrderProfile(const SolutionTopology& initialTopology, const dnf_composer::Simulation& phenotype);
		/// @brief Creates a solution with a fresh genome built from this one's initial topology and a new id.
		/// @return The new solution.
		SolutionPtr clone() const override;
		/// @brief Creates an independent solution with this one's phenotype and a new id.
		/// @return The new solution.
		SolutionPtr copy() const override;
		/// @brief Runs the trial on this solution's phenotype, as the evaluation does, and records it.
		/// @return The activation of the input and the output at every position, every 10, from the start of the trial
		/// to its end.
		[[nodiscard]] ActivationProfile recordProfile();
		/// @brief The profile every design is compared with: the Gauss-only reference design's, read once from
		/// config/references/serial-order-profile.json.
		/// @return The saved reference profile.
		[[nodiscard]] static const ActivationProfile& referenceProfile();
		/// @brief The stages the trial is scored in. The demonstration's follow the schedule: trigger, then "event D",
		/// "event C", "event A" and "event B", each until the next event or, for B, until the input is back at rest; then
		/// "wait" until the first recall. Each recall cycle's follow the reference design's own peaks: "X" until its peak
		/// at D, "recall D", "recall C", "recall A" and "recall B" from each event's first peak, and "rest" from the end
		/// of the peak at B to the next X, each named after its cycle, such as "2 recall C".
		/// @return The 24 stages, in order, from the start of the trial to its end.
		[[nodiscard]] static const std::vector<Stage>& stages();
	private:
		void testPhenotype() override;
		void createPhenotypeEnvironment() override;

		/// @brief Lays the trial down on the built phenotype, runs it from the start, and records the input and output.
		/// @return The recorded profile.
		[[nodiscard]] ActivationProfile runTrial();
	};
}
