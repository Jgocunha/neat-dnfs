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
	class SerialOrderProfile final : public Solution
	{
	public:
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
	private:
		void testPhenotype() override;
		void createPhenotypeEnvironment() override;

		/// @brief Lays the trial down on the built phenotype, runs it from the start, and records the input and output.
		/// @return The recorded profile.
		[[nodiscard]] ActivationProfile runTrial();
	};
}
