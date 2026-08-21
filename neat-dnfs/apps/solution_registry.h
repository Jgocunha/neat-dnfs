#pragma once

// Shared task lookup and CLI parsing for the three example executables
// (neat-dnfs-evol, neat-dnfs-inc-evol, neat-dnfs-sol-eval). Each binary
// selects one of these tasks and, for the evolution binaries, optionally one
// ablation preset (see neat/ablation_presets.h) at runtime instead of at
// compile time, so an ablation sweep is a shell loop rather than a rebuild.

#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

#include "neat/solution.h"

namespace neat_dnfs::examples
{
	/// @brief One selectable task: how to build its default topology, its
	/// default template file (for the incremental/evaluation binaries), and
	/// how to construct the concrete Solution subclass either from scratch or
	/// from a loaded template phenotype.
	struct TaskEntry
	{
		std::string_view slug;
		std::string_view templateFile; // relative to templates/
		int inputs;
		int outputs;
		std::unique_ptr<Solution>(*makeFromTopology)(const SolutionTopology&);
		std::unique_ptr<Solution>(*makeFromTemplate)(const SolutionTopology&, const dnf_composer::Simulation&);
	};

	/// @return Every registered task, in a stable order (also the --list order).
	const std::vector<TaskEntry>& taskEntries();

	/// @return The entry for @p slug, or nullptr if it is not a registered task.
	const TaskEntry* findTask(std::string_view slug);

	/// @brief Builds the default topology for @p task: @c task.inputs INPUT genes
	/// followed by @c task.outputs OUTPUT genes, all sized DimensionConstants::xSize.
	SolutionTopology defaultTopologyFor(const TaskEntry& task);

	/// @brief Parsed command-line flags shared by all three example binaries.
	/// Any field left unset means "the binary's own default applies" -- these
	/// defaults differ per binary (evol vs incremental vs evaluation), so this
	/// struct itself carries none.
	struct CliOptions
	{
		std::optional<std::string> task;
		std::optional<std::string> ablation;
		std::optional<std::string> templateFile;
		std::optional<int> runs;
		std::optional<int> populationSize;
		std::optional<int> numGenerations;
		std::optional<double> targetFitness;
		std::optional<int> evaluations;
		bool listRequested = false;
		bool helpRequested = false;
	};

	/// @brief Parses argv[1..argc) into CliOptions.
	/// @throws std::invalid_argument if an unrecognized flag is given, or a
	/// flag that takes a value is missing one.
	CliOptions parseCliOptions(int argc, char* argv[]);

	/// @brief Prints "Usage: <exeName> [--task NAME] [...]" and a one-line
	/// description of every flag.
	void printUsage(std::ostream& out, std::string_view exeName);

	/// @brief Prints every registered task slug and ablation preset name, one
	/// per line, for --list and for "unknown task/ablation" error messages.
	void printTaskAndAblationList(std::ostream& out);
}
