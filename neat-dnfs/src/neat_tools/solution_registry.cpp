#include "neat_tools/solution_registry.h"

#include <stdexcept>

#include "constants.h"
#include "neat_tools/ablation_presets.h"

#include "solutions/and.h"
#include "solutions/xor.h"
#include "solutions/detection_instability.h"
#include "solutions/memory_instability.h"
#include "solutions/selection_instability.h"
#include "solutions/memory_trace.h"
#include "solutions/delayed_match_to_sample.h"
#include "solutions/inhibition_of_return.h"

namespace neat_dnfs
{
	namespace
	{
		template <typename SolutionType>
		std::unique_ptr<Solution> makeFromTopology(const SolutionTopology& topology)
		{
			return std::make_unique<SolutionType>(topology);
		}

		template <typename SolutionType>
		std::unique_ptr<Solution> makeFromTemplate(const SolutionTopology& topology, const dnf_composer::Simulation& phenotype)
		{
			return std::make_unique<SolutionType>(topology, phenotype);
		}

		template <typename SolutionType>
		constexpr TaskEntry entry(std::string_view slug, std::string_view templateFile, int inputs, int outputs)
		{
			return TaskEntry{ slug, templateFile, inputs, outputs,
				&makeFromTopology<SolutionType>, &makeFromTemplate<SolutionType> };
		}
	}

	const std::vector<TaskEntry>& taskEntries()
	{
		static const std::vector<TaskEntry> entries{
			entry<AND>("and", "and.json", 2, 1),
			entry<XOR>("xor", "xor.json", 2, 1),
			entry<DetectionInstability>("detection-instability", "detection-instability.json", 1, 1),
			entry<MemoryInstability>("memory-instability", "memory-instability.json", 1, 1),
			entry<SelectionInstability>("selection-instability", "selection-instability.json", 1, 1),
			entry<MemoryTrace>("memory-trace", "memory-trace.json", 2, 1),
			entry<DelayedMatchToSample>("dmts", "delayed-match-to-sample.json", 1, 1),
			entry<InhibitionOfReturn>("ior", "inhibition-of-return.json", 1, 1),
		};
		return entries;
	}

	const TaskEntry* findTask(const std::string_view slug)
	{
		for (const auto& task : taskEntries())
		{
			if (task.slug == slug)
			{
				return &task;
			}
		}
		return nullptr;
	}

	SolutionTopology defaultTopologyFor(const TaskEntry& task)
	{
		using dnf_composer::element::ElementDimensions;
		const ElementDimensions dims{ DimensionConstants::xSize, DimensionConstants::dx };

		std::vector<std::pair<FieldGeneType, ElementDimensions>> genes;
		for (int i = 0; i < task.inputs; ++i)
		{
			genes.emplace_back(FieldGeneType::INPUT, dims);
		}
		for (int i = 0; i < task.outputs; ++i)
		{
			genes.emplace_back(FieldGeneType::OUTPUT, dims);
		}
		return SolutionTopology(genes);
	}

	namespace
	{
		bool takesValue(const std::string& flag)
		{
			return flag == "--task" || flag == "--ablation" || flag == "--template"
				|| flag == "--runs" || flag == "--pop" || flag == "--gens"
				|| flag == "--target" || flag == "--evals" || flag == "--config";
		}
	}

	CliOptions parseCliOptions(const int argc, char* argv[])
	{
		CliOptions options;
		for (int i = 1; i < argc; ++i)
		{
			const std::string flag = argv[i];

			if (flag == "--list")
			{
				options.listRequested = true;
				continue;
			}
			if (flag == "--help" || flag == "-h")
			{
				options.helpRequested = true;
				continue;
			}
			if (!takesValue(flag))
			{
				throw std::invalid_argument("Unrecognized flag '" + flag + "'.");
			}
			if (i + 1 >= argc)
			{
				throw std::invalid_argument("Flag '" + flag + "' requires a value.");
			}
			const std::string value = argv[++i];

			if (flag == "--task") options.task = value;
			else if (flag == "--ablation") options.ablation = value;
			else if (flag == "--template") options.templateFile = value;
			else if (flag == "--runs") options.runs = std::stoi(value);
			else if (flag == "--pop") options.populationSize = std::stoi(value);
			else if (flag == "--gens") options.numGenerations = std::stoi(value);
			else if (flag == "--target") options.targetFitness = std::stod(value);
			else if (flag == "--evals") options.evaluations = std::stoi(value);
			else if (flag == "--config") options.config = value;
		}
		return options;
	}

	void printUsage(std::ostream& out, const std::string_view exeName)
	{
		out << "Usage: " << exeName << " [--task NAME] [--ablation NAME] [--template PATH]\n"
			<< "           [--config PATH] [--runs N] [--pop N] [--gens N] [--target F] [--evals N]\n"
			<< "           [--list] [--help]\n\n"
			<< "  --task NAME       Task to evolve/evaluate (default varies by binary). See --list.\n"
			<< "  --ablation NAME   Ablation preset to apply before initialize() (default: none). See --list.\n"
			<< "  --template PATH   Template JSON to load a starting solution from (default: the task's own).\n"
			<< "  --config PATH     Reference hyperparameter JSON (default: <root>/config/neat_dnfs.json,\n"
			<< "                    where <root> is $NEAT_DNFS_ROOT, else the directory holding this\n"
			<< "                    binary, else the source tree it was built from).\n"
			<< "  --runs N          Number of independent Population::evolve() runs.\n"
			<< "  --pop N           Population size per run.\n"
			<< "  --gens N          Max generations per run.\n"
			<< "  --target F        Target fitness that ends a run early.\n"
			<< "                    (--runs/--pop/--gens/--target override the config files.)\n"
			<< "  --evals N         Number of evaluate() calls (solution-evaluation binary only).\n"
			<< "  --list            List available tasks and ablation presets, then exit.\n"
			<< "  --help            Show this message and exit.\n";
	}

	void printTaskAndAblationList(std::ostream& out)
	{
		out << "Tasks:\n";
		for (const auto& task : taskEntries())
		{
			out << "  " << task.slug << "\n";
		}
		out << "Ablations:\n";
		for (const auto& name : AblationPresets::names())
		{
			out << "  " << name << "\n";
		}
	}
}
