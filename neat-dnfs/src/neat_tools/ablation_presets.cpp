#include "neat_tools/ablation_presets.h"

#include <algorithm>
#include <filesystem>

#include "neat_tools/config_loader.h"
#include "constants.h"

namespace neat_dnfs
{
	namespace
	{
		std::string presetDirectory()
		{
			return std::string(PROJECT_DIR) + "/config/ablations";
		}

		std::string presetPath(const std::string_view name)
		{
			return presetDirectory() + "/" + std::string(name) + ".json";
		}

		// applyByName()'s argument comes straight from --ablation on the CLI, so
		// unlike the hardcoded names above it is untrusted: a name containing a
		// path separator or ".." could escape config/ablations/ and merge an
		// arbitrary JSON file into the running config.
		bool isSimplePresetName(const std::string_view name)
		{
			if (name.empty() || name == "." || name == "..")
			{
				return false;
			}
			return name.find_first_of("/\\") == std::string_view::npos;
		}
	}

	// Each preset's field values live in its own file under config/ablations/.
	// The files are flat and self-contained -- no preset inherits from another --
	// so the whole condition is visible in the one file you open. The rationale
	// for each condition stays here, since JSON cannot carry it.

	void AblationPresets::noGrowthIOOnly()
	{
		// Toggle stays live: it changes no gene, so gene set, innovation numbers,
		// genome size and speciation distance are all unaffected by it, meaning
		// "no growth" still holds strictly. It is this codebase's only route to
		// pruning a connection (kernel amplitudes are bounded away from zero), the
		// analogue of a fixed-topology system's weights decaying to zero.
		ConfigLoader::applyAblation(presetPath("no-growth-io-only"));
	}

	void AblationPresets::noGrowthReferenceHiddenFieldCount()
	{
		// Seeds exactly the hidden-field count this experiment treats as its
		// reference (both bounds are referenceHiddenFieldsMin), rather than a
		// fixed one: "one hidden field" is only the right number for tasks that
		// need one to succeed.
		ConfigLoader::applyAblation(presetPath("no-growth-reference-hidden-field-count"));
	}

	void AblationPresets::noSpeciation()
	{
		// Paper (Stanley & Miikkulainen 2002, S5.5): removing speciation from a
		// minimally-starting NEAT is degenerate - no structural innovation can
		// survive, so the population gets stuck in minimal form. Nonspeciated NEAT
		// must instead start from a random initial population, which is why this
		// preset deliberately changes two axes at once (speciation off, random start)
		// rather than one.
		ConfigLoader::applyAblation(presetPath("no-speciation"));
	}

	void AblationPresets::noCrossover()
	{
		ConfigLoader::applyAblation(presetPath("no-crossover"));
	}

	void AblationPresets::randomInitialTopology()
	{
		// Seeded across the experiment's reference hidden-field range. That range
		// should floor at 1, not 0: a genome seeded with zero hidden fields is the
		// stuck-minimal case a random start is meant to avoid (noSpeciation() uses
		// the same reference bounds for the same reason).
		ConfigLoader::applyAblation(presetPath("random-initial-topology"));
	}

	bool AblationPresets::applyByName(const std::string_view name)
	{
		// Any file in config/ablations/ is a usable preset -- adding a condition
		// means adding a JSON file, with no matching change here.
		if (!isSimplePresetName(name))
		{
			return false;
		}
		const std::string path = presetPath(name);
		if (!std::filesystem::exists(path))
		{
			return false;
		}
		ConfigLoader::applyAblation(path);
		return true;
	}

	std::vector<std::string> AblationPresets::names()
	{
		std::vector<std::string> result;
		for (const auto& entry : std::filesystem::directory_iterator(presetDirectory()))
		{
			if (entry.path().extension() == ".json")
			{
				result.push_back(entry.path().stem().string());
			}
		}
		// Directory order is not guaranteed; sort to keep the stable order that
		// --list and the "unknown ablation" message promise.
		std::sort(result.begin(), result.end());
		return result;
	}
}
