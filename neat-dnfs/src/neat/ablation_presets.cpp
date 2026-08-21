#include "neat/ablation_presets.h"
#include "constants.h"

namespace neat_dnfs
{
	void AblationPresets::noGrowthIOOnly()
	{
		AblationConstants::label = " No Growth IO Only";
		AblationConstants::disableAddFieldGene = true;
		AblationConstants::disableAddConnectionGene = true;
		AblationConstants::seedAllLegalConnections = true;
		// Toggle stays live: it changes no gene, so gene set, innovation numbers,
		// genome size and speciation distance are all unaffected by it, meaning
		// "no growth" still holds strictly. It is this codebase's only route to
		// pruning a connection (kernel amplitudes are bounded away from zero), the
		// analogue of a fixed-topology system's weights decaying to zero.
	}

	void AblationPresets::noGrowthOneHidden()
	{
		noGrowthIOOnly();
		AblationConstants::label = " No Growth One Hidden";
		AblationConstants::seedRandomHiddenFields = true;
		AblationConstants::seedHiddenFieldsMin = 1;
		AblationConstants::seedHiddenFieldsMax = 1;
	}

	void AblationPresets::noSpeciation()
	{
		AblationConstants::label = " No Speciation";
		AblationConstants::disableSpeciation = true;
		// Paper (Stanley & Miikkulainen 2002, S5.5): removing speciation from a
		// minimally-starting NEAT is degenerate - no structural innovation can
		// survive, so the population gets stuck in minimal form. Nonspeciated NEAT
		// must instead start from a random initial population, which is why this
		// preset deliberately changes two axes at once (speciation off, random start)
		// rather than one.
		AblationConstants::seedRandomHiddenFields = true;
		AblationConstants::seedHiddenFieldsMin = 1;
		AblationConstants::seedHiddenFieldsMax = 5;
		AblationConstants::seedRandomConnections = true;
	}

	void AblationPresets::noCrossover()
	{
		AblationConstants::label = " No Crossover";
		AblationConstants::disableCrossover = true;
	}

	void AblationPresets::randomInitialTopology()
	{
		AblationConstants::label = " Random Initial Topology";
		AblationConstants::seedRandomHiddenFields = true;
		// Floor of 1, not 0: a genome seeded with zero hidden fields is the
		// stuck-minimal case a random start is meant to avoid (noSpeciation() uses
		// the same [1,5] bounds for the same reason).
		AblationConstants::seedHiddenFieldsMin = 1;
		AblationConstants::seedHiddenFieldsMax = 5;
		AblationConstants::seedRandomConnections = true;
	}

	bool AblationPresets::applyByName(const std::string_view name)
	{
		if (name == "no-growth-io-only")
		{
			noGrowthIOOnly();
			return true;
		}
		if (name == "no-growth-one-hidden")
		{
			noGrowthOneHidden();
			return true;
		}
		if (name == "no-speciation")
		{
			noSpeciation();
			return true;
		}
		if (name == "no-crossover")
		{
			noCrossover();
			return true;
		}
		if (name == "random-initial-topology")
		{
			randomInitialTopology();
			return true;
		}
		return false;
	}

	std::vector<std::string_view> AblationPresets::names()
	{
		return {
			"no-growth-io-only",
			"no-growth-one-hidden",
			"no-speciation",
			"no-crossover",
			"random-initial-topology",
		};
	}
}
