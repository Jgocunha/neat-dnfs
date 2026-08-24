#include "neat_tools/config_loader.h"

#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>

#include "constants.h"
#include "neat_tools/resource_paths.h"

namespace neat_dnfs
{
	nlohmann::json ConfigLoader::loadJsonFile(const std::string& path)
	{
		std::ifstream in(path);
		if (!in)
		{
			throw std::runtime_error("ConfigLoader: config file not found: " + path);
		}

		try
		{
			return nlohmann::json::parse(in);
		}
		catch (const nlohmann::json::parse_error& e)
		{
			throw std::runtime_error("ConfigLoader: '" + path + "' is not valid JSON: " + e.what());
		}
	}

	std::string ConfigLoader::defaultGlobalConfigPath()
	{
		return (paths::resourceRoot() / "config" / "neat_dnfs.json").generic_string();
	}

	void ConfigLoader::weights(const nlohmann::json& j, const char* key,
		std::vector<double>* target, const size_t expected)
	{
		field(j, key, target);
		if (target->size() != expected)
		{
			throw std::runtime_error("ConfigLoader: '" + std::string(key) + "' must have "
				+ std::to_string(expected) + " entries, got " + std::to_string(target->size()));
		}
	}

	std::string ConfigLoader::solutionConfigPath(const std::string& slug)
	{
		return (paths::resourceRoot() / "config" / "solutions" / (slug + ".json")).generic_string();
	}

	void ConfigLoader::loadGlobalConfig(const std::string& path)
	{
		activeConfig = loadJsonFile(path);
		activeConfigPath = path;
		applyConfig(activeConfig, activeConfigPath);
	}

	void ConfigLoader::loadConfig(const std::string& referencePath, const std::string& slug)
	{
		const std::string overridePath = solutionConfigPath(slug);
		activeConfig = loadJsonFile(referencePath);
		activeConfig.merge_patch(loadJsonFile(overridePath));
		activeConfigPath = referencePath + " + " + overridePath;
		applyConfig(activeConfig, activeConfigPath);
	}

	void ConfigLoader::applyAblation(const std::string& path)
	{
		auto merged = activeConfig;
		merged.merge_patch(loadJsonFile(path));
		applyConfig(merged, activeConfigPath + " + " + path);
	}

	std::vector<double> ConfigLoader::loadFitnessWeights(const std::string& slug, const size_t expectedCount)
	{
		// Every Solution construction calls this, and a run constructs a whole
		// population per generation, so the parsed result is cached per slug
		// rather than re-reading the file hundreds of thousands of times. The
		// mutex covers Population's parallel evaluation path.
		static std::mutex cacheMutex;
		static std::map<std::string, std::vector<double>> cache;

		const std::lock_guard<std::mutex> lock(cacheMutex);
		if (const auto it = cache.find(slug); it != cache.end())
		{
			if (it->second.size() != expectedCount)
			{
				throw std::runtime_error("ConfigLoader: cached fitness weights for '" + slug + "' have "
					+ std::to_string(it->second.size()) + " entries, but this caller expected "
					+ std::to_string(expectedCount) + ".");
			}
			return it->second;
		}

		const std::string path = solutionConfigPath(slug);
		const auto j = loadJsonFile(path);
		std::vector<double> result;
		try
		{
			weights(j.at("SolutionConstants"), "fitnessWeights", &result, expectedCount);
		}
		catch (const nlohmann::json::exception& e)
		{
			throw std::runtime_error("ConfigLoader: failed to load '" + path + "': " + e.what());
		}
		cache.emplace(slug, result);
		return result;
	}

	namespace
	{
		/// Reads a hidden-field bound, which a preset may give either as a plain
		/// number or as the name of one of AblationConstants' reference counts.
		/// The count a task needs is experiment-specific, so presets that mean
		/// "however many this experiment uses" name the reference instead of
		/// hardcoding a number that is only right for one task.
		void hiddenFieldBound(const nlohmann::json& j, const char* key, int* target)
		{
			const auto& value = j.at(key);
			if (value.is_string())
			{
				const auto name = value.get<std::string>();
				if (name == "referenceHiddenFieldsMin")
				{
					*target = AblationConstants::referenceHiddenFieldsMin;
					return;
				}
				if (name == "referenceHiddenFieldsMax")
				{
					*target = AblationConstants::referenceHiddenFieldsMax;
					return;
				}
				throw std::runtime_error("ConfigLoader: '" + std::string(key) + "' names unknown reference '"
					+ name + "'; expected referenceHiddenFieldsMin or referenceHiddenFieldsMax");
			}
			*target = value.get<int>();
		}

		// field()/weights()/hiddenFieldBound() below only ever read a *known*
		// key out of each block, so a mistyped struct name (e.g.
		// "SolutonConstants" in an override file) would otherwise merge in
		// silently and the whole block would keep the reference's values with
		// no error at all -- the one class of typo j.at() per-key can't catch,
		// since the block itself is just never read.
		void checkNoUnknownTopLevelKeys(const nlohmann::json& j, const std::string& path)
		{
			static const std::set<std::string> known = {
				"SimulationConstants", "DimensionConstants", "NoiseConstants",
				"GaussStimulusConstants", "NeuralFieldConstants", "KernelConstants",
				"GaussKernelConstants", "MexicanHatKernelConstants", "CompatibilityCoefficients",
				"GenomeMutationConstants", "FieldGeneConstants", "ConnectionGeneConstants",
				"SolutionConstants", "AblationConstants", "PopulationConstants",
			};
			for (const auto& item : j.items())
			{
				if (!known.contains(item.key()))
				{
					throw std::runtime_error("ConfigLoader: '" + path + "' has unknown top-level key '"
						+ item.key() + "'; check for a typo in the struct name.");
				}
			}
		}
	}

	void ConfigLoader::applyConfig(const nlohmann::json& j, const std::string& path)
	{
		// Each block mirrors the matching struct in constants.h field for field,
		// in declaration order, so the two can be diffed side by side. field()
		// uses json::at(), so any key missing from the file throws here rather
		// than leaving a value silently zero-initialised. checkNoUnknownTopLevelKeys()
		// covers the complementary case -- an extra/mistyped struct name -- but
		// only at this top level; a mistyped field *within* a correctly-named
		// struct still silently keeps the reference's value for that one field.
		try
		{
			checkNoUnknownTopLevelKeys(j, path);

			const auto& sim = j.at("SimulationConstants");
			field(sim, "deltaT", &SimulationConstants::deltaT);
			field(sim, "maxSimulationSteps", &SimulationConstants::maxSimulationSteps);

			const auto& dim = j.at("DimensionConstants");
			field(dim, "xSize", &DimensionConstants::xSize);
			field(dim, "dx", &DimensionConstants::dx);

			const auto& noise = j.at("NoiseConstants");
			field(noise, "amplitude", &NoiseConstants::amplitude);

			const auto& gs = j.at("GaussStimulusConstants");
			field(gs, "width", &GaussStimulusConstants::width);
			field(gs, "amplitude", &GaussStimulusConstants::amplitude);
			field(gs, "circularity", &GaussStimulusConstants::circularity);
			field(gs, "normalization", &GaussStimulusConstants::normalization);

			const auto& nf = j.at("NeuralFieldConstants");
			field(nf, "tau", &NeuralFieldConstants::tau);
			field(nf, "restingLevel", &NeuralFieldConstants::restingLevel);
			{
				// SigmoidFunction declares a copy constructor, which implicitly
				// deletes its copy assignment, so the object cannot be replaced
				// wholesale. Its two parameters are public members and are the
				// only state its constructor sets, so they are assigned in place.
				const auto& activation = nf.at("activationFunction");
				field(activation, "xShift", &NeuralFieldConstants::activationFunction.x_shift);
				field(activation, "steepness", &NeuralFieldConstants::activationFunction.steepness);
			}
			field(nf, "tauMinVal", &NeuralFieldConstants::tauMinVal);
			field(nf, "tauMaxVal", &NeuralFieldConstants::tauMaxVal);
			field(nf, "tauStep", &NeuralFieldConstants::tauStep);
			field(nf, "restingLevelMinVal", &NeuralFieldConstants::restingLevelMinVal);
			field(nf, "restingLevelMaxVal", &NeuralFieldConstants::restingLevelMaxVal);
			field(nf, "restingLevelStep", &NeuralFieldConstants::restingLevelStep);

			const auto& kernel = j.at("KernelConstants");
			field(kernel, "circularity", &KernelConstants::circularity);
			field(kernel, "normalization", &KernelConstants::normalization);

			const auto& gk = j.at("GaussKernelConstants");
			field(gk, "width", &GaussKernelConstants::width);
			field(gk, "amplitude", &GaussKernelConstants::amplitude);
			field(gk, "amplitudeGlobal", &GaussKernelConstants::amplitudeGlobal);
			field(gk, "widthMinVal", &GaussKernelConstants::widthMinVal);
			field(gk, "widthMaxVal", &GaussKernelConstants::widthMaxVal);
			field(gk, "widthStep", &GaussKernelConstants::widthStep);
			field(gk, "ampMinVal", &GaussKernelConstants::ampMinVal);
			field(gk, "ampMaxVal", &GaussKernelConstants::ampMaxVal);
			field(gk, "ampStep", &GaussKernelConstants::ampStep);
			field(gk, "ampGlobalMinVal", &GaussKernelConstants::ampGlobalMinVal);
			field(gk, "ampGlobalMaxVal", &GaussKernelConstants::ampGlobalMaxVal);
			field(gk, "ampGlobalStep", &GaussKernelConstants::ampGlobalStep);

			const auto& mhk = j.at("MexicanHatKernelConstants");
			field(mhk, "widthExc", &MexicanHatKernelConstants::widthExc);
			field(mhk, "widthInh", &MexicanHatKernelConstants::widthInh);
			field(mhk, "amplitudeExc", &MexicanHatKernelConstants::amplitudeExc);
			field(mhk, "amplitudeInh", &MexicanHatKernelConstants::amplitudeInh);
			field(mhk, "amplitudeGlobal", &MexicanHatKernelConstants::amplitudeGlobal);
			field(mhk, "widthExcMinVal", &MexicanHatKernelConstants::widthExcMinVal);
			field(mhk, "widthExcMaxVal", &MexicanHatKernelConstants::widthExcMaxVal);
			field(mhk, "widthExcStep", &MexicanHatKernelConstants::widthExcStep);
			field(mhk, "widthInhMinVal", &MexicanHatKernelConstants::widthInhMinVal);
			field(mhk, "widthInhMaxVal", &MexicanHatKernelConstants::widthInhMaxVal);
			field(mhk, "widthInhStep", &MexicanHatKernelConstants::widthInhStep);
			field(mhk, "ampExcMinVal", &MexicanHatKernelConstants::ampExcMinVal);
			field(mhk, "ampExcMaxVal", &MexicanHatKernelConstants::ampExcMaxVal);
			field(mhk, "ampExcStep", &MexicanHatKernelConstants::ampExcStep);
			field(mhk, "ampInhMinVal", &MexicanHatKernelConstants::ampInhMinVal);
			field(mhk, "ampInhMaxVal", &MexicanHatKernelConstants::ampInhMaxVal);
			field(mhk, "ampInhStep", &MexicanHatKernelConstants::ampInhStep);
			field(mhk, "ampGlobMin", &MexicanHatKernelConstants::ampGlobMin);
			field(mhk, "ampGlobMax", &MexicanHatKernelConstants::ampGlobMax);
			field(mhk, "ampGlobStep", &MexicanHatKernelConstants::ampGlobStep);

			const auto& cc = j.at("CompatibilityCoefficients");
			field(cc, "compatibilityThreshold", &CompatibilityCoefficients::compatibilityThreshold);
			field(cc, "excessGenesCompatibilityWeight", &CompatibilityCoefficients::excessGenesCompatibilityWeight);
			field(cc, "disjointGenesCompatibilityWeight", &CompatibilityCoefficients::disjointGenesCompatibilityWeight);
			field(cc, "averageConnectionDifferenceCompatibilityWeight", &CompatibilityCoefficients::averageConnectionDifferenceCompatibilityWeight);
			field(cc, "amplitudeDifferenceCoefficient", &CompatibilityCoefficients::amplitudeDifferenceCoefficient);
			field(cc, "widthDifferenceCoefficient", &CompatibilityCoefficients::widthDifferenceCoefficient);

			const auto& gm = j.at("GenomeMutationConstants");
			field(gm, "toggleConnectionGeneProbability", &GenomeMutationConstants::toggleConnectionGeneProbability);
			field(gm, "addFieldGeneProbability", &GenomeMutationConstants::addFieldGeneProbability);
			field(gm, "addConnectionGeneProbability", &GenomeMutationConstants::addConnectionGeneProbability);
			field(gm, "mutateFieldGenesPerGenomeProbability", &GenomeMutationConstants::mutateFieldGenesPerGenomeProbability);
			field(gm, "mutateConnectionGenesProbability", &GenomeMutationConstants::mutateConnectionGenesProbability);
			field(gm, "mutateFieldGenePerGeneProbability", &GenomeMutationConstants::mutateFieldGenePerGeneProbability);
			field(gm, "mutateConnectionGeneProbability", &GenomeMutationConstants::mutateConnectionGeneProbability);
			field(gm, "checkForDuplicateConnectionGenesInGenome", &GenomeMutationConstants::checkForDuplicateConnectionGenesInGenome);

			const auto& fg = j.at("FieldGeneConstants");
			field(fg, "variableParameters", &FieldGeneConstants::variableParameters);
			field(fg, "gaussKernelProbability", &FieldGeneConstants::gaussKernelProbability);
			field(fg, "mexicanHatKernelProbability", &FieldGeneConstants::mexicanHatKernelProbability);
			field(fg, "mutateFieldGeneKernelProbability", &FieldGeneConstants::mutateFieldGeneKernelProbability);
			field(fg, "mutateFieldGeneKernelTypeProbability", &FieldGeneConstants::mutateFieldGeneKernelTypeProbability);
			field(fg, "mutateFieldGeneNeuralFieldProbability", &FieldGeneConstants::mutateFieldGeneNeuralFieldProbability);
			field(fg, "mutateFieldGeneGaussKernelAmplitudeProbability", &FieldGeneConstants::mutateFieldGeneGaussKernelAmplitudeProbability);
			field(fg, "mutateFieldGeneGaussKernelWidthProbability", &FieldGeneConstants::mutateFieldGeneGaussKernelWidthProbability);
			field(fg, "mutateFieldGeneGaussKernelGlobalAmplitudeProbability", &FieldGeneConstants::mutateFieldGeneGaussKernelGlobalAmplitudeProbability);
			field(fg, "mutateFieldGeneMexicanHatKernelAmplitudeExcProbability", &FieldGeneConstants::mutateFieldGeneMexicanHatKernelAmplitudeExcProbability);
			field(fg, "mutateFieldGeneMexicanHatKernelAmplitudeInhProbability", &FieldGeneConstants::mutateFieldGeneMexicanHatKernelAmplitudeInhProbability);
			field(fg, "mutateFieldGeneMexicanHatKernelWidthExcProbability", &FieldGeneConstants::mutateFieldGeneMexicanHatKernelWidthExcProbability);
			field(fg, "mutateFieldGeneMexicanHatKernelWidthInhProbability", &FieldGeneConstants::mutateFieldGeneMexicanHatKernelWidthInhProbability);
			field(fg, "mutateFieldGeneMexicanHatKernelGlobalAmplitudeProbability", &FieldGeneConstants::mutateFieldGeneMexicanHatKernelGlobalAmplitudeProbability);
			field(fg, "mutateFieldGeneNeuralFieldParametersProbability", &FieldGeneConstants::mutateFieldGeneNeuralFieldParametersProbability);
			field(fg, "mutateFieldGeneNeuralFieldGenerateRandomParametersProbability", &FieldGeneConstants::mutateFieldGeneNeuralFieldGenerateRandomParametersProbability);
			field(fg, "mutateFieldGeneNeuralFieldParametersTauProbability", &FieldGeneConstants::mutateFieldGeneNeuralFieldParametersTauProbability);
			field(fg, "mutateFieldGeneNeuralFieldParametersRestingLevelProbability", &FieldGeneConstants::mutateFieldGeneNeuralFieldParametersRestingLevelProbability);

			const auto& cg = j.at("ConnectionGeneConstants");
			field(cg, "gaussKernelProbability", &ConnectionGeneConstants::gaussKernelProbability);
			field(cg, "mexicanHatKernelProbability", &ConnectionGeneConstants::mexicanHatKernelProbability);
			field(cg, "mutateConnectionGeneKernelProbability", &ConnectionGeneConstants::mutateConnectionGeneKernelProbability);
			field(cg, "mutateConnectionGeneKernelTypeProbability", &ConnectionGeneConstants::mutateConnectionGeneKernelTypeProbability);
			field(cg, "mutateConnectionGeneConnectionSignalProbability", &ConnectionGeneConstants::mutateConnectionGeneConnectionSignalProbability);
			field(cg, "mutateConnectionGeneGaussKernelAmplitudeProbability", &ConnectionGeneConstants::mutateConnectionGeneGaussKernelAmplitudeProbability);
			field(cg, "mutateConnectionGeneGaussKernelWidthProbability", &ConnectionGeneConstants::mutateConnectionGeneGaussKernelWidthProbability);
			field(cg, "mutateConnectionGeneGaussKernelGlobalAmplitudeProbability", &ConnectionGeneConstants::mutateConnectionGeneGaussKernelGlobalAmplitudeProbability);
			field(cg, "mutateConnectionGeneMexicanHatKernelAmplitudeExcProbability", &ConnectionGeneConstants::mutateConnectionGeneMexicanHatKernelAmplitudeExcProbability);
			field(cg, "mutateConnectionGeneMexicanHatKernelAmplitudeInhProbability", &ConnectionGeneConstants::mutateConnectionGeneMexicanHatKernelAmplitudeInhProbability);
			field(cg, "mutateConnectionGeneMexicanHatKernelWidthExcProbability", &ConnectionGeneConstants::mutateConnectionGeneMexicanHatKernelWidthExcProbability);
			field(cg, "mutateConnectionGeneMexicanHatKernelWidthInhProbability", &ConnectionGeneConstants::mutateConnectionGeneMexicanHatKernelWidthInhProbability);
			field(cg, "mutateConnectionGeneMexicanHatKernelGlobalAmplitudeProbability", &ConnectionGeneConstants::mutateConnectionGeneMexicanHatKernelGlobalAmplitudeProbability);

			const auto& sc = j.at("SolutionConstants");
			field(sc, "minInitialInputGenes", &SolutionConstants::minInitialInputGenes);
			field(sc, "minInitialOutputGenes", &SolutionConstants::minInitialOutputGenes);
			field(sc, "fitnessWeights", &SolutionConstants::fitnessWeights);
			field(sc, "populationSize", &SolutionConstants::populationSize);
			field(sc, "numberGenerations", &SolutionConstants::numberGenerations);
			field(sc, "numberRuns", &SolutionConstants::numberRuns);
			field(sc, "targetFitness", &SolutionConstants::targetFitness);

			// The reference config carries the full "no ablation" baseline, and a
			// preset in config/ablations/ is merged over it. The reference counts
			// are read first: seedHiddenFieldsMin/Max may name one of them instead
			// of giving a number, and a solution config may itself have overridden
			// what those counts are.
			const auto& ac = j.at("AblationConstants");
			field(ac, "referenceHiddenFieldsMin", &AblationConstants::referenceHiddenFieldsMin);
			field(ac, "referenceHiddenFieldsMax", &AblationConstants::referenceHiddenFieldsMax);
			field(ac, "label", &AblationConstants::label);
			field(ac, "disableAddFieldGene", &AblationConstants::disableAddFieldGene);
			field(ac, "disableAddConnectionGene", &AblationConstants::disableAddConnectionGene);
			field(ac, "disableToggleConnectionGene", &AblationConstants::disableToggleConnectionGene);
			field(ac, "seedAllLegalConnections", &AblationConstants::seedAllLegalConnections);
			field(ac, "seedRandomHiddenFields", &AblationConstants::seedRandomHiddenFields);
			hiddenFieldBound(ac, "seedHiddenFieldsMin", &AblationConstants::seedHiddenFieldsMin);
			hiddenFieldBound(ac, "seedHiddenFieldsMax", &AblationConstants::seedHiddenFieldsMax);
			field(ac, "seedRandomConnections", &AblationConstants::seedRandomConnections);
			field(ac, "disableSpeciation", &AblationConstants::disableSpeciation);
			field(ac, "disableCrossover", &AblationConstants::disableCrossover);

			const auto& pc = j.at("PopulationConstants");
			field(pc, "pruneRatio", &PopulationConstants::pruneRatio);
			field(pc, "generationsWithoutImprovementThresholdInPopulation", &PopulationConstants::generationsWithoutImprovementThresholdInPopulation);
			field(pc, "generationsWithoutImprovementThresholdInSpecies", &PopulationConstants::generationsWithoutImprovementThresholdInSpecies);
			field(pc, "elitism", &PopulationConstants::elitism);
			field(pc, "elitismFitnessEpsilon", &PopulationConstants::elitismFitnessEpsilon);
			field(pc, "logSolutions", &PopulationConstants::logSolutions);
			field(pc, "logOverview", &PopulationConstants::logOverview);
			field(pc, "logSpecies", &PopulationConstants::logSpecies);
			field(pc, "saveOverview", &PopulationConstants::saveOverview);
			field(pc, "savePerGenerationOverview", &PopulationConstants::savePerGenerationOverview);
			field(pc, "saveChampions", &PopulationConstants::saveChampions);
			field(pc, "saveBestSolutions", &PopulationConstants::saveBestSolutions);
			field(pc, "saveSolutions", &PopulationConstants::saveSolutions);
			field(pc, "saveSpecies", &PopulationConstants::saveSpecies);
		}
		catch (const nlohmann::json::exception& e)
		{
			throw std::runtime_error("ConfigLoader: failed to load '" + path + "': " + e.what());
		}
	}

}
