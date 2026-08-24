#pragma once

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace neat_dnfs
{
	/// @brief Loads the tunable values in constants.h from JSON at startup.
	///
	/// Every field in constants.h (except the compile-time name prefixes, and
	/// except AblationConstants' no-op flags) is declared without an
	/// initializer, so nothing here has a compiled-in fallback: a missing file
	/// or a missing key is a hard error rather than a silent zero. Callers are
	/// the three apps in apps/ (via --config) and tests/entry.cpp.
	///
	/// config/neat_dnfs.json is the complete reference set. A per-experiment
	/// config under config/solutions/ is a *sparse* override of it: the two are
	/// deep-merged (RFC 7386) and the merged result is then validated strictly,
	/// so an override only needs to name the keys that actually differ while
	/// the "every key present" guarantee still holds.
	struct ConfigLoader
	{
		/// @throws std::runtime_error if @p path cannot be opened or is not valid JSON.
		static nlohmann::json loadJsonFile(const std::string& path);

		/// @brief Assigns j.at(key) into *target.
		/// @throws nlohmann::json::out_of_range if @p key is absent from @p j.
		/// @throws nlohmann::json::type_error if the value cannot convert to T.
		template <typename T>
		static void field(const nlohmann::json& j, const char* key, T* target)
		{
			*target = j.at(key).get<T>();
		}

		/// @brief Reads a fitness-weight array, checking it has exactly
		/// @p expected entries. Solutions index these positionally, so a
		/// wrong-length array would silently mis-weight the fitness (or read
		/// out of bounds) rather than fail.
		/// @throws std::runtime_error if the length does not match.
		static void weights(const nlohmann::json& j, const char* key,
			std::vector<double>* target, size_t expected);

		/// @brief Populates every tunable field of the constants.h structs from
		/// @p path alone, with no experiment override applied.
		/// @throws std::runtime_error if the file is missing, is not valid JSON,
		/// or omits any expected struct or key.
		static void loadGlobalConfig(const std::string& path);

		/// @brief Loads @p referencePath, deep-merges config/solutions/<slug>.json
		/// over it, and applies the result. Because the override can change values
		/// the rest of startup depends on (DimensionConstants::xSize feeds
		/// defaultTopologyFor(), AblationConstants' reference counts feed the
		/// ablation presets), this must run before topology construction and
		/// before any preset is applied.
		/// @throws std::runtime_error if either file is missing or malformed, or
		/// if the merged result omits any expected key.
		static void loadConfig(const std::string& referencePath, const std::string& slug);

		/// @brief Reads just SolutionConstants.fitnessWeights out of
		/// config/solutions/<slug>.json, for a Solution to copy into its own
		/// instance. Deliberately narrow: re-applying the whole merged config
		/// here would clobber globals that tests set by hand before constructing
		/// a solution.
		/// @throws std::runtime_error if the file is missing, the key is absent,
		/// or the array does not hold exactly @p expectedCount entries.
		static std::vector<double> loadFitnessWeights(const std::string& slug, size_t expectedCount);

		/// @brief Merges the ablation preset at @p path over the currently loaded
		/// reference+solution config and applies the result -- the third and last
		/// config layer, ahead of only the CLI flags. A preset file is sparse: it
		/// names the flags that condition changes, and optionally a run protocol
		/// to standardise population size / generations / runs across conditions.
		/// @throws std::runtime_error if the file is missing or malformed, or if
		/// the merged result omits any expected key.
		static void applyAblation(const std::string& path);

		/// @return PROJECT_DIR "/config/neat_dnfs.json" -- the config used when
		/// no --config flag is given.
		static std::string defaultGlobalConfigPath();

		/// @return PROJECT_DIR "/config/solutions/<slug>.json", the per-experiment
		/// override for that task.
		static std::string solutionConfigPath(const std::string& slug);

	private:
		/// @brief Strictly validates and applies an already-merged config object.
		/// Shared by every load path so they all run the same single validation
		/// pass. @p path is used only for error messages.
		static void applyConfig(const nlohmann::json& j, const std::string& path);

		/// The merged reference+solution config currently in effect. Retained so
		/// an ablation preset can be merged over it without re-reading the files
		/// underneath, and so applying a preset yields the full three-layer result
		/// rather than only the preset's own keys.
		inline static nlohmann::json activeConfig;
		inline static std::string activeConfigPath;
	};
}
