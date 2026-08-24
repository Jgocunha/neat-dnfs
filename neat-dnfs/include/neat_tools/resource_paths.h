#pragma once

#include <filesystem>
#include <vector>

namespace neat_dnfs::paths
{
	/// @brief The directory that holds config/ and templates/.
	///
	/// Resolved once, on first call, from the first of these that holds
	/// config/neat_dnfs.json:
	///   1. $NEAT_DNFS_ROOT, if set (the only candidate considered when it is,
	///      so a wrong value is an error rather than a silent fallback)
	///   2. the directory of the running executable  -- portable layout
	///   3. <executable directory>/../share/neat-dnfs -- installed layout
	///   4. PROJECT_DIR, the source tree this binary was configured from
	/// Rule 4 is why a build tree behaves exactly as it did before this existed;
	/// rules 2 and 3 are what let a downloaded release run at all.
	/// @throws std::runtime_error naming every candidate tried, if none holds it.
	const std::filesystem::path& resourceRoot();

	/// @brief The directory under which data/<solution>/<timestamp>/ is written.
	/// $NEAT_DNFS_DATA_DIR if set, else resourceRoot() when that is the source
	/// tree (results keep landing in the repo's data/ folder, as before), else
	/// the current working directory -- an installed package may sit somewhere
	/// the user cannot write to.
	const std::filesystem::path& dataRoot();

	/// @brief Returns the first of @p candidates that holds config/neat_dnfs.json,
	/// skipping empty ones. Split out from resourceRoot() so it can be tested
	/// without the process-wide caching that resolution needs.
	/// @throws std::runtime_error naming every candidate, if none matches.
	std::filesystem::path selectResourceRoot(const std::vector<std::filesystem::path>& candidates);
}
