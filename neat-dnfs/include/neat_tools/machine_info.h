#pragma once

#include <cstdint>
#include <string>

namespace neat_dnfs::tools::machine_info
{
	/// @brief The operating system this process is running on.
	/// @return "Windows", "macOS", or "Linux".
	std::string operatingSystem();

	/// @brief Number of logical CPU cores available to this process.
	/// @return The core count, or 0 if it cannot be determined.
	unsigned int logicalCoreCount();

	/// @brief The CPU model/brand string reported by the operating system.
	/// @return The model string, or "unknown" if it cannot be determined.
	std::string cpuModel();

	/// @brief Total physical RAM installed on this machine.
	/// @return The size in bytes, or 0 if it cannot be determined.
	std::uint64_t totalRamBytes();
}
