#include "neat_tools/machine_info.h"

#include <thread>

#ifdef _WIN32
	#include <windows.h>
#elif defined(__APPLE__)
	#include <sys/sysctl.h>
	#include <sys/types.h>
#else
	#include <fstream>
	#include <unistd.h>
#endif

namespace neat_dnfs::tools::machine_info
{
	std::string operatingSystem()
	{
#ifdef _WIN32
		return "Windows";
#elif defined(__APPLE__)
		return "macOS";
#else
		return "Linux";
#endif
	}

	unsigned int logicalCoreCount()
	{
		return std::thread::hardware_concurrency();
	}

	std::string cpuModel()
	{
#ifdef _WIN32
		HKEY key;
		if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
			"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", 0, KEY_READ, &key) == ERROR_SUCCESS)
		{
			char buffer[256]{};
			DWORD size = sizeof(buffer);
			const bool ok = RegQueryValueExA(key, "ProcessorNameString", nullptr, nullptr,
				reinterpret_cast<LPBYTE>(buffer), &size) == ERROR_SUCCESS;
			RegCloseKey(key);
			if (ok)
			{
				return std::string(buffer);
			}
		}
		return "unknown";
#elif defined(__APPLE__)
		char buffer[256]{};
		size_t size = sizeof(buffer);
		if (sysctlbyname("machdep.cpu.brand_string", buffer, &size, nullptr, 0) == 0)
		{
			return std::string(buffer);
		}
		return "unknown";
#else
		std::ifstream cpuinfo("/proc/cpuinfo");
		std::string line;
		while (std::getline(cpuinfo, line))
		{
			if (line.rfind("model name", 0) == 0)
			{
				const auto colon = line.find(':');
				if (colon != std::string::npos)
				{
					return line.substr(colon + 2);
				}
			}
		}
		return "unknown";
#endif
	}

	std::uint64_t totalRamBytes()
	{
#ifdef _WIN32
		MEMORYSTATUSEX status{};
		status.dwLength = sizeof(status);
		if (GlobalMemoryStatusEx(&status))
		{
			return status.ullTotalPhys;
		}
		return 0;
#elif defined(__APPLE__)
		std::uint64_t memSize = 0;
		size_t size = sizeof(memSize);
		if (sysctlbyname("hw.memsize", &memSize, &size, nullptr, 0) == 0)
		{
			return memSize;
		}
		return 0;
#else
		const long pages = sysconf(_SC_PHYS_PAGES);
		const long pageSize = sysconf(_SC_PAGE_SIZE);
		if (pages > 0 && pageSize > 0)
		{
			return static_cast<std::uint64_t>(pages) * static_cast<std::uint64_t>(pageSize);
		}
		return 0;
#endif
	}
}
