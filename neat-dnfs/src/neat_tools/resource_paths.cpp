#include "neat_tools/resource_paths.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <cstdint>
#include <cstring>
#include <mach-o/dyld.h>
#endif

namespace neat_dnfs::paths
{
	namespace
	{
		// The one file whose presence identifies a resource root. It is the
		// complete reference config, so a directory without it cannot serve a
		// run regardless of what else it contains.
		//
		// A string literal rather than a std::filesystem::path object on purpose:
		// tests/entry.cpp resolves the root from a namespace-scope initializer, and
		// a path object here would be read across translation units before its own
		// constructor had run.
		constexpr const char* referenceConfig = "config/neat_dnfs.json";

		std::string environmentVariable(const char* name)
		{
#if defined(_WIN32)
			// getenv() is a deprecation warning under MSVC; this mirrors the
			// localtime_s/localtime_r split used elsewhere in the codebase.
			char* value = nullptr;
			size_t length = 0;
			if (_dupenv_s(&value, &length, name) != 0 || value == nullptr)
			{
				return {};
			}
			std::string result(value);
			std::free(value);
			return result;
#else
			const char* value = std::getenv(name);
			return value != nullptr ? std::string(value) : std::string{};
#endif
		}

		// Returns an empty path if the platform call fails, which
		// selectResourceRoot() then skips like any other non-matching candidate.
		std::filesystem::path executableDirectory()
		{
#if defined(_WIN32)
			std::wstring buffer(MAX_PATH, L'\0');
			for (;;)
			{
				const DWORD written = GetModuleFileNameW(nullptr, buffer.data(),
					static_cast<DWORD>(buffer.size()));
				if (written == 0)
				{
					return {};
				}
				if (written < buffer.size())
				{
					buffer.resize(written);
					break;
				}
				buffer.resize(buffer.size() * 2);
			}
			return std::filesystem::path(buffer).parent_path();
#elif defined(__APPLE__)
			uint32_t size = 0;
			_NSGetExecutablePath(nullptr, &size); // sizes the buffer, always "fails"
			std::string buffer(size, '\0');
			if (_NSGetExecutablePath(buffer.data(), &size) != 0)
			{
				return {};
			}
			buffer.resize(std::strlen(buffer.c_str()));
			// The returned path may contain symlinks and ".." segments.
			std::error_code ec;
			const std::filesystem::path resolved = std::filesystem::canonical(buffer, ec);
			return ec ? std::filesystem::path(buffer).parent_path() : resolved.parent_path();
#else
			std::error_code ec;
			const std::filesystem::path exe = std::filesystem::read_symlink("/proc/self/exe", ec);
			return ec ? std::filesystem::path{} : exe.parent_path();
#endif
		}
	}

	std::filesystem::path selectResourceRoot(const std::vector<std::filesystem::path>& candidates)
	{
		std::string tried;
		for (const auto& candidate : candidates)
		{
			if (candidate.empty())
			{
				continue;
			}
			std::error_code ec;
			if (std::filesystem::exists(candidate / referenceConfig, ec) && !ec)
			{
				return std::filesystem::path(candidate).lexically_normal();
			}
			tried += "\n  " + candidate.generic_string();
		}
		throw std::runtime_error(std::string("resource root not found: no directory below holds ")
			+ referenceConfig
			+ ".\nSet NEAT_DNFS_ROOT to the directory containing config/ and templates/."
			"\nTried:" + tried);
	}

	const std::filesystem::path& resourceRoot()
	{
		static const std::filesystem::path root = []
		{
			const std::string rootOverride = environmentVariable("NEAT_DNFS_ROOT");
			if (!rootOverride.empty())
			{
				return selectResourceRoot({ rootOverride });
			}
			const std::filesystem::path exeDirectory = executableDirectory();
			return selectResourceRoot({
				exeDirectory,
				exeDirectory.empty() ? exeDirectory : exeDirectory / ".." / "share" / "neat-dnfs",
				PROJECT_DIR });
		}();
		return root;
	}

	const std::filesystem::path& dataRoot()
	{
		static const std::filesystem::path root = []
		{
			const std::string dataOverride = environmentVariable("NEAT_DNFS_DATA_DIR");
			if (!dataOverride.empty())
			{
				return std::filesystem::path(dataOverride);
			}
			std::error_code ec;
			if (std::filesystem::equivalent(resourceRoot(), PROJECT_DIR, ec) && !ec)
			{
				return resourceRoot();
			}
			return std::filesystem::current_path();
		}();
		return root;
	}
}
