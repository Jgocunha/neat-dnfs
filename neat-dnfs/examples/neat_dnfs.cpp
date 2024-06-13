// This is a personal academic project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: https://pvs-studio.com

#include "../../../../Program Files/dynamic-neural-field-composer/include/tools/logger.h"
#include "application/application.h"

int main(int argc, char* argv[])
{
	try
	{
		using namespace neat_dnfs;

		dnf_composer::tools::logger::Logger::setMinLogLevel(dnf_composer::tools::logger::LogLevel::WARNING);

		const Application app;
		app.initialize();

		do { app.render(); Sleep(10); } while (!app.hasCloseBeenRequested());

		app.shutdown();

		return 0;
	}
	catch (const dnf_composer::Exception& ex)
	{
		log(neat_dnfs::tools::logger::LogLevel::FATAL, "Exception caught: " + std::string(ex.what()) + ".");
		return static_cast<int>(ex.getErrorCode());
	}
	catch (const std::exception& ex)
	{
		log(neat_dnfs::tools::logger::LogLevel::FATAL, "Exception caught: " + std::string(ex.what()) + ".");
		return 1;
	}
	catch (...)
	{
		log(neat_dnfs::tools::logger::LogLevel::FATAL, "Unknown exception occurred.");
		return 1;
	}
}
