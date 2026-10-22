// This is a personal academic project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: https://pvs-studio.com

#include <exception>
#include <fstream>
#include <iostream>
#include <dnf_composer/simulation/simulation_file_manager.h>
#include <dnf_composer/tools/logger.h>

#include "neat_tools/ablation_presets.h"
#include "neat_tools/config_loader.h"
#include "neat_tools/logger.h"
#include "neat_tools/solution_registry.h"
#include "solutions/serial_order_profile.h"

// Records what a saved design's input and output do through the serial-order trial, as the serial-order-profile task's
// evaluation sees it, and writes it as JSON: config/references/serial-order-profile.json is this tool's recording of the
// Gauss-only reference design.
int main(int argc, char* argv[])
{
	try
	{
		dnf_composer::tools::logger::Logger::setMinLogLevel(dnf_composer::tools::logger::LogLevel::ERROR);
		using namespace neat_dnfs;

		const CliOptions opts = parseCliOptions(argc, argv);
		if (opts.helpRequested)
		{
			printUsage(std::cout, "neat-dnfs-record-profile");
			return 0;
		}
		if (!opts.templateFile || !opts.outputFile)
		{
			std::cerr << "neat-dnfs-record-profile needs --template PATH, the design to record, and --out PATH.\n";
			return 1;
		}

		ConfigLoader::loadConfig(opts.config.value_or(ConfigLoader::defaultGlobalConfigPath()), "serial-order-profile");
		if (opts.ablation && !AblationPresets::applyByName(*opts.ablation))
		{
			std::cerr << "Unknown ablation '" << *opts.ablation << "'.\n";
			return 1;
		}

		const auto design = std::make_shared<dnf_composer::Simulation>();
		const dnf_composer::SimulationFileManager sfm(design, *opts.templateFile);
		sfm.loadElementsFromJson();

		const dnf_composer::element::ElementDimensions dimensions{ DimensionConstants::xSize, DimensionConstants::dx };
		const SolutionTopology topology({ { FieldGeneType::INPUT, dimensions }, { FieldGeneType::OUTPUT, dimensions } });
		SerialOrderProfile solution(topology, *design);
		const ActivationProfile profile = solution.recordProfile();

		std::ofstream out(*opts.outputFile);
		if (!out)
		{
			std::cerr << "Cannot write " << *opts.outputFile << ".\n";
			return 1;
		}
		out << profile.toJson().dump();
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
