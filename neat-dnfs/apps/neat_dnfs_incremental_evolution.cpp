 // This is a personal academic project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: https://pvs-studio.com

#include <exception>
#include <iostream>
#include "dnf_composer/application/application.h"
#include <dnf_composer/simulation/simulation_file_manager.h>
#include <dnf_composer/tools/logger.h>

#include "neat_tools/config_loader.h"
#include "neat/population.h"
#include "neat_tools/ablation_presets.h"
#include "neat_tools/logger.h"
#include "neat_tools/key_listener.h"
#include "neat_tools/solution_registry.h"

int main(int argc, char* argv[])
{
	try
	{
		dnf_composer::tools::logger::Logger::setMinLogLevel(dnf_composer::tools::logger::LogLevel::ERROR);
		using namespace neat_dnfs;

		const CliOptions opts = parseCliOptions(argc, argv);
		if (opts.helpRequested)
		{
			printUsage(std::cout, "neat-dnfs-inc-evol");
			return 0;
		}
		if (opts.listRequested)
		{
			printTaskAndAblationList(std::cout);
			return 0;
		}

		const std::string taskSlug = opts.task.value_or("ior");
		const TaskEntry* task = findTask(taskSlug);
		if (task == nullptr)
		{
			std::cerr << "Unknown task '" << taskSlug << "'.\n";
			printTaskAndAblationList(std::cerr);
			return 1;
		}

		ConfigLoader::loadConfig(opts.config.value_or(ConfigLoader::defaultGlobalConfigPath()), std::string(task->slug));

		if (opts.ablation && !AblationPresets::applyByName(*opts.ablation))
		{
			std::cerr << "Unknown ablation '" << *opts.ablation << "'.\n";
			printTaskAndAblationList(std::cerr);
			return 1;
		}

		// load a previous solution
		const std::string templatePath = opts.templateFile.value_or(
			std::string(PROJECT_DIR) + "/templates/" + std::string(task->templateFile));
		const auto previousSolution = std::make_shared<dnf_composer::Simulation>();
		const dnf_composer::SimulationFileManager sfm(previousSolution, templatePath);
		sfm.loadElementsFromJson();
		const dnf_composer::Simulation& templateSolution = *previousSolution;

		const SolutionTopology topology = defaultTopologyFor(*task);

		const int numberRuns = opts.runs.value_or(SolutionConstants::numberRuns);
		const int populationSize = opts.populationSize.value_or(SolutionConstants::populationSize);
		const int numberGenerations = opts.numGenerations.value_or(SolutionConstants::numberGenerations);
		const double targetFitness = opts.targetFitness.value_or(SolutionConstants::targetFitness);

		for (int i = 0; i < numberRuns; i++)
		{
			const PopulationParameters parameters{ populationSize, numberGenerations, targetFitness };
			Population population{ parameters, task->makeFromTemplate(topology, templateSolution) };

			population.initialize();
			KeyListener keyListener{ population };
			population.evolve();
		}

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
