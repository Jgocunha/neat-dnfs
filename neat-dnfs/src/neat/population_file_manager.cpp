#include "neat/population_file_manager.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <numeric>

#include "neat/population.h"
#include "neat_tools/build_info.h"
#include "neat_tools/logger.h"
#include "neat_tools/machine_info.h"
#include "neat_tools/resource_paths.h"

#ifdef NEAT_DNFS_PROFILE
#include "neat_tools/profiler.h"
#include <string_view>
#endif

namespace neat_dnfs
{
	namespace
	{
		std::string solutionIdentifier(int id, int generation, int speciesId, double fitness)
		{
			return std::format("solution {} generation {} species {} fitness {:f}", id, generation, speciesId, fitness);
		}
	}

	PopulationFileManager::PopulationFileManager(const Population& population)
		: population(&population)
	{}

	void PopulationFileManager::saveOverviewForGeneration() const
	{
		if (PopulationConstants::saveOverview)
		{
			savePerGenerationOverview();
		}
		if (PopulationConstants::saveStructuredOverview)
		{
			savePerGenerationOverviewJson();
		}
	}

	void PopulationFileManager::savePerGenerationData() const
	{
		if (PopulationConstants::saveBestSolutions)
		{
			saveBestSolutionOfEachGeneration();
		}
		if (PopulationConstants::saveChampions)
		{
			saveChampionsOfEachGeneration();
		}
		if (PopulationConstants::saveSolutions)
		{
			saveAllSolutionsPerGeneration();
		}
		if (PopulationConstants::savePerGenerationOverview)
		{
			savePerGenerationStatistics();
		}
		if (PopulationConstants::saveSpecies)
		{
			savePerGenerationSpecies();
		}
	}

	void PopulationFileManager::saveEndOfRunData() const
	{
		if (PopulationConstants::saveSolutions && population->bestSolution != nullptr)
		{
			saveAllSolutionsWithFitnessAbove(population->bestSolution->getFitness() - 0.1);
		}
		if (PopulationConstants::saveOverview)
		{
			saveTimestampsAndDuration();
			saveRunMetadata();
		}
		if (PopulationConstants::saveChampions)
		{
			saveChampions();
		}
	}

	void PopulationFileManager::setFileDirectory()
	{
		using namespace dnf_composer;
		if (population->solutions.empty())
		{
			throw std::runtime_error("No solutions in population.");
		}

		const std::string solutionName = population->solutions[0]->getName() + AblationConstants::label;
		const auto now = std::time(nullptr);
		struct tm localTime{};
#ifdef _WIN32
		localtime_s(&localTime, &now);
#else
		localtime_r(&now, &localTime);
#endif
		std::array<char, 100> timeBuffer{};
		(void)std::strftime(timeBuffer.data(), timeBuffer.size(), "%Y-%m-%d %Hh%Mm%Ss", &localTime);

		fileDirectory = (paths::dataRoot() / "data" / solutionName / timeBuffer.data()).generic_string() + "/";
		std::filesystem::create_directories(fileDirectory); // Ensure directory exist
	}

	void PopulationFileManager::saveAllSolutionsWithFitnessAbove(const double fitness) const
	{
		using namespace dnf_composer;

		const std::string directoryPath = std::format("{}best_solutions/last_generation/", fileDirectory);
		std::filesystem::create_directories(directoryPath); // Ensure directory exist

		for (const auto& solution : population->solutions)
		{
			if (solution->getFitness() > fitness)
			{
				solution->buildPhenotype();
				solution->createPhenotypeEnvironment();
				auto simulation = solution->getPhenotype();
				solution->clearPhenotype();
				// save weights
				for (const auto& element : simulation.getElements())
				{
					if (element->getLabel() == element::ElementLabel::FIELD_COUPLING)
					{
						const auto fieldCoupling = std::dynamic_pointer_cast<element::FieldCoupling>(element);
						fieldCoupling->writeWeights();
					}
				}
				// save elements
				const std::string uniqueIdentifier = solutionIdentifier(solution->getId(),
					population->parameters.currentGeneration, solution->getSpeciesId(), solution->getFitness());
				simulation.setUniqueIdentifier(uniqueIdentifier);
				SimulationFileManager sfm(std::make_shared<Simulation>(simulation), directoryPath);
				sfm.saveElementsToJson();
			}
		}
	}

	void PopulationFileManager::saveChampions() const
	{
		using namespace dnf_composer;

		const std::string directoryPath = std::format("{}champions/last_generation/", fileDirectory);
		std::filesystem::create_directories(directoryPath); // Ensure directory exist

		if (population->champions.empty())
		{
			log(tools::logger::LogLevel::ERROR, "No champions to save.");
		}

		for (const auto& champion : population->champions)
		{
			if (champion == nullptr)
			{
				continue;
			}
			champion->buildPhenotype();
			champion->createPhenotypeEnvironment();
			auto simulation = champion->getPhenotype();
			champion->clearPhenotype();
			// save weights
			for (const auto& element : simulation.getElements())
			{
				if (element->getLabel() == element::ElementLabel::FIELD_COUPLING)
								{
					const auto fieldCoupling = std::dynamic_pointer_cast<element::FieldCoupling>(element);
					fieldCoupling->writeWeights();
				}
			}
			// save elements
			const std::string uniqueIdentifier = solutionIdentifier(champion->getId(),
				population->parameters.currentGeneration, champion->getSpeciesId(), champion->getFitness());
			simulation.setUniqueIdentifier(uniqueIdentifier);
			SimulationFileManager sfm(std::make_shared<Simulation>(simulation), directoryPath);
			sfm.saveElementsToJson();
		}
	}

	void PopulationFileManager::saveTimestampsAndDuration() const
	{
		const std::string directoryPath = std::format("{}/", fileDirectory);
		std::filesystem::create_directories(directoryPath); // Ensure directory exists

		std::ofstream logFile(std::format("{}evolution_timestamps.txt", directoryPath), std::ios::app);
		if (logFile.is_open())
		{
			// Convert steady_clock timestamps to system_clock timestamps
			const auto system_start = std::chrono::system_clock::now() +
				std::chrono::duration_cast<std::chrono::system_clock::duration>(
					population->statistics.start - std::chrono::steady_clock::now());

			const auto system_end = std::chrono::system_clock::now() +
				std::chrono::duration_cast<std::chrono::system_clock::duration>(
					population->statistics.end - std::chrono::steady_clock::now());

			// Convert to time_t for formatting
			const std::time_t start_time_t = std::chrono::system_clock::to_time_t(system_start);
			const std::time_t end_time_t = std::chrono::system_clock::to_time_t(system_end);

			struct tm startTm{};
			struct tm endTm{};
#ifdef _WIN32
			localtime_s(&startTm, &start_time_t);
			localtime_s(&endTm, &end_time_t);
#else
			localtime_r(&start_time_t, &startTm);
			localtime_r(&end_time_t, &endTm);
#endif

			// Log number of generations
			logFile << "Number of generations: " << population->parameters.currentGeneration << "\n";
			// Format and write timestamps
			logFile << "Evolution Start Time: " << std::put_time(&startTm, "%Y-%m-%d %H:%M:%S") << "\n";
			logFile << "Evolution End Time: " << std::put_time(&endTm, "%Y-%m-%d %H:%M:%S") << "\n";
			logFile << "Duration (seconds): " << population->statistics.duration << "\n";
			logFile << "Duration (minutes): " << population->statistics.duration / 60 << "\n";
			logFile << "Duration (hours): " << population->statistics.duration / 3600 << "\n";

			logFile.close();
		}
		else
		{
			tools::logger::log(tools::logger::LogLevel::ERROR, "Failed to open log file for timestamps.");
		}
	}

	void PopulationFileManager::saveRunMetadata() const
	{
		using namespace tools;

		nlohmann::json metadata;
		metadata["build"] = {
			{"compiler_id", std::string(build_info::compilerId)},
			{"compiler_version", std::string(build_info::compilerVersion)},
			{"cmake_version", std::string(build_info::cmakeVersion)},
			{"neat_dnfs_version", std::string(build_info::neatDnfsVersion)},
			{"sanitizer", std::string(build_info::sanitizer)},
			{"build_type", std::string(build_info::buildType)},
			{"git_sha", std::string(build_info::gitSha)},
			{"git_dirty", build_info::gitDirty}
		};
		metadata["dependencies"] = {
			{"imgui_platform_kit_sha", std::string(build_info::imguiPlatformKitSha)},
			{"dynamic_neural_field_composer_sha", std::string(build_info::dynamicNeuralFieldComposerSha)},
			{"vcpkg_packages", std::string(build_info::vcpkgPackageList)}
		};
		metadata["machine"] = {
			{"os", machine_info::operatingSystem()},
			{"cpu_model", machine_info::cpuModel()},
			{"logical_cores", machine_info::logicalCoreCount()},
			{"total_ram_bytes", machine_info::totalRamBytes()}
		};
		metadata["run_parameters"] = {
			{"parallel_evolution", population->parameters.parallelEvolution}
		};

		const std::string directoryPath = std::format("{}/", fileDirectory);
		std::filesystem::create_directories(directoryPath); // Ensure directory exists

		std::ofstream metadataFile(std::format("{}run_metadata.json", directoryPath));
		if (metadataFile.is_open())
		{
			metadataFile << metadata.dump(2);
		}
		else
		{
			logger::log(logger::LogLevel::ERROR, "Failed to open log file for run metadata.");
		}
	}

	void PopulationFileManager::saveAllSolutionsPerGeneration() const
	{
		using namespace dnf_composer;

		for (const auto& solution : population->solutions)
		{
			const std::string directoryPath = std::format("{}solutions/gen {}/", fileDirectory, population->parameters.currentGeneration);
			std::filesystem::create_directories(directoryPath); // Ensure directory exists

			solution->buildPhenotype();
			solution->createPhenotypeEnvironment();
			auto simulation = solution->getPhenotype();
			solution->clearPhenotype();

			const std::string uniqueIdentifier = solutionIdentifier(solution->getId(),
				population->parameters.currentGeneration, solution->getSpeciesId(), solution->getFitness());
			simulation.setUniqueIdentifier(uniqueIdentifier);
			SimulationFileManager sfm(std::make_shared<Simulation>(simulation), directoryPath);
			sfm.saveElementsToJson();
		}
	}

	void PopulationFileManager::savePerGenerationOverview() const
	{
		const std::string directoryPath = std::format("{}/", fileDirectory);
		std::filesystem::create_directories(directoryPath); // Ensure directory exists

		std::ofstream logFile(std::format("{}per_generation_overview.txt", directoryPath), std::ios::app);
		if (logFile.is_open())
		{
			logFile << std::format(
				"Current generation: {} Number of solutions: {} Number of species: {} "
				"Number of active species: {} Has fitness improved: {} "
				"Number of generations without improvement: {} Average fitness: {:.3f} "
				"Best fitness: {:.3f} Innovation number: {} Average genome size: {:.3f} "
				"Average connection genes: {:.3f} Average field genes: {:.3f} Best solution: [{}]\n",
				population->parameters.currentGeneration,
				population->solutions.size(),
				population->perGenStatistics.numberOfSpecies,
				population->perGenStatistics.numberOfActiveSpecies,
				population->hasFitnessImproved ? "yes" : "no",
				population->generationsWithoutImprovement,
				population->perGenStatistics.averageFitness,
				population->perGenStatistics.bestFitness,
				population->perGenStatistics.innovationNumber,
				population->perGenStatistics.averageGenomeSize,
				population->perGenStatistics.averageConnectionGenes,
				population->perGenStatistics.averageFieldGenes,
				population->bestSolution->toString());
			logFile.close();
		}
		else
		{
			tools::logger::log(tools::logger::LogLevel::ERROR,
				"Failed to open log file for field gene per generation statistics.");
		}
	}

	namespace
	{
		struct FitnessDistribution
		{
			double min{};
			double max{};
			double mean{};
			double median{};
			double stddev{};
			double q1{};
			double q3{};
		};

		double percentile(const std::vector<double>& sortedValues, const double fraction)
		{
			const double position = fraction * static_cast<double>(sortedValues.size() - 1);
			const auto lowerIndex = static_cast<size_t>(std::floor(position));
			const auto upperIndex = static_cast<size_t>(std::ceil(position));
			if (lowerIndex == upperIndex)
			{
				return sortedValues[lowerIndex];
			}
			const double weight = position - static_cast<double>(lowerIndex);
			return sortedValues[lowerIndex] * (1.0 - weight) + sortedValues[upperIndex] * weight;
		}

		FitnessDistribution computeFitnessDistribution(const std::vector<SolutionPtr>& solutions)
		{
			std::vector<double> fitnessValues;
			fitnessValues.reserve(solutions.size());
			for (const auto& solution : solutions)
			{
				fitnessValues.push_back(solution->getFitness());
			}
			std::ranges::sort(fitnessValues);

			FitnessDistribution distribution;
			if (fitnessValues.empty())
			{
				return distribution;
			}

			const double sum = std::accumulate(fitnessValues.begin(), fitnessValues.end(), 0.0);
			const double mean = sum / static_cast<double>(fitnessValues.size());

			double squaredDeviationSum = 0.0;
			for (const double value : fitnessValues)
			{
				squaredDeviationSum += (value - mean) * (value - mean);
			}

			distribution.min = fitnessValues.front();
			distribution.max = fitnessValues.back();
			distribution.mean = mean;
			distribution.median = percentile(fitnessValues, 0.5);
			distribution.stddev = std::sqrt(squaredDeviationSum / static_cast<double>(fitnessValues.size()));
			distribution.q1 = percentile(fitnessValues, 0.25);
			distribution.q3 = percentile(fitnessValues, 0.75);
			return distribution;
		}

		nlohmann::json toJson(const FitnessDistribution& distribution)
		{
			return {
				{"min", distribution.min},
				{"max", distribution.max},
				{"mean", distribution.mean},
				{"median", distribution.median},
				{"stddev", distribution.stddev},
				{"q1", distribution.q1},
				{"q3", distribution.q3}
			};
		}
	}

	void PopulationFileManager::savePerGenerationOverviewJson() const
	{
		nlohmann::json record;
		record["generation"] = population->parameters.currentGeneration;
		record["numberOfSolutions"] = population->solutions.size();
		record["numberOfSpecies"] = population->perGenStatistics.numberOfSpecies;
		record["numberOfActiveSpecies"] = population->perGenStatistics.numberOfActiveSpecies;
		record["hasFitnessImproved"] = population->hasFitnessImproved;
		record["generationsWithoutImprovement"] = population->generationsWithoutImprovement;
		record["averageFitness"] = population->perGenStatistics.averageFitness;
		record["bestFitness"] = population->perGenStatistics.bestFitness;
		record["innovationNumber"] = population->perGenStatistics.innovationNumber;
		record["averageGenomeSize"] = population->perGenStatistics.averageGenomeSize;
		record["averageConnectionGenes"] = population->perGenStatistics.averageConnectionGenes;
		record["averageFieldGenes"] = population->perGenStatistics.averageFieldGenes;
		record["fitnessDistribution"] = toJson(computeFitnessDistribution(population->solutions));

		nlohmann::json speciesArray = nlohmann::json::array();
		for (const auto& species : population->speciesList)
		{
			speciesArray.push_back({
				{"id", species->getId()},
				{"size", species->size()}
			});
		}
		record["species"] = speciesArray;

		if (population->bestSolution != nullptr)
		{
			const auto [parent1, parent2] = population->bestSolution->getParents();
			record["bestSolution"] = {
				{"id", population->bestSolution->getId()},
				{"fitness", population->bestSolution->getFitness()},
				{"parentIds", {parent1, parent2}},
				{"partialFitness", population->bestSolution->getParameters().partialFitness}
			};
		}
		else
		{
			record["bestSolution"] = nullptr;
		}

		const std::string directoryPath = fileDirectory + "/";
		std::filesystem::create_directories(directoryPath); // Ensure directory exists

		std::ofstream logFile(directoryPath + "overview.jsonl", std::ios::app);
		if (logFile.is_open())
		{
			logFile << record.dump() << "\n";
			logFile.close();
		}
		else
		{
			tools::logger::log(tools::logger::LogLevel::ERROR,
				"Failed to open log file for structured per generation overview.");
		}
	}

	void PopulationFileManager::saveBestSolutionOfEachGeneration() const
	{
		using namespace dnf_composer;

		const std::string directoryPath = std::format("{}best_solutions/prev_generations/", fileDirectory);
		std::filesystem::create_directories(directoryPath); // Ensure directory exist

		population->bestSolution->buildPhenotype();
		population->bestSolution->createPhenotypeEnvironment();
		auto simulation = population->bestSolution->getPhenotype();
		population->bestSolution->clearPhenotype();
		// save weights
		for (const auto& element : simulation.getElements())
		{
			if (element->getLabel() == element::ElementLabel::FIELD_COUPLING)
			{
				const auto fieldCoupling = std::dynamic_pointer_cast<element::FieldCoupling>(element);
				fieldCoupling->writeWeights();
			}
		}
		// save elements
		const std::string uniqueIdentifier = solutionIdentifier(population->bestSolution->getId(),
			population->parameters.currentGeneration, population->bestSolution->getSpeciesId(), population->bestSolution->getFitness());
		simulation.setUniqueIdentifier(uniqueIdentifier);
		SimulationFileManager sfm(std::make_shared<Simulation>(simulation), directoryPath);
		sfm.saveElementsToJson();
	}

	void PopulationFileManager::saveChampionsOfEachGeneration() const
	{
		using namespace dnf_composer;

		const std::string directoryPath = std::format("{}champions/prev_generations/", fileDirectory);
		std::filesystem::create_directories(directoryPath); // Ensure directory exist

		for (const auto& champion : population->champions)
		{
			if (champion == nullptr)
			{
				continue;
			}
			champion->buildPhenotype();
			champion->createPhenotypeEnvironment();
			auto simulation = champion->getPhenotype();
			champion->clearPhenotype();
			// save weights
			for (const auto& element : simulation.getElements())
			{
				if (element->getLabel() == element::ElementLabel::FIELD_COUPLING)
				{
					const auto fieldCoupling = std::dynamic_pointer_cast<element::FieldCoupling>(element);
					fieldCoupling->writeWeights();
				}
			}
			// save elements
			const std::string uniqueIdentifier = solutionIdentifier(champion->getId(),
				population->parameters.currentGeneration, champion->getSpeciesId(), champion->getFitness());
			simulation.setUniqueIdentifier(uniqueIdentifier);
			SimulationFileManager sfm(std::make_shared<Simulation>(simulation), directoryPath);
			sfm.saveElementsToJson();
		}
	}

	void PopulationFileManager::savePerGenerationStatistics() const
	{
		const std::string directoryPath = std::format("{}statistics/", fileDirectory);
		std::filesystem::create_directories(directoryPath); // Ensure directory exists

		std::ofstream logFile(std::format("{}generation_{}.txt", directoryPath, population->parameters.currentGeneration),
			std::ios::app);
		if (logFile.is_open())
		{
			for (const auto& solution : population->solutions)
			{
				logFile << solution->toString() << '\n';
			}
			logFile.close();
		}
		else
		{
			tools::logger::log(tools::logger::LogLevel::ERROR, "Failed to open log file for statistics.");
		}
	}

	void PopulationFileManager::savePerGenerationSpecies() const
	{
		const std::string directoryPath = std::format("{}species/", fileDirectory);
		std::filesystem::create_directories(directoryPath); // Ensure directory exists

		std::ofstream logFile(std::format("{}generation_{}.txt", directoryPath, population->parameters.currentGeneration),
			std::ios::app);
		if (logFile.is_open())
		{
			for (const auto& species : population->speciesList)
			{
				logFile << species->toString() << '\n';
			}
			logFile.close();
		}
		else
		{
			tools::logger::log(tools::logger::LogLevel::ERROR, "Failed to open log file for species.");
		}
	}

#ifdef NEAT_DNFS_PROFILE
	void PopulationFileManager::saveProfileForGeneration() const
	{
		// Fixed, explicit column set -- not whatever tools::profiler::snapshot()
		// happens to contain this generation. "save" in particular is only ever
		// recorded when fileManager is set, so deriving columns from the snapshot
		// would silently shift every value to the wrong header on a generation
		// where that phase's bucket is absent. elapsedSeconds() already returns
		// 0.0 for a phase that was not timed, which keeps every row the same
		// shape as the header written once below.
		static constexpr std::array<std::string_view, 5> phaseNames{
			"evaluate", "speciate", "upkeep", "reproduceAndSelect", "save"
		};

		const std::string filePath = fileDirectory + "profile.csv";
		const bool fileExists = std::filesystem::exists(filePath);

		std::ofstream csvFile(filePath, std::ios::app);
		if (!csvFile.is_open())
		{
			tools::logger::log(tools::logger::LogLevel::ERROR, "Failed to open log file for profiling data.");
			return;
		}

		if (!fileExists)
		{
			csvFile << "generation";
			for (const auto& phaseName : phaseNames)
			{
				csvFile << "," << phaseName;
			}
			csvFile << "\n";
		}

		csvFile << population->parameters.currentGeneration;
		for (const auto& phaseName : phaseNames)
		{
			csvFile << "," << tools::profiler::elapsedSeconds(phaseName);
		}
		csvFile << "\n";
	}
#endif
}
