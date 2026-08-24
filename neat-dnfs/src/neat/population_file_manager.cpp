#include "neat/population_file_manager.h"

#include <array>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>

#include "neat/population.h"
#include "neat_tools/logger.h"
#include "neat_tools/resource_paths.h"

namespace neat_dnfs
{
	PopulationFileManager::PopulationFileManager(const Population& population)
		: population(&population)
	{}

	void PopulationFileManager::saveOverviewForGeneration() const
	{
		if (PopulationConstants::saveOverview)
		{
			savePerGenerationOverview();
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

		const std::string directoryPath = fileDirectory + "best_solutions/last_generation/";
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
				const std::string uniqueIdentifier = "solution " + std::to_string(solution->getId())
					+ " generation " + std::to_string(population->parameters.currentGeneration)
					+ " species " + std::to_string(solution->getSpeciesId())
					+ " fitness " + std::to_string(solution->getFitness());
				simulation.setUniqueIdentifier(uniqueIdentifier);
				SimulationFileManager sfm(std::make_shared<Simulation>(simulation), directoryPath);
				sfm.saveElementsToJson();
			}
		}
	}

	void PopulationFileManager::saveChampions() const
	{
		using namespace dnf_composer;

		const std::string directoryPath = fileDirectory + "champions/last_generation/";
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
			const std::string uniqueIdentifier = "solution " + std::to_string(champion->getId())
				+ " generation " + std::to_string(population->parameters.currentGeneration)
				+ " species " + std::to_string(champion->getSpeciesId())
				+ " fitness " + std::to_string(champion->getFitness());
			simulation.setUniqueIdentifier(uniqueIdentifier);
			SimulationFileManager sfm(std::make_shared<Simulation>(simulation), directoryPath);
			sfm.saveElementsToJson();
		}
	}

	void PopulationFileManager::saveTimestampsAndDuration() const
	{
		const std::string directoryPath = fileDirectory + "/";
		std::filesystem::create_directories(directoryPath); // Ensure directory exists

		std::ofstream logFile(directoryPath + "evolution_timestamps.txt", std::ios::app);
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

	void PopulationFileManager::saveAllSolutionsPerGeneration() const
	{
		using namespace dnf_composer;

		for (const auto& solution : population->solutions)
		{
			const std::string directoryPath = fileDirectory + "solutions/gen " + std::to_string(population->parameters.currentGeneration) + "/";
			std::filesystem::create_directories(directoryPath); // Ensure directory exists

			solution->buildPhenotype();
			solution->createPhenotypeEnvironment();
			auto simulation = solution->getPhenotype();
			solution->clearPhenotype();

			const std::string uniqueIdentifier = "solution " + std::to_string(solution->getId())
				+ " generation " + std::to_string(population->parameters.currentGeneration)
				+ " species " + std::to_string(solution->getSpeciesId())
				+ " fitness " + std::to_string(solution->getFitness());
			simulation.setUniqueIdentifier(uniqueIdentifier);
			SimulationFileManager sfm(std::make_shared<Simulation>(simulation), directoryPath);
			sfm.saveElementsToJson();
		}
	}

	void PopulationFileManager::savePerGenerationOverview() const
	{
		const std::string directoryPath = fileDirectory + "/";
		std::filesystem::create_directories(directoryPath); // Ensure directory exists

		std::ofstream logFile(directoryPath + "per_generation_overview.txt", std::ios::app);
		if (logFile.is_open())
		{
			logFile << "Current generation: " + std::to_string(population->parameters.currentGeneration);
			logFile << " Number of solutions: " + std::to_string(population->solutions.size());
			logFile << " Number of species: " + std::to_string(population->perGenStatistics.numberOfSpecies);
			logFile << " Number of active species: " + std::to_string(population->perGenStatistics.numberOfActiveSpecies);
			logFile << " Has fitness improved: " << (population->hasFitnessImproved ? "yes" : "no");
			logFile << " Number of generations without improvement: " + std::to_string(population->generationsWithoutImprovement);
			logFile << " Average fitness: " + std::to_string(population->perGenStatistics.averageFitness);
			logFile << " Best fitness: " + std::to_string(population->perGenStatistics.bestFitness);
			logFile << " Innovation number: " + std::to_string(population->perGenStatistics.innovationNumber);
			logFile << " Average genome size: " + std::to_string(population->perGenStatistics.averageGenomeSize);
			logFile << " Average connection genes: " + std::to_string(population->perGenStatistics.averageConnectionGenes);
			logFile << " Average field genes: " + std::to_string(population->perGenStatistics.averageFieldGenes);
			logFile << " Best solution: [" + population->bestSolution->toString() + "]";
			logFile << "\n";
			logFile.close();
		}
		else
		{
			tools::logger::log(tools::logger::LogLevel::ERROR,
				"Failed to open log file for field gene per generation statistics.");
		}
	}

	void PopulationFileManager::saveBestSolutionOfEachGeneration() const
	{
		using namespace dnf_composer;

		const std::string directoryPath = fileDirectory + "best_solutions/prev_generations/";
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
		const std::string uniqueIdentifier = "solution " + std::to_string(population->bestSolution->getId())
			+ " generation " + std::to_string(population->parameters.currentGeneration)
			+ " species " + std::to_string(population->bestSolution->getSpeciesId())
			+ " fitness " + std::to_string(population->bestSolution->getFitness());
		simulation.setUniqueIdentifier(uniqueIdentifier);
		SimulationFileManager sfm(std::make_shared<Simulation>(simulation), directoryPath);
		sfm.saveElementsToJson();
	}

	void PopulationFileManager::saveChampionsOfEachGeneration() const
	{
		using namespace dnf_composer;

		const std::string directoryPath = fileDirectory + "champions/prev_generations/";
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
			const std::string uniqueIdentifier = "solution " + std::to_string(champion->getId())
				+ " generation " + std::to_string(population->parameters.currentGeneration)
				+ " species " + std::to_string(champion->getSpeciesId())
				+ " fitness " + std::to_string(champion->getFitness());
			simulation.setUniqueIdentifier(uniqueIdentifier);
			SimulationFileManager sfm(std::make_shared<Simulation>(simulation), directoryPath);
			sfm.saveElementsToJson();
		}
	}

	void PopulationFileManager::savePerGenerationStatistics() const
	{
		const std::string directoryPath = fileDirectory + "statistics/";
		std::filesystem::create_directories(directoryPath); // Ensure directory exists

		std::ofstream logFile(directoryPath + "generation_" + std::to_string(population->parameters.currentGeneration) + ".txt",
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
		const std::string directoryPath = fileDirectory + "species/";
		std::filesystem::create_directories(directoryPath); // Ensure directory exists

		std::ofstream logFile(directoryPath + "generation_" + std::to_string(population->parameters.currentGeneration) + ".txt",
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
}
