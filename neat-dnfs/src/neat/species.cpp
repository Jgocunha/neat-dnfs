#include "neat/species.h"

namespace neat_dnfs
{
	Species::Species()
	{
		id = currentSpeciesId++;
		offspringCount = 0;
	}

	std::shared_ptr<Solution> Species::getRepresentative() const
	{
		return representative;
	}

	void Species::setRepresentative(const SolutionPtr& newRepresentative)
	{
		representative = newRepresentative;
	}

	void Species::addSolution(const SolutionPtr& solution)
	{
		if (!contains(solution))
			members.push_back(solution);
	}

	void Species::removeSolution(const SolutionPtr& solution)
	{
		const auto it = std::ranges::find(members, solution);
		if (it != members.end())
			members.erase(it);
	}

	bool Species::isCompatible(const SolutionPtr& solution) const
	{
		if (representative == nullptr)
			throw std::runtime_error("Species " + std::to_string(id) + " has no representative.");

		int N = static_cast<int>(std::max(representative->getGenomeSize(), solution->getGenomeSize()));
		if (N < 20) N = 1; // Normalize for small genomes

		const auto representativeGenome = representative->getGenome();
		const auto solutionGenome = solution->getGenome();

		const double excessCoefficient = SpeciesConstants::excessGenesCompatibilityWeight
			* representativeGenome.excessGenes(solutionGenome);
		const double disjointCoefficient = SpeciesConstants::disjointGenesCompatibilityWeight
			* representativeGenome.disjointGenes(solutionGenome);
		const double weightCoefficient = SpeciesConstants::averageConnectionDifferenceCompatibilityWeight
			* representativeGenome.averageConnectionDifference(solutionGenome);

		const double geneticDistance = (excessCoefficient + disjointCoefficient + weightCoefficient) / N;

		return geneticDistance < SpeciesConstants::compatibilityThreshold;
	}

	bool Species::contains(const SolutionPtr& solution) const
	{
		return std::ranges::find(members, solution) != members.end();
	}

	double Species::totalAdjustedFitness() const
	{
		double total = 0;
		for (const auto& member : members)
			total += member->getParameters().adjustedFitness;

		return total;
	}

	uint16_t Species::getKillCount() const
	{
		if (members.size() <= 2)
			return 0;
		return std::ceil(static_cast<double>(members.size()) * PopulationConstants::killRatio);
	}

	std::vector<SolutionPtr> Species::killLeastFitSolutions()
	{
		if (members.size() <= 2)
		{
			log(tools::logger::LogLevel::INFO, "Species " + std::to_string(id) +
				" has " + std::to_string(members.size()) + " members. Killing no individuals.");
			return {};
		}

		sortMembersByFitness();

		const size_t numToRemove = getKillCount();
		const size_t numSurvivors = members.size() - numToRemove; // Keep only the number of offspring specified

		log(tools::logger::LogLevel::INFO, "Species " + std::to_string(id) + " has " +
			std::to_string(members.size()) + " members. Killing " +
			std::to_string(numToRemove) + " least fit members. Remaining: " + std::to_string(numSurvivors));

		std::vector<SolutionPtr> removed;
		removed.reserve(numToRemove);
		std::move(members.end() - static_cast<uint16_t>(numToRemove), members.end(), std::back_inserter(removed));
		members.resize(numSurvivors);

		return removed;
	}

	void Species::crossover()
	{
		offspring.clear();
		if (members.size() <= 1)
		{
			for (size_t i = 0; i < offspringCount; ++i)
			{
				const SolutionPtr parent1 = members[tools::utils::generateRandomInt(0, static_cast<int>(members.size() - 1))];
				offspring.push_back(parent1);
			}
		}
		else
		{
			for (size_t i = 0; i < offspringCount; ++i)
			{
				const SolutionPtr parent1 = members[tools::utils::generateRandomInt(0, static_cast<int>(members.size() - 1))];
				const SolutionPtr parent2 = members[tools::utils::generateRandomInt(0, static_cast<int>(members.size() - 1))];
				offspring.push_back(parent1->crossover(parent2));
			}
		}
		log(tools::logger::LogLevel::INFO, "Species " + std::to_string(id) + " has " +
			std::to_string(members.size()) + " members. Created " + std::to_string(offspringCount) + " offspring.");
	}

	void Species::sortMembersByFitness()
	{
		std::ranges::sort(members, [](const SolutionPtr& a, const SolutionPtr& b)
			{
				return a->getParameters().fitness > b->getParameters().fitness;
			}
		);
	}
}