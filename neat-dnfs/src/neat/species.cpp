#include "neat/species.h"


namespace neat_dnfs
{
	Species::Species()
		: id(currentSpeciesId++), offspringCount(0)
	{
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

		const double excessCoefficient = CompatibilityCoefficients::excessGenesCompatibilityWeight
			* representativeGenome.excessGenes(solutionGenome);
		const double disjointCoefficient = CompatibilityCoefficients::disjointGenesCompatibilityWeight
			* representativeGenome.disjointGenes(solutionGenome);
		const double weightCoefficient = CompatibilityCoefficients::averageConnectionDifferenceCompatibilityWeight
			* representativeGenome.averageConnectionDifference(solutionGenome);

		const double geneticDistance = (excessCoefficient + disjointCoefficient + weightCoefficient) / N;

		return geneticDistance < CompatibilityCoefficients::compatibilityThreshold;
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

	void Species::sortMembersByFitness()
	{
		std::ranges::sort(members, [](const SolutionPtr& a, const SolutionPtr& b)
			{
				return a->getParameters().fitness > b->getParameters().fitness;
			}
		);
	}

	void Species::pruneWorsePerformingMembers(double ratio)
	{
		sortMembersByFitness();
		for (size_t i = 0; i < static_cast<size_t>(members.size() * ratio); ++i)
			members.pop_back();
	}

	void Species::crossover()
	{
		offspring.clear();

		if (members.empty())
		{
			if (offspringCount > 0)
				log(tools::logger::LogLevel::FATAL, "Species " + std::to_string(id) + " with no members has offspring count > 0.");
			extinct = true;
			return;
		}

		if (members.size() == 1) // only one organism in the species
		{
			for (size_t i = 0; i < offspringCount; ++i)
			{
				const SolutionPtr parent1 = members[tools::utils::generateRandomInt(0, static_cast<int>(members.size() - 1))];
				const SolutionPtr son = parent1->crossover(parent1);
				offspring.push_back(son);
			}
		}
		else // more than one organism in the species
		{
			for (size_t i = 0; i < offspringCount; ++i)
			{
				const SolutionPtr parent1 = members[tools::utils::generateRandomInt(0, static_cast<int>(members.size() - 1))];
				const SolutionPtr parent2 = members[tools::utils::generateRandomInt(0, static_cast<int>(members.size() - 1))];
				const SolutionPtr son = parent1->crossover(parent2);
				offspring.push_back(son);
			}
		}
		extinct = false;
	}

	void Species::replaceMembersWithOffspring()
	{
		members.clear();
		for (const auto& child : offspring)
			members.push_back(child);
	}

	std::string Species::toString() const
	{
		std::string str = "Species " + std::to_string(id);
		str += " { Offspring count: " + std::to_string(offspringCount);
		str += ", Members count: " + std::to_string(members.size());
		str += " Representative: " + (representative == nullptr ? "None" :
			" { Address: " + representative->getAddress()
			+ ", Fitness: " + std::to_string(representative->getParameters().fitness) 
			+ ", Gene count: " + std::to_string(representative->getGenomeSize()) + " } }");
		return str;
	}

	void Species::print() const
	{
		tools::logger::log(tools::logger::INFO, toString());
	}
}
