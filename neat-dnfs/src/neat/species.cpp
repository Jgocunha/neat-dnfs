#include "neat/species.h"
#include <format> 


namespace neat_dnfs
{
	int Species::currentSpeciesId = 0;

	Species::Species()
		: id(currentSpeciesId++), representative(nullptr)
	{
	}

	void Species::setRepresentative(const SolutionPtr& newRepresentative)
	{
		representative = newRepresentative;
	}

	void Species::randomlyAssignRepresentative()
	{
		if (members.empty())
		{
			return;
		}

		representative = members[tools::utils::generateRandomInt(0, static_cast<int>(members.size() - 1))];
	}

	void Species::assignChampion()
	{
		if (members.empty())
		{
			return;
		}

		sortMembersByFitness();

		const double prevFitness = champion == nullptr ? 0 : champion->getParameters().fitness;
		const double currentFitness = members[0]->getParameters().fitness;
		if (currentFitness > prevFitness)
		{
			hasFitnessImproved = true;
			generationsSinceFitnessImproved = 0;
		}
		else
		{
			hasFitnessImproved = false;
			generationsSinceFitnessImproved++;
		}

		champion = members[0];
	}

	size_t Species::size() const
	{
		return members.size();
	}

	void Species::setOffspringCount(int count)
	{
		offspringCount = count;
	}

	SolutionPtr Species::getRepresentative() const
	{
		return representative;
	}

	SolutionPtr Species::getChampion() const
	{
		return champion;
	}

	int Species::getId() const
	{
		return id;
	}

	double Species::totalAdjustedFitness() const
	{
		double total = 0;
		for (const auto& member : members)
		{
			total += member->getParameters().adjustedFitness;
		}

		return total;
	}

	int Species::getOffspringCount() const
	{
		return offspringCount;
	}

	std::vector<SolutionPtr> Species::getMembers() const
	{
		return members;
	}

	bool Species::isExtinct() const
	{
		return extinct;
	}

	bool Species::hasFitnessImprovedOverTheLastGenerations() const
	{
		return generationsSinceFitnessImproved < PopulationConstants::generationsWithoutImprovementThresholdInSpecies;
	}

	void Species::incrementAge()
	{
		age++;
	}

	void Species::addSolution(const SolutionPtr& solution)
	{
		if (!contains(solution))
		{
			members.push_back(solution);
		}
	}

	void Species::removeSolution(const SolutionPtr& solution)
	{
		const auto it = std::ranges::find(members, solution);
		if (it != members.end())
		{
			members.erase(it);
		}
		// A representative that leaves must not linger: another species may
		// legitimately hold the same solution as ITS representative or a member,
		// which would otherwise let two species share one representative pointer.
		// randomlyAssignRepresentative() no-ops on an empty members list, so clear
		// explicitly first to cover the case where the departing solution was the
		// species' only member.
		if (representative == solution)
		{
			representative = nullptr;
			randomlyAssignRepresentative();
		}
	}

	bool Species::isCompatible(const SolutionPtr& solution) const
	{
		if (representative == nullptr)
		{
			return false;
		}

		if (AblationConstants::disableSpeciation)
		{
			return true;
		}

		// N is the NEAT compatibility-distance normalizer: the larger genome's
		// connection-gene count (a standard NEAT choice, not total genome size).
		int N = static_cast<int>(std::max(representative->getNumConnectionGenes(), solution->getNumConnectionGenes()));
		if (N < 20)
		{
			N = 1; // Normalize for small genomes
		}

		const auto& representativeGenome = representative->getGenome();
		const auto& solutionGenome = solution->getGenome();

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
		const auto toRemove = static_cast<size_t>(static_cast<double>(members.size()) * ratio);
		for (size_t i = 0; i < toRemove && !members.empty(); ++i)
		{
			members.pop_back();
		}
		// A pruned representative must not linger: isCompatible() would keep
		// measuring genetic distance against a solution this species no longer
		// holds. randomlyAssignRepresentative() no-ops on an empty members list,
		// so clear explicitly first to cover a full prune.
		if (representative != nullptr && !contains(representative))
		{
			representative = nullptr;
			randomlyAssignRepresentative();
		}
	}
 
	void Species::crossover()
	{
		offspring.clear();

		if (members.empty())
		{
			if (offspringCount > 0)
			{
				log(tools::logger::LogLevel::FATAL, std::format("Species {} with no members has offspring count > 0.", id));
			}
			extinct = true;
			representative = nullptr;
			champion = nullptr;
			members.clear();
			offspring.clear();
			return;
		}

		if (members.size() == 1) // only one organism in the species
		{
			for (size_t i = 0; i < offspringCount; ++i)
			{
				const SolutionPtr parent1 = members[tools::utils::generateRandomInt(0, static_cast<int>(members.size() - 1))];
				const SolutionPtr son = parent1->crossover(parent1);
				offspring.emplace_back(son);
			}
		}
		else // more than one organism in the species
		{
			for (size_t i = 0; i < offspringCount; ++i)
			{
				const SolutionPtr parent1 = members[tools::utils::generateRandomInt(0, static_cast<int>(members.size() - 1))];
				const SolutionPtr parent2 = AblationConstants::disableCrossover
					? parent1
					: members[tools::utils::generateRandomInt(0, static_cast<int>(members.size() - 1))];
				const SolutionPtr son = parent1->crossover(parent2);
				if (son->getId() == parent1->getId() || son->getId() == parent2->getId())
				{
					std::cout << "When crossing over id's are the same " << parent1->getId() << " or " << parent2->getId() << " is equal to " << son->getId() << '\n';
				}
				offspring.emplace_back(son);
			}
		}
		extinct = false;
	}

	void Species::replaceMembersWithOffspring()
	{
		members.clear();
		for (const auto& child : offspring)
		{
			members.emplace_back(child);
		}
		// The representative must always be a current member: a stale pointer to
		// last generation's representative (which offspring replaced) could later
		// collide with another species' representative and violate the invariant
		// that an individual can only represent the one species it belongs to.
		randomlyAssignRepresentative();
	}

	void Species::copyChampionToNextGeneration()
	{
		if (champion == nullptr)
		{
			return;
		}

		const size_t initialMembersSize = members.size();
		members.pop_back();
		members.emplace_back(champion);
		const size_t finalMembersSize = members.size();

		if (initialMembersSize != finalMembersSize)
		{
			log(tools::logger::LogLevel::FATAL, "Champion was not added to the species.");
			throw std::runtime_error("Champion was not added to the species.");
		}
	}

	std::string Species::toString() const
{
    return std::format(
        "species {} [ age: {}, extinct: {}, improved: {}, gens. since imp.: {}, offs.: {}, mem: {}, rep.: {{{}}}, champ.: {{{}}}]",
        id,
        age,
        extinct ? "yes" : "no",
        hasFitnessImproved ? "yes" : "no",
        generationsSinceFitnessImproved,
        offspringCount,
        members.size(),
        representative == nullptr ? "none" : representative->toString(),
        champion == nullptr ? "none" : champion->toString()
    );
}

	void Species::print() const
	{
		tools::logger::log(tools::logger::INFO, toString());
	}
}
