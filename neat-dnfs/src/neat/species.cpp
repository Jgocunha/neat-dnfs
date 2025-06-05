#include "neat/species.h"


namespace neat_dnfs
{
	int Species::currentSpeciesId = 0;

	Species::Species()
		: id(currentSpeciesId++), offspringCount(0), representative(nullptr),
			members(), offspring(), extinct(false), age(0)
	{
	}

	void Species::setRepresentative(const SolutionPtr& newRepresentative)
	{
		representative = newRepresentative;
	}

	void Species::randomlyAssignRepresentative()
	{
		if (members.empty())
			return;

		representative = members[tools::utils::generateRandomInt(0, static_cast<int>(members.size() - 1))];
	}

	void Species::assignChampion()
	{
		if (members.empty())
			return;

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
			total += member->getParameters().adjustedFitness;

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
		if (generationsSinceFitnessImproved >= PopulationConstants::generationsWithoutImprovementThresholdInSpecies)
			return false;
		return true;
	}

	void Species::incrementAge()
	{
		age++;
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
		{
			//throw std::runtime_error("Species " + std::to_string(id) + " has no representative.");
			return false;
		}

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
		// narrowing conversion from size_t to double
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
				//if (son->getId() == parent1->getId())
					//std::cout << "When crossing over with clone parents id's are the same " << parent1->getId() << std::endl;
				offspring.emplace_back(son);
			}
		}
		else // more than one organism in the species
		{
			for (size_t i = 0; i < offspringCount; ++i)
			{
				const SolutionPtr parent1 = members[tools::utils::generateRandomInt(0, static_cast<int>(members.size() - 1))];
				const SolutionPtr parent2 = members[tools::utils::generateRandomInt(0, static_cast<int>(members.size() - 1))];
				const SolutionPtr son = parent1->crossover(parent2);
				if (son->getId() == parent1->getId() || son->getId() == parent2->getId())
					std::cout << "When crossing over id's are the same " << parent1->getId() << " or " << parent2->getId() << " is equal to " << son->getId() << std::endl;
				offspring.emplace_back(son);
			}
		}
		extinct = false;
	}

	void Species::replaceMembersWithOffspring()
	{
		members.clear();
		for (const auto& child : offspring)
			members.emplace_back(child);
	}

	void Species::copyChampionToNextGeneration()
	{
		if (champion == nullptr)
			return;

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
		std::string str = "species " + std::to_string(id);
		str += " [ age: " + std::to_string(age);
		str += ", extinct: " + std::string((extinct ? "yes" : "no"));
		str += ", improved: " + std::string((hasFitnessImproved ? "yes" : "no"));
		str += ", gens. since imp.: " + std::to_string(generationsSinceFitnessImproved);
		str += "  offs.: " + std::to_string(offspringCount);
		str += ", mem: " + std::to_string(members.size());
		//str += " rep.: {" + (representative == nullptr ? "none}]" : representative->toString()) + "}";
		//str += " champ.: {" + (champion == nullptr ? "none}]" : champion->toString()) + "}";
		str += "]";
		return str;
	}

	void Species::print() const
	{
		tools::logger::log(tools::logger::INFO, toString());
	}
}
