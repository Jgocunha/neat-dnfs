#include "neat/species.h"

namespace neat_dnfs
{
	Species::Species()
		: id(currentSpeciesId++), offspringCount(0)
	{
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

	void Species::sortMembersByFitness()
	{
		std::ranges::sort(members, [](const SolutionPtr& a, const SolutionPtr& b)
			{
				return a->getParameters().fitness > b->getParameters().fitness;
			}
		);
	}

	void Species::selectElitesAndLeastFit()
	{
		elites.clear();
		leastFit.clear();
		sortMembersByFitness();

		const auto numElites = static_cast<size_t>(std::ceil((1 - PopulationConstants::killRatio) * static_cast<double>(members.size())));
		const size_t numLeastFit = members.size() - numElites;
		assert(members.size() == numElites + numLeastFit);

		elites.reserve(numElites);
		for (size_t i = 0; i < numElites; ++i)
			elites.push_back(members[i]);

		leastFit.reserve(numLeastFit);
		for (size_t i = numElites; i < members.size(); ++i)
			leastFit.push_back(members[i]);

		// make sure no solution is in both lists
		for (const auto& elite : elites)
		{
			const auto it = std::ranges::find(leastFit, elite);
			if (it != leastFit.end())
				throw std::runtime_error("Solution is both elite and least fit.");
		}
	}

	void Species::crossover()
	{
		offspring.clear();

		if (members.empty())
		{
			if (offspringCount > 0)
				log(tools::logger::LogLevel::FATAL, "Species " + std::to_string(id) + " with no members has offspring count > 0.");
			return;
		}

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
		updateMembers();
	}

	void Species::updateMembers()
	{
		members.clear();
		members.reserve(elites.size() + offspring.size());
		for (const auto& elite : elites)
			members.push_back(elite);
		for (const auto& child : offspring)
			members.push_back(child);
	}


}