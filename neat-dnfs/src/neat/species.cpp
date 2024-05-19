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
		const auto it = std::find(members.begin(), members.end(), solution);
		if (it != members.end()) 
			members.erase(it);
	}

	bool Species::isCompatible(const SolutionPtr& solution) const
	{
		constexpr double compatibilityThreshold = 3.0;
		constexpr double c1 = 0.5, c2 = 0.4, c3 = 0.1;
		int N = std::max(representative->getGenomeSize(), solution->getGenomeSize());
		if (N < 20) N = 1; // Normalize for small genomes

		const auto representativeGenome = representative->getGenome();
		const auto solutionGenome = solution->getGenome();

		const double excessCoeff = c1 * representativeGenome.excessGenes(solutionGenome);
		const double disjointCoeff = c2 * representativeGenome.disjointGenes(solutionGenome);
		const double weightCoeff = c3 * representativeGenome.averageConnectionDifference(solutionGenome);

		const double geneticDistance = (excessCoeff + disjointCoeff + weightCoeff) / N;

		return geneticDistance < compatibilityThreshold;
	}

	bool Species::contains(const SolutionPtr& solution) const
	{
		return std::find(members.begin(), members.end(), solution) != members.end();
	}

	double Species::totalAdjustedFitness() const
	{
		double total = 0;
		for (const auto& member : members)
			total += member->getParameters().adjustedFitness;

		return total;
	}

	std::vector<SolutionPtr> Species::killLeastFitSolutions()
	{
		if (members.size() <= 2)
		{
			log(tools::logger::LogLevel::INFO, "Species " + std::to_string(id) + " has " + std::to_string(members.size()) + " members. Killing none.");
			return {};
		}

		std::vector<SolutionPtr> removed;

		std::sort(members.begin(), members.end(), [](const SolutionPtr& a, const SolutionPtr& b)
			{
			return a->getParameters().fitness > b->getParameters().fitness;  // higher is better
			}
		); 

		const size_t numSurvivors = members.size() - offspringCount;
		log(tools::logger::LogLevel::INFO, "Species " + std::to_string(id) + " has " + std::to_string(members.size()) + " members. Killing " 
			+ std::to_string(numSurvivors) + " least fit members. Remaining: " + std::to_string(offspringCount));
		removed.assign(members.begin() + numSurvivors, members.end());
		members.erase(members.begin() + numSurvivors, members.end());

		return removed;
	}


	void Species::crossover()
	{
		offspring.clear();
		if(members.size() <= 1)
		{
			for (size_t i = 0; i < offspringCount; ++i)
			{
				const SolutionPtr parent1 = members[rand() % members.size()];
				offspring.push_back(parent1);
			}
		}
		else
		{
			for (size_t i = 0; i < offspringCount; ++i)
			{
				const SolutionPtr parent1 = members[rand() % members.size()];
				const SolutionPtr parent2 = members[rand() % members.size()];
				//offspring.push_back(Solution::crossover(parent1, parent2));
			}
		}
		log(tools::logger::LogLevel::INFO, "Species " + std::to_string(id) + " has " + std::to_string(members.size()) + " members. Created " + std::to_string(offspringCount) + " offspring.");
		//members.insert(members.end(), offspring.begin(), offspring.end());
	}

}
