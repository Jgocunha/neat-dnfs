#include "neat/species.h"

namespace neat_dnfs
{
	Species::Species()
	{
		id = currentSpeciesId++;
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

	void Species::killLeastFitSolution()
	{
		if (members.empty())
			return;

		sortSolutionsByFitness();
		members.pop_back();
	}

	void Species::updateRepresentative()
	{
		representative = getMostFitSolution();
	}

	void Species::calculateAdjustedFitness() const
	{
		for (const auto& member : members)
		{
			const double adjustedFitness = 
				member->getParameters().fitness / static_cast<double>(members.size());
			member->setAdjustedFitness(adjustedFitness);
		}
	}

	SolutionPtr Species::getLeastFitSolution() const
	{
		if (members.empty())
			return nullptr;

		SolutionPtr leastFit = members.front();
		for (const auto& member : members)
		{
			if (member->getParameters().fitness < leastFit->getParameters().fitness)
				leastFit = member;
		}

		return leastFit;
	}

	SolutionPtr Species::getMostFitSolution() const
	{
		if (members.empty())
			return nullptr;

		SolutionPtr mostFit = members.front();
		for (const auto& member : members)
		{
			if (member->getParameters().fitness > mostFit->getParameters().fitness)
				mostFit = member;
		}

		return mostFit;
	}

	SolutionPtr Species::getRandomSolution() const
	{
		if (members.empty())
			return nullptr;

		return members[rand() % members.size()];
	}

	void Species::sortSolutionsByFitness()
	{
		std::sort(members.begin(), members.end(), [](const SolutionPtr& a, const SolutionPtr& b)
		{
			return a->getParameters().fitness > b->getParameters().fitness;  // higher is better
		});
	}

}
