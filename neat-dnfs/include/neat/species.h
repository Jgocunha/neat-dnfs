#pragma once

#include <vector>
#include "solution.h"

namespace neat_dnfs
{
	class Species;
	static uint16_t currentSpeciesId = 1;
    typedef std::unique_ptr<Species> SpeciesPtr;

    class Species
    {
    private:
        uint16_t id;
        uint16_t offspringCount;
		SolutionPtr representative;
		std::vector<SolutionPtr> members;
    public:
        Species();
    	SolutionPtr getRepresentative() const;
        void setRepresentative(const SolutionPtr& newRepresentative);
        uint16_t getId() const { return id; }

        void addSolution(const SolutionPtr& solution);
        void removeSolution(const SolutionPtr& solution);
        bool isCompatible(const SolutionPtr& solution) const;
        bool contains(const SolutionPtr& solution) const;

        size_t size() const { return members.size(); }
        double totalAdjustedFitness() const;
        void setOffspringCount(uint16_t offspring) { offspringCount = offspring; }
        uint16_t getOffspringCount() const { return offspringCount; }

        std::vector<SolutionPtr> killLeastFitSolutions();

        void crossover();
   };
}