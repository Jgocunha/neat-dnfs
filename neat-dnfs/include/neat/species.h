#pragma once

#include <vector>
#include "solution.h"

namespace neat_dnfs
{
    class Species;
    static uint16_t currentSpeciesId = 0;
    typedef std::unique_ptr<Species> SpeciesPtr;

    class Species
    {
    private:
        uint16_t id;
        uint16_t offspringCount;
        SolutionPtr representative;
        std::vector<SolutionPtr> members;
        std::vector<SolutionPtr> elites;
        std::vector<SolutionPtr> leastFit;
        std::vector<SolutionPtr> offspring;
    public:
        Species();
        void setRepresentative(const SolutionPtr& newRepresentative);
        size_t size() const { return members.size(); }
        void setOffspringCount(uint16_t count) { offspringCount = count; }
        SolutionPtr getRepresentative() const;
        uint16_t getId() const { return id; }
        double totalAdjustedFitness() const;
        uint16_t getOffspringCount() const { return offspringCount; }
        std::vector<SolutionPtr> getMembers() const { return members; }
        std::vector<SolutionPtr> getElites() const { return elites; }
        std::vector<SolutionPtr> getLeastFit() const { return leastFit; }
        std::vector<SolutionPtr> getOffspring() const { return offspring; }

        void addSolution(const SolutionPtr& solution);
        void removeSolution(const SolutionPtr& solution);
        bool isCompatible(const SolutionPtr& solution) const;
        bool contains(const SolutionPtr& solution) const;
        void clearOffspring() { offspring.clear(); }
        void sortMembersByFitness();

        void selectElitesAndLeastFit();
        void crossover();
        void updateMembers();
    private:
        static int validateUniqueSolutions(const std::vector<SolutionPtr>& solutions);
    };
}