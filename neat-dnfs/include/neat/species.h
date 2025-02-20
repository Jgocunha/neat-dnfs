#pragma once

#include <vector>
#include "solution.h"

namespace neat_dnfs
{
    class Species;
    static int currentSpeciesId = 0;
    typedef std::unique_ptr<Species> SpeciesPtr;

    class Species
    {
    private:
        int id;
        int offspringCount;
        SolutionPtr representative;
        std::vector<SolutionPtr> members;
        std::vector<SolutionPtr> offspring;
        bool extinct;
        int age;
    public:
        Species();
        void setRepresentative(const SolutionPtr& newRepresentative);
        size_t size() const;
        void setOffspringCount(int count);
        SolutionPtr getRepresentative() const;
        int getId() const;
        double totalAdjustedFitness() const;
        int getOffspringCount() const;
        std::vector<SolutionPtr> getMembers() const;
        bool isExtinct() const;
        void incrementAge();

        void addSolution(const SolutionPtr& solution);
        void removeSolution(const SolutionPtr& solution);
        bool isCompatible(const SolutionPtr& solution) const;
        bool contains(const SolutionPtr& solution) const;
        void sortMembersByFitness();
        void pruneWorsePerformingMembers(double ratio);
    	void crossover();
        void replaceMembersWithOffspring();

        std::string toString() const;
        void print() const;
    };
}