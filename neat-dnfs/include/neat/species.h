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
        bool extinct = false;
        int age = 0;
    public:
        Species();
        void setRepresentative(const SolutionPtr& newRepresentative);
        size_t size() const { return members.size(); }
        void setOffspringCount(int count) { offspringCount = count; }
        SolutionPtr getRepresentative() const { return representative; }
        int getId() const { return id; }
        double totalAdjustedFitness() const;
        int getOffspringCount() const { return offspringCount; }
        std::vector<SolutionPtr> getMembers() const { return members; }
        bool isExtinct() const { return extinct; }
        void incrementAge() { age++; }

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