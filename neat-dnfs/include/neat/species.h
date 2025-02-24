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
        SolutionPtr champion;
        std::vector<SolutionPtr> members;
        std::vector<SolutionPtr> offspring;
        bool extinct;
        int age;
        bool hasFitnessImproved;
        int generationsSinceFitnessImproved;
    public:
        Species();
        void setRepresentative(const SolutionPtr& newRepresentative);
        void randomlyAssignRepresentative();
        void assignChampion();

        size_t size() const;
        void setOffspringCount(int count);
        SolutionPtr getRepresentative() const;
        SolutionPtr getChampion() const;
        int getId() const;
        double totalAdjustedFitness() const;
        int getOffspringCount() const;
        std::vector<SolutionPtr> getMembers() const;
        bool isExtinct() const;
        bool hasFitnessImprovedOverTheLastGenerations() const;
        void incrementAge();

        void addSolution(const SolutionPtr& solution);
        void removeSolution(const SolutionPtr& solution);
        bool isCompatible(const SolutionPtr& solution) const;
        bool contains(const SolutionPtr& solution) const;
        void sortMembersByFitness();
        void pruneWorsePerformingMembers(double ratio);
    	void crossover();
        void replaceMembersWithOffspring();
        void copyChampionToNextGeneration();

        std::string toString() const;
        void print() const;
    };
}