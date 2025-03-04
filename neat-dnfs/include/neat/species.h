#pragma once

#include <vector>
#include "solution.h"

namespace neat_dnfs
{
    class Species;
    typedef std::unique_ptr<Species> SpeciesPtr;

    class Species
    {
    private:
		static int currentSpeciesId;
        int id = 0;
        int offspringCount;
        SolutionPtr representative;
        SolutionPtr champion;
        std::vector<SolutionPtr> members;
        std::vector<SolutionPtr> offspring;
        bool extinct;
        int age;
        bool hasFitnessImproved = true;
        int generationsSinceFitnessImproved = 0;
    public:
        Species();
		~Species()
		{
			// Clean up any dynamically allocated resources
			representative = nullptr;
			champion = nullptr;
			members.clear();
			offspring.clear();
		}
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
        static void resetUniqueIdentifier()
        {
			currentSpeciesId = 0;
        }

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