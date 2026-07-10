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

        [[nodiscard]] size_t size() const;
        void setOffspringCount(int count);
        [[nodiscard]] SolutionPtr getRepresentative() const;
        [[nodiscard]] SolutionPtr getChampion() const;
        [[nodiscard]] int getId() const;
        [[nodiscard]] double totalAdjustedFitness() const;
        [[nodiscard]] int getOffspringCount() const;
        [[nodiscard]] std::vector<SolutionPtr> getMembers() const;
        [[nodiscard]] bool isExtinct() const;
        [[nodiscard]] bool hasFitnessImprovedOverTheLastGenerations() const;
        void incrementAge();
        static void resetUniqueIdentifier()
        {
			currentSpeciesId = 0;
        }

        void addSolution(const SolutionPtr& solution);
        void removeSolution(const SolutionPtr& solution);
        /// @brief Returns true if @p solution's genome is within the compatibility distance threshold of this species' representative.
        [[nodiscard]] bool isCompatible(const SolutionPtr& solution) const;
        [[nodiscard]] bool contains(const SolutionPtr& solution) const;
        void sortMembersByFitness();
        void pruneWorsePerformingMembers(double ratio);
    	void crossover();
        void replaceMembersWithOffspring();
        void copyChampionToNextGeneration();

        [[nodiscard]] std::string toString() const;
        void print() const;
    };
}