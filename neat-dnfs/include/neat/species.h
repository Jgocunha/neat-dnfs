#pragma once

#include <vector>
#include "solution.h"

namespace neat_dnfs
{
    class Species
    {
    private:
		static int currentSpeciesId;
        int id = 0;
        int offspringCount{0};
        SolutionPtr representative;
        SolutionPtr champion;
        std::vector<SolutionPtr> members;
        std::vector<SolutionPtr> offspring;
        bool extinct{false};
        int age{0};
        bool hasFitnessImproved = true;
        int generationsSinceFitnessImproved = 0;
    public:
        Species();
		~Species() = default;
		Species(const Species& other) = default;
		Species(Species&& other) = default;
		Species& operator=(const Species& other) = default;
		Species& operator=(Species&& other) = default;
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
        void extinguish()
        {
            extinct = true;
            representative = nullptr;
            champion = nullptr;
            members.clear();
            offspring.clear();
        }
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
        /// @brief Sorts members by fitness and removes the floor(size() * ratio) worst-performing members.
        /// Reassigns the representative if it was among those removed.
        /// @param ratio Fraction of the current membership to remove, in [0, 1].
        void pruneWorsePerformingMembers(double ratio);
    	void crossover();
        void replaceMembersWithOffspring();
        void copyChampionToNextGeneration();

        [[nodiscard]] std::string toString() const;
        void print() const;
    };
}