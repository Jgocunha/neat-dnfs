#pragma once

#include <vector>
#include "solution.h"

namespace neat_dnfs
{
    static uint8_t currentSpeciesId = 1;

    class Species
    {
    private:
        uint8_t id;
		SolutionPtr representative;
    public:
        Species();
    	SolutionPtr getRepresentative() const;
        void setRepresentative(const SolutionPtr& newRepresentative);
        uint8_t getId() const { return id; }
    };
}
