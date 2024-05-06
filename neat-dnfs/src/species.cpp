#include "species.h"

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
}
