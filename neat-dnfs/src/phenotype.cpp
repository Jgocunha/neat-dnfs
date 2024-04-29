#include "phenotype.h"

namespace neat_dnfs
{
	Phenotype::Phenotype()
	{
	}

	std::shared_ptr<Architecture> Phenotype::getArchitecture() const
	{
		return std::shared_ptr<Architecture>();
	}

	void Phenotype::buildFromGenome(const Genome& genome)
	{

	}

	void Phenotype::evaluate()
	{

	}
}