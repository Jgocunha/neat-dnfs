#pragma once
#include <memory>
#include <random>

#include <elements/gauss_kernel.h>


namespace neat_dnfs
{
	static unsigned long int currentInnovationNumber = 1;

	struct ConnectionGeneParameters
	{
		unsigned long int inGeneId;
		unsigned long int outGeneId;
		bool enabled;
		unsigned long int innovationNumber;

		ConnectionGeneParameters(unsigned long int inGeneId, 
			unsigned long int outGeneId)
			: inGeneId(inGeneId),
			outGeneId(outGeneId),
			enabled(true),
			innovationNumber(currentInnovationNumber++)
		{}
	};

	class ConnectionGene
	{
	private:
		ConnectionGeneParameters parameters;
		std::shared_ptr<dnf_composer::element::Kernel> kernel;
	public:
		ConnectionGene(unsigned long int inGeneId, unsigned long int outGeneId);
		ConnectionGeneParameters getParameters() const;
	};
}
