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
		unsigned long int innovationNumber;
		bool enabled;

		ConnectionGeneParameters(unsigned long int inGeneId, 
			unsigned long int outGeneId)
			: inGeneId(inGeneId),
			outGeneId(outGeneId),
			innovationNumber(currentInnovationNumber++),
			enabled(true)
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
		std::shared_ptr<dnf_composer::element::Kernel> getKernel() const;
		unsigned long int getInnovationNumber() const;
		unsigned long int getInGeneId() const;
		unsigned long int getOutGeneId() const;
		bool isEnabled() const;
		void mutate();
	};
}
