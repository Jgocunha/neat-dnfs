#pragma once
#include <memory>
#include <random>

#include <elements/gauss_kernel.h>


namespace neat_dnfs
{
	static uint16_t currentInnovationNumber = 1;

	struct ConnectionGeneParameters
	{
		uint16_t inGeneId;
		uint16_t outGeneId;
		uint16_t innovationNumber;
		bool enabled;

		ConnectionGeneParameters(uint16_t inGeneId, 
			uint16_t outGeneId)
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
		ConnectionGene(uint16_t inGeneId, uint16_t outGeneId);
		ConnectionGene(uint16_t inGeneId, uint16_t outGeneId, const dnf_composer::element::GaussKernelParameters& gkp);
		ConnectionGeneParameters getParameters() const;
		std::shared_ptr<dnf_composer::element::Kernel> getKernel() const;
		uint16_t getInnovationNumber() const;
		uint16_t getInGeneId() const;
		uint16_t getOutGeneId() const;
		bool isEnabled() const;
		void mutate();
		void setInnovationNumber(uint16_t innovationNumber);
		void disable();
		void toggle();
	};
}
