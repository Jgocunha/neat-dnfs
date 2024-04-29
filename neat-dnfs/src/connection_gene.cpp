#include "connection_gene.h"


namespace neat_dnfs
{
	ConnectionGene::ConnectionGene(unsigned long int inGeneId, unsigned long int outGeneId)
		: parameters(inGeneId, outGeneId)
	{
		using namespace dnf_composer::element;
		static constexpr bool circularity = false;
		static constexpr bool normalization = false;
		static constexpr int xSize = 100;
		static constexpr double dx = 1.0;

		const GaussKernelParameters gkp{ 2, 5, circularity, normalization };
		const ElementCommonParameters gkcp{ "gk cg " + std::to_string(inGeneId) + " - " + std::to_string(outGeneId), {xSize, dx} };
		kernel = std::make_shared<GaussKernel>(gkcp, gkp);
	}

	ConnectionGeneParameters ConnectionGene::getParameters() const
	{
		return parameters;
	}

	std::shared_ptr<dnf_composer::element::Kernel> ConnectionGene::getKernel() const
	{
		return kernel;
	}

	unsigned long int ConnectionGene::getInnovationNumber() const
	{
		return parameters.innovationNumber;
	}

	unsigned long int ConnectionGene::getInGeneId() const
	{
		return parameters.inGeneId;
	}

	unsigned long int ConnectionGene::getOutGeneId() const
	{
		return parameters.outGeneId;
	}

}
