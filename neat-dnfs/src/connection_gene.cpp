#include "connection_gene.h"


namespace neat_dnfs
{
	ConnectionGene::ConnectionGene(unsigned long int inGeneId, unsigned long int outGeneId)
		: parameters(inGeneId, outGeneId)
	{
		using namespace dnf_composer::element;
		static constexpr bool circularity = false;
		static constexpr bool normalization = false;

		const GaussKernelParameters gkp{ 2, 1, circularity, normalization };
		const ElementCommonParameters gkcp{ "gk cg " + std::to_string(inGeneId) + " - " + std::to_string(outGeneId), {100, 1.0} };
		kernel = std::make_shared<GaussKernel>(gkcp, gkp);
	}

	ConnectionGeneParameters ConnectionGene::getParameters() const
	{
		return parameters;
	}
}
