#include "connection_gene.h"


namespace neat_dnfs
{
	ConnectionGene::ConnectionGene(uint16_t inGeneId, uint16_t outGeneId)
		: parameters(inGeneId, outGeneId)
	{
		using namespace dnf_composer::element;
		static constexpr bool circularity = false;
		static constexpr bool normalization = false;
		static constexpr int xSize = 100;
		static constexpr double dx = 1.0;

		const double randomSigmaBetween0And10 = 10.0 * (static_cast<double>(rand()) / RAND_MAX);
		const double randomAmplitudeBetween0And10 = 10.0 * (static_cast<double>(rand()) / RAND_MAX);

		const GaussKernelParameters gkp{ randomSigmaBetween0And10, randomAmplitudeBetween0And10, circularity, normalization };
		const ElementCommonParameters gkcp{ "gk cg " + std::to_string(inGeneId) + " - " + std::to_string(outGeneId), {xSize, dx} };
		kernel = std::make_shared<GaussKernel>(gkcp, gkp);
	}

	ConnectionGene::ConnectionGene(uint16_t inGeneId, uint16_t outGeneId, const dnf_composer::element::GaussKernelParameters& gkp)
		: parameters(inGeneId, outGeneId)
	{
		using namespace dnf_composer::element;
		static constexpr int xSize = 100;
		static constexpr double dx = 1.0;

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

	uint16_t ConnectionGene::getInnovationNumber() const
	{
		return parameters.innovationNumber;
	}

	uint16_t ConnectionGene::getInGeneId() const
	{
		return parameters.inGeneId;
	}

	uint16_t ConnectionGene::getOutGeneId() const
	{
		return parameters.outGeneId;
	}

	bool ConnectionGene::isEnabled() const
	{
		return parameters.enabled;
	}

	void ConnectionGene::mutate()
	{
		using namespace dnf_composer::element;

		GaussKernelParameters gkp = std::dynamic_pointer_cast<GaussKernel>(kernel)->getParameters();

		// Mutate sigma value
		const double sigmaMutation = 0.5 * (2.0 * (static_cast<double>(rand()) / RAND_MAX) - 1.0); // Random number between -0.1 and 0.1
		gkp.sigma += sigmaMutation;
		gkp.sigma = std::max(0.0, std::min(10.0, gkp.sigma)); // Ensure sigma stays within the range of 0 to 10

		// Mutate amplitude value
		const double amplitudeMutation = 0.5 * (2.0 * (static_cast<double>(rand()) / RAND_MAX) - 1.0); // Random number between -0.1 and 0.1
		gkp.amplitude += amplitudeMutation;
		gkp.amplitude = std::max(0.0, std::min(10.0, gkp.amplitude)); // Ensure amplitude stays within the range of 0 to 10

		const ElementCommonParameters gkcp = kernel->getElementCommonParameters();
		kernel = std::make_shared<GaussKernel>(gkcp, gkp);
	}

	void ConnectionGene::setInnovationNumber(uint16_t innovationNumber)
	{
		parameters.innovationNumber = innovationNumber;
	}

	void ConnectionGene::disable()
	{
		parameters.enabled = false;
	}

	void ConnectionGene::toggle()
	{
		parameters.enabled = !parameters.enabled;
	}

}
