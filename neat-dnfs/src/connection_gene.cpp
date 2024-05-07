#include "connection_gene.h"


namespace neat_dnfs
{
	ConnectionGene::ConnectionGene(ConnectionTuple connectionTuple)
		: parameters(connectionTuple)
	{
		using namespace dnf_composer::element;
		static constexpr bool circularity = false;
		static constexpr bool normalization = false;
		static constexpr int xSize = 100;
		static constexpr double dx = 1.0;

		const double randomSigmaBetween0And10 = 10.0 * (static_cast<double>(rand()) / RAND_MAX);
		const double randomAmplitudeBetween0And10 = 10.0 * (static_cast<double>(rand()) / RAND_MAX);

		const GaussKernelParameters gkp{ randomSigmaBetween0And10,
			randomAmplitudeBetween0And10,
			circularity, normalization };
		const ElementCommonParameters gkcp{ "gk cg " + std::to_string(connectionTuple.inFieldGeneId) + 
			" - " + std::to_string(connectionTuple.outFieldGeneId), {xSize, dx} };
		kernel = std::make_shared<GaussKernel>(gkcp, gkp);
	}

	ConnectionGene::ConnectionGene(ConnectionTuple connectionTuple,
		const dnf_composer::element::GaussKernelParameters& gkp)
		: parameters(connectionTuple)
	{
		using namespace dnf_composer::element;
		static constexpr int xSize = 100;
		static constexpr double dx = 1.0;

		const ElementCommonParameters gkcp{ "gk cg " + std::to_string(connectionTuple.inFieldGeneId) + 
			" - " + std::to_string(connectionTuple.outFieldGeneId), {xSize, dx} };
		kernel = std::make_shared<GaussKernel>(gkcp, gkp);
	}

	void ConnectionGene::mutate()
	{
		using namespace dnf_composer::element;

		GaussKernelParameters gkp = 
			std::dynamic_pointer_cast<GaussKernel>(kernel)->getParameters();

		// Mutate sigma value
		const double sigmaMutation =
			0.5 * (2.0 * (static_cast<double>(rand()) / RAND_MAX) - 1.0); 
		gkp.sigma += sigmaMutation;
		gkp.sigma = std::max(0.0, std::min(10.0, gkp.sigma)); 

		// Mutate amplitude value
		const double amplitudeMutation = 
			0.5 * (2.0 * (static_cast<double>(rand()) / RAND_MAX) - 1.0); 
		gkp.amplitude += amplitudeMutation;
		gkp.amplitude = std::max(0.0, std::min(10.0, gkp.amplitude)); 

		const ElementCommonParameters gkcp = kernel->getElementCommonParameters();
		kernel = std::make_shared<GaussKernel>(gkcp, gkp);
	}

	void ConnectionGene::disable()
	{
		parameters.enabled = false;
	}

	void ConnectionGene::toggle()
	{
		parameters.enabled = !parameters.enabled;
	}

	bool ConnectionGene::isEnabled() const
	{
		return parameters.enabled;
	}

	void ConnectionGene::setInnovationNumber(uint16_t innovationNumber)
	{
		parameters.innovationNumber = innovationNumber;
	}

	ConnectionGeneParameters ConnectionGene::getParameters() const
	{
		return parameters;
	}

	KernelPtr ConnectionGene::getKernel() const
	{
		return kernel;
	}

	uint16_t ConnectionGene::getInnovationNumber() const
	{
		return parameters.innovationNumber;
	}

	uint16_t ConnectionGene::getInFieldGeneId() const
	{
		return parameters.connectionTuple.inFieldGeneId;
	}

	uint16_t ConnectionGene::getOutFieldGeneId() const
	{
		return parameters.connectionTuple.outFieldGeneId;
	}

	double ConnectionGene::getKernelAmplitude() const
	{
		return std::dynamic_pointer_cast<dnf_composer::element::GaussKernel>(kernel)->getParameters().amplitude;
	}

	double ConnectionGene::getKernelWidth() const
	{
		return std::dynamic_pointer_cast<dnf_composer::element::GaussKernel>(kernel)->getParameters().sigma;
	}
}
