#include "neat/connection_gene.h"


namespace neat_dnfs
{
	ConnectionGene::ConnectionGene(ConnectionTuple connectionTuple)
		: parameters(connectionTuple)
	{
		using namespace dnf_composer::element;
		using namespace neat_dnfs::tools::utils;

		const double randomWidthBetween0And10 = generateRandomDouble(GaussKernelConstants::initialWidthMin,
			GaussKernelConstants::initialWidthMax);
		const double randomAmplitudeBetween0And10 = generateRandomDouble(GaussKernelConstants::initialAmplitudeMin,
			GaussKernelConstants::initialAmplitudeMax);

		const GaussKernelParameters gkp{ randomWidthBetween0And10,
										randomAmplitudeBetween0And10,
									KernelConstants::circularity,
									KernelConstants::normalization };
		const std::string elementName = GaussKernelConstants::namePrefixConnectionGene +
			std::to_string(connectionTuple.inFieldGeneId) +
			" - " + std::to_string(connectionTuple.outFieldGeneId) + " " +
			std::to_string(parameters.innovationNumber);
		const ElementCommonParameters gkcp{ elementName,
			{DimensionConstants::xSize, DimensionConstants::dx} };
		kernel = std::make_shared<GaussKernel>(gkcp, gkp);
	}

	ConnectionGene::ConnectionGene(ConnectionTuple connectionTuple,
		const dnf_composer::element::GaussKernelParameters& gkp)
		: parameters(connectionTuple)
	{
		using namespace dnf_composer::element;

		const std::string elementName = GaussKernelConstants::namePrefixConnectionGene +
			std::to_string(connectionTuple.inFieldGeneId) +
			" - " + std::to_string(connectionTuple.outFieldGeneId) + " " +
			std::to_string(parameters.innovationNumber);
		const ElementCommonParameters gkcp{ elementName,
			{DimensionConstants::xSize, DimensionConstants::dx} };
		kernel = std::make_shared<GaussKernel>(gkcp, gkp);
	}

	void ConnectionGene::mutate() const
	{
		using namespace dnf_composer::element;

		const auto gaussKernel = std::dynamic_pointer_cast<GaussKernel>(kernel);
		if (!gaussKernel)
		{
			const std::string message = "Calling mutate() on ConnectionGene with ConnectionTuple: " +
				std::to_string(parameters.connectionTuple.inFieldGeneId) + " - " +
				std::to_string(parameters.connectionTuple.outFieldGeneId) + " but kernel is not a GaussKernel";
			tools::logger::log(tools::logger::FATAL, message);
			throw std::runtime_error(message);
		}

		using namespace neat_dnfs::tools::utils;
		const double mutationStep = 
			generateRandomDouble(-MutationConstants::mutationStep, MutationConstants::mutationStep);
		const int mutationSelection = generateRandomInt(0, 1);

		GaussKernelParameters gkp = std::dynamic_pointer_cast<GaussKernel>(kernel)->getParameters();

		if (mutationSelection == 0)
			gkp.width = std::clamp(gkp.width + mutationStep, MutationConstants::minWidth,
				MutationConstants::maxWidth);
		else
			gkp.amplitude = std::clamp(gkp.amplitude + mutationStep, MutationConstants::minAmplitude,
				MutationConstants::maxAmplitude);

		std::dynamic_pointer_cast<GaussKernel>(kernel)->setParameters(gkp);
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
		return std::dynamic_pointer_cast<dnf_composer::element::GaussKernel>(kernel)->getParameters().width;
	}

	bool ConnectionGene::operator==(const ConnectionGene& other) const
	{
		return parameters.innovationNumber == other.parameters.innovationNumber;
	}

	bool ConnectionGene::isCloneOf(const ConnectionGene& other) const
	{
		const auto gk = std::dynamic_pointer_cast<dnf_composer::element::GaussKernel>(kernel);
		const dnf_composer::element::GaussKernelParameters gkp = gk->getParameters();
		const auto other_gk = std::dynamic_pointer_cast<dnf_composer::element::GaussKernel>(other.getKernel());
		const dnf_composer::element::GaussKernelParameters other_gkp = other_gk->getParameters();
		return parameters.innovationNumber == other.parameters.innovationNumber &&
			parameters.connectionTuple == other.parameters.connectionTuple &&
			parameters.enabled == other.parameters.enabled &&
			gkp == other_gkp;
	}

	std::string ConnectionGene::toString() const
	{
		std::string result = "ConnectionGene with innovation number: " + std::to_string(parameters.innovationNumber) +
			" and connection tuple: " + std::to_string(parameters.connectionTuple.inFieldGeneId) + " - " +
			std::to_string(parameters.connectionTuple.outFieldGeneId) + " and enabled: " +
			std::to_string(parameters.enabled) + " and kernel: " + kernel->toString();
		return result;
	}

}