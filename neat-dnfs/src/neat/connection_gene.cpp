#include "neat/connection_gene.h"


namespace neat_dnfs
{
	ConnectionGene::ConnectionGene(ConnectionTuple connectionTuple)
		: parameters(connectionTuple)
	{
		using namespace dnf_composer::element;
		using namespace dnf_composer::tools::utils;

		const double randomSigmaBetween0And10 = generateRandomNumber(GaussKernelConstants::initialSigmaMin, 
																	GaussKernelConstants::initialSigmaMax);
		const double randomAmplitudeBetween0And10 = generateRandomNumber(GaussKernelConstants::initialAmplitudeMin,
																		GaussKernelConstants::initialAmplitudeMax);

		const GaussKernelParameters gkp{	randomSigmaBetween0And10,
										randomAmplitudeBetween0And10,
									KernelConstants::circularity,
									KernelConstants::normalization };
		const ElementCommonParameters gkcp{ GaussKernelConstants::namePrefixConnectionGene + 
														std::to_string(connectionTuple.inFieldGeneId) + 
															" - " + std::to_string(connectionTuple.outFieldGeneId),
			{DimensionConstants::xSize, DimensionConstants::dx} };
		kernel = std::make_shared<GaussKernel>(gkcp, gkp);
	}

	ConnectionGene::ConnectionGene(ConnectionTuple connectionTuple,
		const dnf_composer::element::GaussKernelParameters& gkp)
		: parameters(connectionTuple)
	{
		using namespace dnf_composer::element;

		const ElementCommonParameters gkcp{ GaussKernelConstants::namePrefixConnectionGene +
												std::to_string(connectionTuple.inFieldGeneId) +
													" - " + std::to_string(connectionTuple.outFieldGeneId),
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
			tools::logger::log(tools::logger::ERROR, message);
			return;
		}

		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<> mutationStep(-MutationConstants::mutationStep,
			MutationConstants::mutationStep);
		std::uniform_int_distribution<> mutationSelection(0, 1);

		GaussKernelParameters gkp = std::dynamic_pointer_cast<GaussKernel>(kernel)->getParameters();

		if (mutationSelection(gen) == 0)
			gkp.sigma = std::clamp(gkp.sigma + mutationStep(gen), MutationConstants::minSigma,
				MutationConstants::maxSigma);
		else
			gkp.amplitude = std::clamp(gkp.amplitude + mutationStep(gen), MutationConstants::minAmplitude,
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
		return std::dynamic_pointer_cast<dnf_composer::element::GaussKernel>(kernel)->getParameters().sigma;
	}

	bool ConnectionGene::operator==(const ConnectionGene& other) const
	{
		return parameters.innovationNumber == other.parameters.innovationNumber;
	}
}
