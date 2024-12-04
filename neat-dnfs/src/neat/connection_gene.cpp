#include "neat/connection_gene.h"


namespace neat_dnfs
{
	ConnectionGene::ConnectionGene(ConnectionTuple connectionTuple)
		: parameters(connectionTuple)
	{
		using namespace dnf_composer::element;
		using namespace neat_dnfs::tools::utils;

		const FieldCouplingParameters fcp{ {DimensionConstants::xSize, DimensionConstants::dx}, 			FieldCouplingConstants::learningRule, FieldCouplingConstants::strength, FieldCouplingConstants::learningRate};

		const std::string elementName = FieldCouplingConstants::namePrefixConnectionGene +
			std::to_string(connectionTuple.inFieldGeneId) +
			" - " + std::to_string(connectionTuple.outFieldGeneId) + " " +
			std::to_string(parameters.innovationNumber);
		const ElementCommonParameters fccp{ elementName,
			{DimensionConstants::xSize, DimensionConstants::dx} };
		coupling = std::make_unique<FieldCoupling>(fccp, fcp);
	}

	ConnectionGene::ConnectionGene(ConnectionTuple connectionTuple,
		const dnf_composer::element::FieldCouplingParameters& fcp)
		: parameters(connectionTuple)
	{
		using namespace dnf_composer::element;

		const std::string elementName = FieldCouplingConstants::namePrefixConnectionGene +
			std::to_string(connectionTuple.inFieldGeneId) +
			" - " + std::to_string(connectionTuple.outFieldGeneId) + " " +
			std::to_string(parameters.innovationNumber);
		const ElementCommonParameters fccp{ elementName,
			{DimensionConstants::xSize, DimensionConstants::dx} };
		coupling = std::make_unique<FieldCoupling>(fccp, fcp);
	}

	ConnectionGene::ConnectionGene(const ConnectionGeneParameters& parameters,
		const dnf_composer::element::FieldCouplingParameters& fcp)
		: parameters(parameters)
	{
		using namespace dnf_composer::element;

		const std::string elementName = FieldCouplingConstants::namePrefixConnectionGene +
			std::to_string(parameters.connectionTuple.inFieldGeneId) +
			" - " + std::to_string(parameters.connectionTuple.outFieldGeneId) + " " +
			std::to_string(parameters.innovationNumber);
		const ElementCommonParameters fccp{ elementName,
			{DimensionConstants::xSize, DimensionConstants::dx} };
		coupling = std::make_unique<FieldCoupling>(fccp, fcp);
	}

	void ConnectionGene::mutate() const
	{
		using namespace dnf_composer::element;

		const auto targetCoupling = dynamic_cast<FieldCoupling*>(coupling.get());
		if (!targetCoupling)
		{
			const std::string message = "Calling mutate() on ConnectionGene with ConnectionTuple: " +
				std::to_string(parameters.connectionTuple.inFieldGeneId) + " - " +
				std::to_string(parameters.connectionTuple.outFieldGeneId) + " but kernel is not a GaussKernel";
			tools::logger::log(tools::logger::FATAL, message);
			throw std::runtime_error(message);
		}

		using namespace neat_dnfs::tools::utils;
		const double mutationStep = generateRandomDouble(0.1, 1.0);


		FieldCouplingParameters fcp = targetCoupling->getParameters();

		targetCoupling->setParameters(fcp);
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

	FieldCouplingPtr ConnectionGene::getFieldCoupling() const
	{
		return coupling;
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

	double ConnectionGene::getCouplingStrength() const
	{
		return std::dynamic_pointer_cast<dnf_composer::element::FieldCoupling>(coupling)->getParameters().scalar;
	}

	bool ConnectionGene::operator==(const ConnectionGene& other) const
	{
		return parameters.innovationNumber == other.parameters.innovationNumber;
	}

	bool ConnectionGene::isCloneOf(const ConnectionGene& other) const
	{
		const auto fc = dynamic_cast<dnf_composer::element::FieldCoupling*>(coupling.get());
		const dnf_composer::element::FieldCouplingParameters fcp = fc->getParameters();
		const auto other_fc = dynamic_cast<dnf_composer::element::FieldCoupling*>(other.coupling.get());
		const dnf_composer::element::FieldCouplingParameters other_fcp = other_fc->getParameters();
		return parameters.innovationNumber == other.parameters.innovationNumber &&
			parameters.connectionTuple == other.parameters.connectionTuple &&
			parameters.enabled == other.parameters.enabled &&
			fcp == other_fcp;
	}

	std::string ConnectionGene::toString() const
	{
		std::string result = "ConnectionGene: ";
		result += parameters.toString();

		std::stringstream address;
		address << coupling.get();
		result += "Kernel address: " + address.str() + '\n';
		result += coupling->toString();
		return result;
	}

	void ConnectionGene::print() const
	{
		tools::logger::log(tools::logger::INFO, toString());
	}

	ConnectionGene ConnectionGene::clone() const
	{
		return ConnectionGene{ parameters, std::dynamic_pointer_cast<dnf_composer::element::FieldCoupling>(coupling)->getParameters() };
	}
}