#include "neat/connection_gene.h"


namespace neat_dnfs
{
	ConnectionTuple::ConnectionTuple(int inFieldGeneId, int outFieldGeneId)
		: inFieldGeneId(inFieldGeneId), outFieldGeneId(outFieldGeneId)
	{}

	bool ConnectionTuple::operator==(const ConnectionTuple& other) const
	{
		return inFieldGeneId == other.inFieldGeneId && outFieldGeneId == other.outFieldGeneId;
	}

	bool ConnectionTuple::operator<(const ConnectionTuple& other) const {
		if (inFieldGeneId == other.inFieldGeneId)
			return outFieldGeneId < other.outFieldGeneId;
		return inFieldGeneId < other.inFieldGeneId;
	}

	std::string ConnectionTuple::toString() const
	{
		return std::to_string(inFieldGeneId) + "-" + std::to_string(outFieldGeneId);
	}

	void ConnectionTuple::print() const
	{
		tools::logger::log(tools::logger::INFO, toString());
	}

	ConnectionGeneParameters::ConnectionGeneParameters(ConnectionTuple connectionTuple, int innov)
		: connectionTuple(connectionTuple), innovationNumber(innov), enabled(true)
	{}

	ConnectionGeneParameters::ConnectionGeneParameters(int inFieldGeneId, int outFieldGeneId, int innov)
		: connectionTuple(inFieldGeneId, outFieldGeneId), innovationNumber(innov), enabled(true)
	{}

	bool ConnectionGeneParameters::operator==(const ConnectionGeneParameters& other) const
	{
		return connectionTuple == other.connectionTuple &&
			innovationNumber == other.innovationNumber;
	}

	std::string ConnectionGeneParameters::toString() const
	{
		return connectionTuple.toString() +
			", innov: " + std::to_string(innovationNumber) +
			", enabled: " + (enabled ? "true" : "false");
	}

	void ConnectionGeneParameters::print() const
	{
		tools::logger::log(tools::logger::INFO, toString());
	}

	ConnectionGene::ConnectionGene(ConnectionTuple connectionTuple,
	                               const dnf_composer::element::ElementDimensions& inputFieldDimensions,
	                               const dnf_composer::element::ElementDimensions& outputFieldDimensions)
	: fc_id(fc_id_count++), parameters(connectionTuple)
	{
		using namespace dnf_composer::element;
		using namespace neat_dnfs::tools::utils;

		const double couplingStrength = generateRandomDouble(FieldCouplingConstants::couplingStrengthMinVal, FieldCouplingConstants::couplingStrengthMaxVal);
		const FieldCouplingParameters fcp{ inputFieldDimensions, FieldCouplingConstants::learningRule, couplingStrength, FieldCouplingConstants::learningRate};

		const std::string elementName = FieldCouplingConstants::namePrefixConnectionGene +
			std::to_string(connectionTuple.inFieldGeneId) +
			" - " + std::to_string(connectionTuple.outFieldGeneId) + " " +
			std::to_string(parameters.innovationNumber) + " " + std::to_string(fc_id);
		const ElementCommonParameters fccp{ elementName,
			outputFieldDimensions };
		coupling = std::make_unique<FieldCoupling>(fccp, fcp);
	}

	ConnectionGene::ConnectionGene(ConnectionTuple connectionTuple,
		const dnf_composer::element::FieldCouplingParameters& fcp, 
		const dnf_composer::element::ElementDimensions& outputFieldDimensions)
		: fc_id(fc_id_count++), parameters(connectionTuple)
	{
		using namespace dnf_composer::element;

		const std::string elementName = FieldCouplingConstants::namePrefixConnectionGene +
			std::to_string(connectionTuple.inFieldGeneId) +
			" - " + std::to_string(connectionTuple.outFieldGeneId) + " " +
			std::to_string(parameters.innovationNumber) + " " + std::to_string(fc_id);
		const ElementCommonParameters fccp{ elementName, outputFieldDimensions };
		coupling = std::make_unique<FieldCoupling>(fccp, fcp);
	}

	ConnectionGene::ConnectionGene(const ConnectionGeneParameters& parameters,
		const dnf_composer::element::FieldCouplingParameters& fcp,
		const dnf_composer::element::ElementDimensions& outputFieldDimensions)
		: fc_id(fc_id_count++), parameters(parameters)
	{
		using namespace dnf_composer::element;

		const std::string elementName = FieldCouplingConstants::namePrefixConnectionGene +
			std::to_string(parameters.connectionTuple.inFieldGeneId) +
			" - " + std::to_string(parameters.connectionTuple.outFieldGeneId) + " " +
			std::to_string(parameters.innovationNumber) + " " + std::to_string(fc_id);
		const ElementCommonParameters fccp{ elementName, outputFieldDimensions };
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
				std::to_string(parameters.connectionTuple.outFieldGeneId) + " but coupling is nullptr.";
			tools::logger::log(tools::logger::FATAL, message);
			throw std::runtime_error(message);
		}

		using namespace neat_dnfs::tools::utils;
		const double mutationSignal = generateRandomSignal();

		FieldCouplingParameters fcp = targetCoupling->getParameters();

		fcp.scalar = std::clamp(fcp.scalar + mutationSignal * FieldCouplingConstants::couplingStrengthStep, 
			FieldCouplingConstants::couplingStrengthMinVal, FieldCouplingConstants::couplingStrengthMaxVal);

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
		result += "Coupling address: " + address.str() + '\n';
		result += coupling->toString();
		return result;
	}

	void ConnectionGene::print() const
	{
		tools::logger::log(tools::logger::INFO, toString());
	}

	ConnectionGene ConnectionGene::clone() const
	{
		const auto fcp = std::dynamic_pointer_cast<dnf_composer::element::FieldCoupling>(coupling)->getParameters();
		const auto outputFieldDimensions = std::dynamic_pointer_cast<dnf_composer::element::FieldCoupling>(coupling)->getElementCommonParameters().dimensionParameters;

		return ConnectionGene{ parameters,
			std::dynamic_pointer_cast<dnf_composer::element::FieldCoupling>(coupling)->getParameters(), outputFieldDimensions};
	}
}