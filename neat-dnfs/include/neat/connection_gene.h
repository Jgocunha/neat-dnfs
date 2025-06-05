#pragma once
#include <memory>
#include <random>

#include <dnf_composer/elements/field_coupling.h>
#include "tools/utils.h"
#include "constants.h"

namespace neat_dnfs
{
	struct ConnectionTuple
	{
		int inFieldGeneId;
		int outFieldGeneId;

		ConnectionTuple(int inFieldGeneId, int outFieldGeneId);
		bool operator==(const ConnectionTuple& other) const;
		bool operator<(const ConnectionTuple& other) const;
		[[nodiscard]] std::string toString() const;
		void print() const;
	};

	struct ConnectionGeneParameters
	{
		ConnectionTuple connectionTuple;
		uint16_t innovationNumber;
		bool enabled;

		explicit ConnectionGeneParameters(ConnectionTuple connectionTuple);
		ConnectionGeneParameters(int inFieldGeneId, uint16_t outFieldGeneId, int innov);
		ConnectionGeneParameters(const ConnectionGeneParameters& other);
		bool operator==(const ConnectionGeneParameters& other) const;
		[[nodiscard]] std::string toString() const;
		void print() const;
	};

	class ConnectionGene
	{
	private:
		//static inline int fc_id_count = 0;
		//int fc_id;
		ConnectionGeneParameters parameters;
		FieldCouplingPtr coupling;
	public:
		ConnectionGene(ConnectionTuple connectionTuple, int innov);

		ConnectionGene(ConnectionTuple connectionTuple, int innov,
			const dnf_composer::element::ElementDimensions& inputFieldDimensions,
			const dnf_composer::element::ElementDimensions& outputFieldDimensions);

		ConnectionGene(ConnectionTuple connectionTuple,
			const dnf_composer::element::FieldCouplingParameters& fcp,
			const dnf_composer::element::ElementDimensions& outputFieldDimensions);

		ConnectionGene(const ConnectionGeneParameters& parameters,
			const dnf_composer::element::FieldCouplingParameters& fcp,
			const dnf_composer::element::ElementDimensions& outputFieldDimensions);

		void mutate() const;
		void disable();
		void toggle();

		[[nodiscard]] bool isEnabled() const;

		void setInnovationNumber(int innovationNumber);

		[[nodiscard]] ConnectionGeneParameters getParameters() const;
		[[nodiscard]] FieldCouplingPtr getFieldCoupling() const;
		[[nodiscard]] int getInnovationNumber() const;
		[[nodiscard]] int getInFieldGeneId() const;
		[[nodiscard]] int getOutFieldGeneId() const;
		[[nodiscard]] double getCouplingStrength() const;

		bool operator==(const ConnectionGene&) const;
		[[nodiscard]] bool isCloneOf(const ConnectionGene&) const;
		[[nodiscard]] std::string toString() const;
		void print() const;
		[[nodiscard]] ConnectionGene clone() const;
	};
}
