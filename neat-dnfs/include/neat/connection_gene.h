#pragma once
#include <memory>
#include <random>

#include <dnf_composer/elements/gauss_kernel.h>
#include "tools/utils.h"
#include "constants.h"

namespace neat_dnfs
{
	static uint16_t currentInnovationNumber = 0;

	struct ConnectionTuple
	{
		uint16_t inFieldGeneId;
		uint16_t outFieldGeneId;

		ConnectionTuple(uint16_t inFieldGeneId, uint16_t outFieldGeneId)
			: inFieldGeneId(inFieldGeneId), outFieldGeneId(outFieldGeneId)
		{}

		bool operator==(const ConnectionTuple& other) const
		{
			return inFieldGeneId == other.inFieldGeneId &&
				outFieldGeneId == other.outFieldGeneId;
		}

		bool operator<(const ConnectionTuple& other) const {
			if (inFieldGeneId == other.inFieldGeneId) {
				return outFieldGeneId < other.outFieldGeneId;
			}
			return inFieldGeneId < other.inFieldGeneId;
		}

		std::string toString() const
		{
			return "InFieldGeneId: " + std::to_string(inFieldGeneId) +
				", OutFieldGeneId: " + std::to_string(outFieldGeneId);
		}

		void print() const
		{
			tools::logger::log(tools::logger::INFO, toString());
		}
	};

	struct ConnectionGeneParameters
	{
		ConnectionTuple connectionTuple;
		uint16_t innovationNumber;
		bool enabled;

		ConnectionGeneParameters(ConnectionTuple connectionTuple)
			: connectionTuple(connectionTuple),
			innovationNumber(currentInnovationNumber++), enabled(true)
		{}

		ConnectionGeneParameters(uint16_t inFieldGeneId, uint16_t outFieldGeneId)
			: connectionTuple(inFieldGeneId, outFieldGeneId),
			innovationNumber(currentInnovationNumber++), enabled(true)
		{}

		ConnectionGeneParameters(const ConnectionGeneParameters& other)
			: connectionTuple(other.connectionTuple) ,
					innovationNumber(other.innovationNumber), enabled(other.enabled)
		{}
		
		bool operator==(const ConnectionGeneParameters& other) const
		{
			return connectionTuple == other.connectionTuple &&
				innovationNumber == other.innovationNumber;
		}

		std::string toString() const
		{
			return connectionTuple.toString() +
				", InnovationNumber: " + std::to_string(innovationNumber) +
				", Enabled: " + (enabled ? "true" : "false") + '\n';
		}

		void print() const
		{
			tools::logger::log(tools::logger::INFO, toString());
		}

	};

	class ConnectionGene
	{
	private:
		ConnectionGeneParameters parameters;
		KernelPtr kernel;
	public:
		ConnectionGene(ConnectionTuple connectionTuple);
		ConnectionGene(ConnectionTuple connectionTuple,
			const dnf_composer::element::GaussKernelParameters& gkp);
		ConnectionGene(const ConnectionGeneParameters& parameters, 
			const dnf_composer::element::GaussKernelParameters& gkp);

		void mutate() const;
		void disable();
		void toggle();

		bool isEnabled() const;

		void setInnovationNumber(uint16_t innovationNumber);

		ConnectionGeneParameters getParameters() const;
		KernelPtr getKernel() const;
		uint16_t getInnovationNumber() const;
		uint16_t getInFieldGeneId() const;
		uint16_t getOutFieldGeneId() const;
		double getKernelAmplitude() const;
		double getKernelWidth() const;

		bool operator==(const ConnectionGene&) const;
		bool isCloneOf(const ConnectionGene&) const;
		std::string toString() const;
		void print() const;
		ConnectionGene clone() const;
	};
}
