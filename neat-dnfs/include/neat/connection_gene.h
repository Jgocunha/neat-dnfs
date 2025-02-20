#pragma once
#include <memory>
#include <random>

#include <dnf_composer/elements/gauss_kernel.h>
#include <dnf_composer/elements/mexican_hat_kernel.h>
#include <dnf_composer/elements/oscillatory_kernel.h>
#include "tools/utils.h"
#include "constants.h"

namespace neat_dnfs
{
	struct ConnectionTuple
	{
		int inFieldGeneId;
		int outFieldGeneId;

		ConnectionTuple(int inFieldGeneId, int outFieldGeneId)
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
		int innovationNumber;
		bool enabled;

		ConnectionGeneParameters(ConnectionTuple connectionTuple, int innov)
			: connectionTuple(connectionTuple), innovationNumber(innov), enabled(true)
		{}

		ConnectionGeneParameters(int inFieldGeneId, int outFieldGeneId, int innov)
			: connectionTuple(inFieldGeneId, outFieldGeneId), innovationNumber(innov), enabled(true)
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
		ConnectionGene(ConnectionTuple connectionTuple, int innov);

		ConnectionGene(ConnectionTuple connectionTuple, int innov,
			const dnf_composer::element::GaussKernelParameters& gkp);
		ConnectionGene(ConnectionTuple connectionTuple, int innov,
			const dnf_composer::element::MexicanHatKernelParameters& mhkp);
		ConnectionGene(ConnectionTuple connectionTuple, int innov,
			const dnf_composer::element::OscillatoryKernelParameters& osckp);

		ConnectionGene(const ConnectionGeneParameters& parameters,
			const dnf_composer::element::GaussKernelParameters& gkp);
		ConnectionGene(const ConnectionGeneParameters& parameters,
			const dnf_composer::element::MexicanHatKernelParameters& mhkp);
		ConnectionGene(const ConnectionGeneParameters& parameters,
			const dnf_composer::element::OscillatoryKernelParameters& osckp);

		ConnectionGene(ConnectionTuple connectionTuple, int innov, const KernelPtr& kernel);

		void mutate();
		void disable();
		void toggle();

		bool isEnabled() const;

		void setInnovationNumber(int innovationNumber);

		ConnectionGeneParameters getParameters() const;
		KernelPtr getKernel() const;
		int getInnovationNumber() const;
		int getInFieldGeneId() const;
		int getOutFieldGeneId() const;
		double getKernelAmplitude() const;
		double getKernelWidth() const;

		bool operator==(const ConnectionGene&) const;
		bool isCloneOf(const ConnectionGene&) const;
		std::string toString() const;
		void print() const;
		ConnectionGene clone() const;

	private:
		void initializeKernel(const dnf_composer::element::ElementDimensions& dimensions);
		void initializeGaussKernel(const dnf_composer::element::ElementDimensions& dimensions);
		void initializeMexicanHatKernel(const dnf_composer::element::ElementDimensions& dimensions);
		void initializeOscillatoryKernel(const dnf_composer::element::ElementDimensions& dimensions);

		void mutateKernel() const;
		void mutateKernelType();
		void mutateGaussKernel() const;
		void mutateMexicanHatKernel() const;
		void mutateOscillatoryKernel() const;
		void mutateConnectionSignal() const;
	};
}
