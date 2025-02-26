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

		ConnectionTuple(int inFieldGeneId, int outFieldGeneId);
		bool operator==(const ConnectionTuple& other) const;
		bool operator<(const ConnectionTuple& other) const;
		std::string toString() const;
		void print() const;
	};

	struct ConnectionGeneParameters
	{
		ConnectionTuple connectionTuple;
		int innovationNumber;
		bool enabled;

		ConnectionGeneParameters(ConnectionTuple connectionTuple, int innov);
		ConnectionGeneParameters(int inFieldGeneId, int outFieldGeneId, int innov);
		ConnectionGeneParameters(const ConnectionGeneParameters& other) = default;
		bool operator==(const ConnectionGeneParameters& other) const;
		std::string toString() const;
		void print() const;
	};

	struct ConnectionGeneStatistics
	{
		int numKernelMutationsPerGeneration = 0;
		int numKernelTypeMutationsPerGeneration = 0;
		int numGaussKernelMutationsPerGeneration = 0;
		int numMexicanHatKernelMutationsPerGeneration = 0;
		int numOscillatoryKernelMutationsPerGeneration = 0;
		int numConnectionSignalMutationsPerGeneration = 0;

		int numKernelMutationsTotal = 0;
		int numKernelTypeMutationsTotal = 0;
		int numGaussKernelMutationsTotal = 0;
		int numMexicanHatKernelMutationsTotal = 0;
		int numOscillatoryKernelMutationsTotal = 0;
		int numConnectionSignalMutationsTotal = 0;

		ConnectionGeneStatistics() = default;
		void resetPerGenerationStatistics();
		std::string toString() const;
		void print() const;
		void savePerGeneration(const std::string& directory) const;
		void saveTotal(const std::string& directory) const;
	};


	class ConnectionGene
	{
	private:
		ConnectionGeneParameters parameters;
		static ConnectionGeneStatistics statistics;
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

		ConnectionGene(ConnectionTuple connectionTuple, int innov, KernelPtr kernel);

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
		static void resetMutationStatisticsPerGeneration();
		static ConnectionGeneStatistics getStatistics();

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
