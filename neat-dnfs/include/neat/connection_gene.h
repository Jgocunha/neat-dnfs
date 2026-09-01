#pragma once
#include <memory>
#include <random>

#include <dnf_composer/elements/gauss_kernel.h>
#include <dnf_composer/elements/mexican_hat_kernel.h>
#include <dnf_composer/elements/oscillatory_kernel.h>
#include "neat_tools/utils.h"
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
		int innovationNumber;
		bool enabled;

		ConnectionGeneParameters(ConnectionTuple connectionTuple, int innov);
		ConnectionGeneParameters(int inFieldGeneId, int outFieldGeneId, int innov);
		ConnectionGeneParameters(const ConnectionGeneParameters& other) = default;
		ConnectionGeneParameters(ConnectionGeneParameters&& other) = default;
		ConnectionGeneParameters& operator=(const ConnectionGeneParameters& other) = default;
		ConnectionGeneParameters& operator=(ConnectionGeneParameters&& other) = default;
		~ConnectionGeneParameters() = default;
		bool operator==(const ConnectionGeneParameters& other) const;
		[[nodiscard]] std::string toString() const;
		void print() const;
	};

	class ConnectionGene
	{
	private:
		ConnectionGeneParameters parameters;
		KernelPtr kernel;
		std::string mutationsInLastGeneration;
	public:
		ConnectionGene(ConnectionTuple connectionTuple, int innov);

		/// @brief Builds a gauss-kernel connection gene.
		/// @param connectionTuple Source and target field gene ids.
		/// @param innov Innovation number.
		/// @param gkp Gauss kernel parameters.
		/// @param dimensions Field dimensions the coupling kernel must match. Defaults
		/// to the global DimensionConstants; pass the source field's own dimensions when
		/// decoding a phenotype, since a kernel whose size differs from the fields it
		/// joins is silently dropped by dnf_composer when the interaction is created.
		ConnectionGene(ConnectionTuple connectionTuple, int innov,
			const dnf_composer::element::GaussKernelParameters& gkp,
			const dnf_composer::element::ElementDimensions& dimensions =
				dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
		/// @brief Builds a mexican-hat-kernel connection gene.
		/// @param connectionTuple Source and target field gene ids.
		/// @param innov Innovation number.
		/// @param mhkp Mexican hat kernel parameters.
		/// @param dimensions Field dimensions the coupling kernel must match. See the
		/// gauss-kernel overload above for why this matters when decoding a phenotype.
		ConnectionGene(ConnectionTuple connectionTuple, int innov,
			const dnf_composer::element::MexicanHatKernelParameters& mhkp,
			const dnf_composer::element::ElementDimensions& dimensions =
				dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });

		ConnectionGene(const ConnectionGeneParameters& parameters,
			const dnf_composer::element::GaussKernelParameters& gkp);
		ConnectionGene(const ConnectionGeneParameters& parameters,
			const dnf_composer::element::MexicanHatKernelParameters& mhkp);

		ConnectionGene(ConnectionTuple connectionTuple, int innov, KernelPtr kernel);

		void mutate();
		void clearLastMutations();
		void disable();
		void toggle();

		[[nodiscard]] bool isEnabled() const;

		void setInnovationNumber(int innovationNumber);

		[[nodiscard]] ConnectionGeneParameters getParameters() const;
		[[nodiscard]] std::string getMutationsInLastGeneration() const;
		[[nodiscard]] KernelPtr getKernel() const;
		[[nodiscard]] int getInnovationNumber() const;
		[[nodiscard]] int getInFieldGeneId() const;
		[[nodiscard]] int getOutFieldGeneId() const;
		[[nodiscard]] double getKernelAmplitude() const;
		[[nodiscard]] double getKernelWidth() const;

		bool operator==(const ConnectionGene& /*other*/) const;
		[[nodiscard]] bool isCloneOf(const ConnectionGene& /*other*/) const;
		[[nodiscard]] std::string toString() const;
		void print() const;
		[[nodiscard]] ConnectionGene clone() const;
	private:
		void initializeKernel(const dnf_composer::element::ElementDimensions& dimensions);
		void initializeGaussKernel(const dnf_composer::element::ElementDimensions& dimensions);
		void initializeMexicanHatKernel(const dnf_composer::element::ElementDimensions& dimensions);

		void mutateKernel();
		void mutateKernelType();
		void mutateGaussKernel() ;
		void mutateMexicanHatKernel();
		void mutateConnectionSignal();
	};
}
