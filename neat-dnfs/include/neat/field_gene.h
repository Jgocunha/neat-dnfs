#pragma once

#include "constants.h"

namespace neat_dnfs
{
	/// @brief Role of a field gene within the genome topology.
	enum class FieldGeneType
	{
		INPUT = 1,
		OUTPUT = 2,
		HIDDEN = 3
	};

	struct FieldGeneParameters
	{
		FieldGeneType type;
		int id;

		FieldGeneParameters(const FieldGeneParameters& other) = default;
		FieldGeneParameters(FieldGeneType type, int id);

		bool operator==(const FieldGeneParameters& other) const;
		[[nodiscard]] std::string toString() const;
		void print() const;
	};

	class FieldGene
	{
	private:
		FieldGeneParameters parameters;
		NeuralFieldPtr neuralField;
		KernelPtr kernel;
		NormalNoisePtr noise;
		std::string mutationsInLastGeneration;
	public:
		explicit FieldGene(const FieldGeneParameters& parameters,
		                   const dnf_composer::element::ElementDimensions& dimensions = dnf_composer::element::ElementDimensions{100, 1.0});
		FieldGene(const FieldGeneParameters& parameters,
			const NeuralFieldPtr& neuralField, 
			KernelPtr kernel);
		FieldGene(const FieldGeneParameters& parameters, const FieldGene& other);

		void setAsInput(const dnf_composer::element::ElementDimensions& dimensions);
		void setAsOutput(const dnf_composer::element::ElementDimensions& dimensions);
		void setAsHidden(const dnf_composer::element::ElementDimensions& dimensions);

		void mutate();
		void clearLastMutations();

		[[nodiscard]] FieldGeneParameters getParameters() const;
		[[nodiscard]] std::string getMutationsInLastGeneration() const;
		[[nodiscard]] std::shared_ptr<dnf_composer::element::NeuralField> getNeuralField() const;
		[[nodiscard]] std::shared_ptr<dnf_composer::element::Kernel> getKernel() const;
		[[nodiscard]] std::shared_ptr<dnf_composer::element::NormalNoise> getNoise() const;

		bool operator==(const FieldGene& /*other*/) const;
		/// @brief True if both genes share identical parameters and kernel values (deep equality), unlike @c operator== which compares by id.
		[[nodiscard]] bool isCloneOf(const FieldGene& /*other*/) const;
		[[nodiscard]] std::string toString() const;
		void print() const;
		[[nodiscard]] FieldGene clone() const;
	private:
		void initializeNeuralField(const dnf_composer::element::ElementDimensions& dimensions);
		void initializeKernel(const dnf_composer::element::ElementDimensions& dimensions);
		void initializeGaussKernel(const dnf_composer::element::ElementDimensions& dimensions);
		void initializeMexicanHatKernel(const dnf_composer::element::ElementDimensions& dimensions);
		void initializeNoise(const dnf_composer::element::ElementDimensions& dimensions);

		void mutateKernel();
		void mutateGaussKernel();
		void mutateMexicanHatKernel();

		void mutateKernelType();
		void mutateNeuralField();
	};
}
