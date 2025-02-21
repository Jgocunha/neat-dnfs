#pragma once

#include "constants.h"

namespace neat_dnfs
{
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
		std::string toString() const;
		void print() const;
	};

	struct FieldGeneStatistics
	{
		int numNeuralFieldMutationsPerGeneration = 0;
		int numKernelMutationsPerGeneration = 0;
		int numKernelTypeMutationsPerGeneration = 0;
		int numGaussKernelMutationsPerGeneration = 0;
		int numMexicanHatKernelMutationsPerGeneration = 0;
		int numOscillatoryKernelMutationsPerGeneration = 0;

		int numNeuralFieldMutationsTotal = 0;
		int numKernelMutationsTotal = 0;
		int numKernelTypeMutationsTotal = 0;
		int numGaussKernelMutationsTotal = 0;
		int numMexicanHatKernelMutationsTotal = 0;
		int numOscillatoryKernelMutationsTotal = 0;

		FieldGeneStatistics() = default;
		void resetPerGenerationStatistics();
		std::string toString() const;
		void print() const;
	};


	class FieldGene
	{
	private:
		FieldGeneParameters parameters;
		static FieldGeneStatistics statistics;
		NeuralFieldPtr neuralField;
		KernelPtr kernel;
		NormalNoisePtr noise;
	public:
		FieldGene(const FieldGeneParameters& parameters, 
			const dnf_composer::element::ElementDimensions& dimensions = {100, 1.0});
		FieldGene(const FieldGeneParameters& parameters,
			const NeuralFieldPtr& neuralField, 
			KernelPtr kernel);

		void setAsInput(const dnf_composer::element::ElementDimensions& dimensions);
		void setAsOutput(const dnf_composer::element::ElementDimensions& dimensions);
		void setAsHidden(const dnf_composer::element::ElementDimensions& dimensions);

		void mutate();

		FieldGeneParameters getParameters() const;
		std::shared_ptr<dnf_composer::element::NeuralField> getNeuralField() const;
		std::shared_ptr<dnf_composer::element::Kernel> getKernel() const;
		std::shared_ptr<dnf_composer::element::NormalNoise> getNoise() const;
		static void resetMutationStatisticsPerGeneration();
		static FieldGeneStatistics getStatistics();

		bool operator==(const FieldGene&) const;
		bool isCloneOf(const FieldGene&) const;
		std::string toString() const;
		void print() const;
		FieldGene clone() const;
	private:
		void initializeNeuralField(const dnf_composer::element::ElementDimensions& dimensions);
		void initializeKernel(const dnf_composer::element::ElementDimensions& dimensions);
		void initializeGaussKernel(const dnf_composer::element::ElementDimensions& dimensions);
		void initializeMexicanHatKernel(const dnf_composer::element::ElementDimensions& dimensions);
		void initializeOscillatoryKernel(const dnf_composer::element::ElementDimensions& dimensions);
		void initializeNoise(const dnf_composer::element::ElementDimensions& dimensions);

		void mutateKernel() const;
		void mutateGaussKernel() const;
		void mutateMexicanHatKernel() const;
		void mutateOscillatoryKernel() const;

		void mutateKernelType();
		void mutateNeuralField();
	};
}
