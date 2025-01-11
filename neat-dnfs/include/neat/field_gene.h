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
		uint16_t id;

		FieldGeneParameters(const FieldGeneParameters& other)
			: type(other.type), id(other.id)
		{}

		FieldGeneParameters(FieldGeneType type, uint16_t id)
			: type(type), id(id)
		{}

		bool operator==(const FieldGeneParameters& other) const
		{
			return type == other.type && id == other.id;
		}

		std::string toString() const
		{
			std::string typeStr;
			switch (type)
			{
			case FieldGeneType::INPUT:
				typeStr = "INPUT";
				break;
			case FieldGeneType::OUTPUT:
				typeStr = "OUTPUT";
				break;
			case FieldGeneType::HIDDEN:
				typeStr = "HIDDEN";
				break;
			}
			return "FieldGeneParameters{type=" + typeStr + ", id=" + std::to_string(id) + "}" + '\n';
		}

		void print() const
		{
			tools::logger::log(tools::logger::INFO, toString());
		}
	};

	class FieldGene
	{
	private:
		FieldGeneParameters parameters;
		NeuralFieldPtr neuralField;
		KernelPtr kernel;
	public:
		FieldGene(const FieldGeneParameters& parameters, 
			const dnf_composer::element::ElementDimensions& dimensions = {100, 1.0});
		FieldGene(const FieldGeneParameters& parameters,
			const NeuralFieldPtr& neuralField, 
			const KernelPtr& kernel);

		void setAsInput(const dnf_composer::element::ElementDimensions& dimensions);
		void setAsOutput(const dnf_composer::element::ElementDimensions& dimensions);
		void setAsHidden(const dnf_composer::element::ElementDimensions& dimensions);

		void mutate();

		FieldGeneParameters getParameters() const;
		std::shared_ptr<dnf_composer::element::NeuralField> getNeuralField() const;
		std::shared_ptr<dnf_composer::element::Kernel> getKernel() const;

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
	};
}
