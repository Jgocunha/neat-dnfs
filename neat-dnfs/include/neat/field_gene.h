#pragma once

#include <elements/element_factory.h>
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
		NeuralFieldPtr neuralField;
		KernelPtr kernel;
		FieldGeneParameters parameters;
	public:
		FieldGene(FieldGeneParameters parameters);
		FieldGene(FieldGeneParameters parameters,
			const NeuralFieldPtr& neuralField, 
			const KernelPtr& kernel);

		void setAsInput();
		void setAsOutput();
		void setAsHidden();

		void mutate() const;

		FieldGeneParameters getParameters() const;
		std::shared_ptr<dnf_composer::element::NeuralField> getNeuralField() const;
		std::shared_ptr<dnf_composer::element::Kernel> getKernel() const;

		bool operator==(const FieldGene&) const;
		bool isCloneOf(const FieldGene&) const;
		std::string toString() const;
		void print() const;
		FieldGene clone() const;
	};
}
