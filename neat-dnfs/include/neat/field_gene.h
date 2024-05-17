#pragma once

#include <elements/element_factory.h>
#include "constants.h"

namespace neat_dnfs
{
	typedef std::shared_ptr<dnf_composer::element::NeuralField> NeuralFieldPtr;
	typedef std::shared_ptr<dnf_composer::element::Kernel> KernelPtr;

	static uint16_t currentFieldGeneId = 1;

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

		FieldGeneParameters(FieldGeneType type)
			: type(type), id(currentFieldGeneId++)
		{}
	};

	class FieldGene
	{
	private:
		NeuralFieldPtr neuralField;
		KernelPtr kernel;
		FieldGeneParameters parameters;
	public:
		FieldGene(FieldGeneParameters parameters);

		void setAsInput();
		void setAsOutput();
		void setAsHidden();

		void mutate() const;

		FieldGeneParameters getParameters() const;
		std::shared_ptr<dnf_composer::element::NeuralField> getNeuralField() const;
		std::shared_ptr<dnf_composer::element::Kernel> getKernel() const;

		bool operator==(const FieldGene&) const;
	};
}
