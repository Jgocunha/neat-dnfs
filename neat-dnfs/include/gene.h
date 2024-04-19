#pragma once

#include <dnf-composer/elements/element_factory.h>


namespace neat_dnfs
{
	enum class GeneType
	{
		INPUT = 1,
		OUTPUT = 2,
		HIDDEN = 3
	};

	struct GeneParameters
	{
		GeneType type;
		unsigned long int id;

		GeneParameters(GeneType type, unsigned long int id)
			: type(type),
			id(id)
		{}
	};

	class Gene
	{
	private:
		std::shared_ptr<dnf_composer::element::NeuralField> neuralField;
		std::shared_ptr<dnf_composer::element::Kernel> kernel;
		GeneParameters parameters;
	public:
		Gene(GeneParameters parameters);

		void setAsInput();
		void setAsOutput();
		void setAsHidden();

		GeneParameters getParameters() const;
		std::shared_ptr<dnf_composer::element::NeuralField> getNeuralField() const;
		std::shared_ptr<dnf_composer::element::Kernel> getKernel() const;
	};
}
