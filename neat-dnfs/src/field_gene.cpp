#include "field_gene.h"

namespace neat_dnfs
{
	FieldGene::FieldGene(FieldGeneParameters parameters)
		: parameters(parameters)
	{
		neuralField = nullptr;
		switch (parameters.type)
		{
		case FieldGeneType::INPUT:
			setAsInput();
			break;
		case FieldGeneType::OUTPUT:
			setAsOutput();
			break;
		case FieldGeneType::HIDDEN:
			setAsHidden();
			break;
		}
	}

	void FieldGene::setAsInput()
	{
		using namespace dnf_composer::element;
		static constexpr bool circularity = false;
		static constexpr bool normalization = false;
		static constexpr int xSize = 100;
		static constexpr double dx = 1.0;

		// for input neural fields use gauss kernel
		// this allows for multiple bumps of activation
		parameters.type = FieldGeneType::INPUT;
		const NeuralFieldParameters nfp{25, -10,
			HeavisideFunction{0.0}};
		const ElementCommonParameters nfcp{ "nf " + 
			std::to_string(parameters.id), {xSize, dx} };
		const GaussKernelParameters gkp{ 2, 1,
			circularity, normalization };
		const ElementCommonParameters gkcp{ "gk " + 
			std::to_string(parameters.id), {xSize, dx} };
		neuralField = std::make_shared<NeuralField>(nfcp, nfp);
		kernel = std::make_shared<GaussKernel>(gkcp, gkp);
	}

	void FieldGene::setAsOutput()
	{
		using namespace dnf_composer::element;
		static constexpr bool circularity = false;
		static constexpr bool normalization = false;
		static constexpr int xSize = 100;
		static constexpr double dx = 1.0;

		// for output neural fields use lateral interactions self-excitation kernel
		// with global inhibition to exhibit selection behavior
		parameters.type = FieldGeneType::OUTPUT;
		const NeuralFieldParameters nfp{ 25, -10,
			HeavisideFunction{0.0} };
		const ElementCommonParameters nfcp{ "nf " + 
			std::to_string(parameters.id), {xSize, dx} };
		const LateralInteractionsParameters lip{ 5.3,7.4,6,
			6, -0.55, circularity, normalization };
		const ElementCommonParameters licp{ "lik " + 
			std::to_string(parameters.id), {xSize, dx} };
		neuralField = std::make_shared<NeuralField>(nfcp, nfp);
		kernel = std::make_shared<LateralInteractions>(licp, lip);
	}

	void FieldGene::setAsHidden()
	{
		using namespace dnf_composer::element;
		static constexpr bool circularity = false;
		static constexpr bool normalization = false;
		static constexpr int xSize = 100;
		static constexpr double dx = 1.0;

		// for input neural fields use gauss kernel initially
		// this can be changed later through mutation
		parameters.type = FieldGeneType::HIDDEN;
		const NeuralFieldParameters nfp{ 25, -10,
			HeavisideFunction{0.0} };
		const ElementCommonParameters nfcp{ "nf " + 
			std::to_string(parameters.id), {xSize, dx} };
		const GaussKernelParameters gkp{ 2, 1,
			circularity, normalization };
		const ElementCommonParameters gkcp{ "gk " + 
			std::to_string(parameters.id), {xSize, dx} };
		neuralField = std::make_shared<NeuralField>(nfcp, nfp);
		kernel = std::make_shared<GaussKernel>(gkcp, gkp);
	}

	void FieldGene::mutate() const
	{
		using namespace dnf_composer::element;

		GaussKernelParameters gkp = 
			std::dynamic_pointer_cast<GaussKernel>(kernel)->getParameters();

		// Mutate sigma value
		const double sigmaMutation
			= 0.5 * (2.0 * (static_cast<double>(rand()) / RAND_MAX) - 1.0); 
		gkp.sigma += sigmaMutation;
		gkp.sigma = std::max(0.0, std::min(10.0, gkp.sigma)); 

		// Mutate amplitude value
		const double amplitudeMutation
			= 0.5 * (2.0 * (static_cast<double>(rand()) / RAND_MAX) - 1.0); 
		gkp.amplitude += amplitudeMutation;
		gkp.amplitude = std::max(0.0, std::min(10.0, gkp.amplitude));

		std::dynamic_pointer_cast<GaussKernel>(kernel)->setParameters(gkp);
	}

	FieldGeneParameters FieldGene::getParameters() const
	{
		return parameters;
	}

	NeuralFieldPtr FieldGene::getNeuralField() const
	{
		return neuralField;
	}

	KernelPtr FieldGene::getKernel() const
	{
		return kernel;
	}
}