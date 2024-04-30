#include "gene.h"

namespace neat_dnfs
{
	Gene::Gene(GeneParameters parameters)
		: parameters(parameters)
	{
		neuralField = nullptr;
		switch (parameters.type)
		{
		case GeneType::INPUT:
			setAsInput();
			break;
		case GeneType::OUTPUT:
			setAsOutput();
			break;
		case GeneType::HIDDEN:
			setAsHidden();
			break;
		}
	}

	void Gene::setAsInput()
	{
		using namespace dnf_composer::element;
		static constexpr bool circularity = false;
		static constexpr bool normalization = false;
		static constexpr int xSize = 100;
		static constexpr double dx = 1.0;

		// for input neural fields use gauss kernel
		// this allows for multiple bumps of activation
		parameters.type = GeneType::INPUT;
		const NeuralFieldParameters nfp{25, -10, HeavisideFunction{0.0}};
		const ElementCommonParameters nfcp{ "nf " + std::to_string(parameters.id), {xSize, dx} };
		const GaussKernelParameters gkp{ 2, 1, circularity, normalization };
		const ElementCommonParameters gkcp{ "gk " + std::to_string(parameters.id), {xSize, dx} };
		neuralField = std::make_shared<NeuralField>(nfcp, nfp);
		kernel = std::make_shared<GaussKernel>(gkcp, gkp);
	}

	void Gene::setAsOutput()
	{
		using namespace dnf_composer::element;
		static constexpr bool circularity = false;
		static constexpr bool normalization = false;
		static constexpr int xSize = 100;
		static constexpr double dx = 1.0;

		// for output neural fields use lateral interactions self-excitation kernel
		// with global inhibition to exhibit selection behavior
		parameters.type = GeneType::OUTPUT;
		const NeuralFieldParameters nfp{ 25, -10, HeavisideFunction{0.0} };
		const ElementCommonParameters nfcp{ "nf " + std::to_string(parameters.id), {xSize, dx} };
		const LateralInteractionsParameters lip{ 5.3,7.4,6,6, -0.55, circularity, normalization };
		const ElementCommonParameters licp{ "lik " + std::to_string(parameters.id), {xSize, dx} };
		neuralField = std::make_shared<NeuralField>(nfcp, nfp);
		kernel = std::make_shared<LateralInteractions>(licp, lip);
	}

	void Gene::setAsHidden()
	{
		using namespace dnf_composer::element;
		static constexpr bool circularity = false;
		static constexpr bool normalization = false;
		static constexpr int xSize = 100;
		static constexpr double dx = 1.0;

		// for input neural fields use gauss kernel initially
		// this can be changed later through mutation
		parameters.type = GeneType::HIDDEN;
		const NeuralFieldParameters nfp{ 25, -10, HeavisideFunction{0.0} };
		const ElementCommonParameters nfcp{ "nf " + std::to_string(parameters.id), {xSize, dx} };
		const GaussKernelParameters gkp{ 2, 1, circularity, normalization };
		const ElementCommonParameters gkcp{ "gk " + std::to_string(parameters.id), {xSize, dx} };
		neuralField = std::make_shared<NeuralField>(nfcp, nfp);
		kernel = std::make_shared<GaussKernel>(gkcp, gkp);
	}

	void Gene::mutate()
	{
		using namespace dnf_composer::element;

		GaussKernelParameters gkp = std::dynamic_pointer_cast<GaussKernel>(kernel)->getParameters();

		// Mutate sigma value
		const double sigmaMutation = 0.5 * (2.0 * (static_cast<double>(rand()) / RAND_MAX) - 1.0); // Random number between -0.1 and 0.1
		gkp.sigma += sigmaMutation;
		gkp.sigma = std::max(0.0, std::min(10.0, gkp.sigma)); // Ensure sigma stays within the range of 0 to 10

		// Mutate amplitude value
		const double amplitudeMutation = 0.5 * (2.0 * (static_cast<double>(rand()) / RAND_MAX) - 1.0); // Random number between -0.1 and 0.1
		gkp.amplitude += amplitudeMutation;
		gkp.amplitude = std::max(0.0, std::min(10.0, gkp.amplitude)); // Ensure amplitude stays within the range of 0 to 10

		std::dynamic_pointer_cast<GaussKernel>(kernel)->setParameters(gkp);
	}

	GeneParameters Gene::getParameters() const
	{
		return parameters;
	}

	std::shared_ptr<dnf_composer::element::NeuralField> Gene::getNeuralField() const
	{
		return neuralField;
	}

	std::shared_ptr<dnf_composer::element::Kernel> Gene::getKernel() const
	{
		return kernel;
	}
}