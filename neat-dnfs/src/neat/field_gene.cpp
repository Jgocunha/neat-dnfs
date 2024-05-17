#include "neat/field_gene.h"

namespace neat_dnfs
{
	FieldGene::FieldGene(FieldGeneParameters parameters)
		: parameters(parameters)
	{
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

		parameters.type = FieldGeneType::INPUT;
		const NeuralFieldParameters nfp{	NeuralFieldConstants::tau,
											NeuralFieldConstants::restingLevel,
											NeuralFieldConstants::activationFunction};
		const ElementCommonParameters nfcp{	NeuralFieldConstants::namePrefix + std::to_string(parameters.id),
						{DimensionConstants::xSize,
											DimensionConstants::dx}
											};
		const GaussKernelParameters gkp{	GaussKernelConstants::sigma,
										GaussKernelConstants::amplitude,
									KernelConstants::circularity,
									KernelConstants::normalization };
		const ElementCommonParameters gkcp{	GaussKernelConstants::namePrefix + std::to_string(parameters.id),
						{DimensionConstants::xSize,
											DimensionConstants::dx}
											};
		neuralField = std::make_shared<NeuralField>(nfcp, nfp);
		kernel = std::make_shared<GaussKernel>(gkcp, gkp);
	}

	void FieldGene::setAsOutput()
	{
		using namespace dnf_composer::element;

		parameters.type = FieldGeneType::OUTPUT;
		const NeuralFieldParameters nfp{ NeuralFieldConstants::tau,
									NeuralFieldConstants::restingLevel,
									NeuralFieldConstants::activationFunction };
		const ElementCommonParameters nfcp{ NeuralFieldConstants::namePrefix + std::to_string(parameters.id),
						{DimensionConstants::xSize,
											DimensionConstants::dx}
		};
		const LateralInteractionsParameters lip{	LateralInteractionsConstants::sigmaExc,
													LateralInteractionsConstants::amplitudeExc,
													LateralInteractionsConstants::sigmaInh,
											LateralInteractionsConstants::amplitudeExc,
													LateralInteractionsConstants::amplitudeGlobal,
											KernelConstants::circularity,
											KernelConstants::normalization };
		const ElementCommonParameters licp{ LateralInteractionsConstants::namePrefix + std::to_string(parameters.id),
											{	DimensionConstants::xSize,
												DimensionConstants::dx}
											};
		neuralField = std::make_shared<NeuralField>(nfcp, nfp);
		kernel = std::make_shared<LateralInteractions>(licp, lip);
	}

	void FieldGene::setAsHidden()
	{
		using namespace dnf_composer::element;

		parameters.type = FieldGeneType::HIDDEN;
		const NeuralFieldParameters nfp{ NeuralFieldConstants::tau,
											NeuralFieldConstants::restingLevel,
											NeuralFieldConstants::activationFunction };
		const ElementCommonParameters nfcp{ NeuralFieldConstants::namePrefix + std::to_string(parameters.id),
						{DimensionConstants::xSize,
											DimensionConstants::dx}
		};
		const GaussKernelParameters gkp{ GaussKernelConstants::sigma,
										GaussKernelConstants::amplitude,
									KernelConstants::circularity,
									KernelConstants::normalization };
		const ElementCommonParameters gkcp{ GaussKernelConstants::namePrefix + std::to_string(parameters.id),
						{DimensionConstants::xSize,
											DimensionConstants::dx}
		};
		neuralField = std::make_shared<NeuralField>(nfcp, nfp);
		kernel = std::make_shared<GaussKernel>(gkcp, gkp);
	}

	void FieldGene::mutate() const
	{
		using namespace dnf_composer::element;

		const auto gaussKernel = std::dynamic_pointer_cast<GaussKernel>(kernel);
		if (!gaussKernel) 
		{
			const std::string message = "Calling mutate() on FieldGene with id " + std::to_string(parameters.id) +
				" and type " + std::to_string(static_cast<int>(parameters.type)) + 
				" is not possible, because the kernel is not a GaussKernel.";
			tools::logger::log(tools::logger::ERROR, message);
			return;
		}
		if (parameters.type != FieldGeneType::HIDDEN) 
		{
			const std::string message = "Calling mutate() on FieldGene with id " + std::to_string(parameters.id) +
				" and type " + std::to_string(static_cast<int>(parameters.type)) + 
				" is not possible, because the type is not a HIDDEN.";
			tools::logger::log(tools::logger::ERROR, message);
			return;
		}

		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<> mutationStep(-MutationConstants::mutationStep, 
												MutationConstants::mutationStep);
		std::uniform_int_distribution<> mutationSelection(0, 1);

		GaussKernelParameters gkp = std::dynamic_pointer_cast<GaussKernel>(kernel)->getParameters();

		if (mutationSelection(gen) == 0)
			gkp.sigma = std::clamp(gkp.sigma + mutationStep(gen), MutationConstants::minSigma,
															MutationConstants::maxSigma);
		else
			gkp.amplitude = std::clamp(gkp.amplitude + mutationStep(gen), MutationConstants::minAmplitude,
																	MutationConstants::maxAmplitude);

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

	bool FieldGene::operator==(const FieldGene& other) const
	{
		return parameters.type == other.parameters.type &&
			parameters.id == other.parameters.id;
	}
}