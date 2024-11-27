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

	FieldGene::FieldGene(FieldGeneParameters parameters, const NeuralFieldPtr& neuralField, const KernelPtr& kernel)
		: parameters(parameters), neuralField(neuralField), kernel(kernel)
	{}

	void FieldGene::setAsInput()
	{
		using namespace dnf_composer::element;

		parameters.type = FieldGeneType::INPUT;
		const NeuralFieldParameters nfp{ NeuralFieldConstants::tau,
											NeuralFieldConstants::restingLevel,
											NeuralFieldConstants::activationFunction };
		const ElementCommonParameters nfcp{ NeuralFieldConstants::namePrefix + std::to_string(parameters.id),
						{DimensionConstants::xSize,
											DimensionConstants::dx}
		};
		const GaussKernelParameters gkp{ GaussKernelConstants::width,
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
		const MexicanHatKernelParameters mhkp{ MexicanHatKernelConstants::widthExc,
												MexicanHatKernelConstants::amplitudeExc,
												MexicanHatKernelConstants::widthInh,
												MexicanHatKernelConstants::amplitudeInh,
												MexicanHatKernelConstants::amplitudeGlobal,
												KernelConstants::circularity,
												KernelConstants::normalization };


		//const LateralInteractionsParameters lip{ LateralInteractionsConstants::widthExc,
		//											LateralInteractionsConstants::amplitudeExc,
		//											LateralInteractionsConstants::widthInh,
		//									LateralInteractionsConstants::amplitudeExc,
		//											LateralInteractionsConstants::amplitudeGlobal,
		//									KernelConstants::circularity,
		//									KernelConstants::normalization };
		const ElementCommonParameters mhcp{ MexicanHatKernelConstants::namePrefix + std::to_string(parameters.id),
											{	DimensionConstants::xSize,
												DimensionConstants::dx}
		};
		neuralField = std::make_shared<NeuralField>(nfcp, nfp);
		kernel = std::make_shared<MexicanHatKernel>(mhcp, mhkp);
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
		const GaussKernelParameters gkp{ GaussKernelConstants::width,
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
				" is not possible, because the type is not HIDDEN.";
			tools::logger::log(tools::logger::ERROR, message);
			return;
		}

		using namespace neat_dnfs::tools::utils;
		const double mutationStep =
			generateRandomDouble(-MutationConstants::mutationStep, MutationConstants::mutationStep);
		const int mutationSelection = generateRandomInt(0, 1);

		GaussKernelParameters gkp = std::dynamic_pointer_cast<GaussKernel>(kernel)->getParameters();

		if (mutationSelection == 0)
			gkp.width = std::clamp(gkp.width + mutationStep, MutationConstants::minWidth,
				MutationConstants::maxWidth);
		else
			gkp.amplitude = std::clamp(gkp.amplitude + mutationStep, MutationConstants::minAmplitude,
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
		return parameters == other.parameters;
	}

	bool FieldGene::isCloneOf(const FieldGene& other) const
	{
		using namespace dnf_composer::element;
		const auto k_other = other.getKernel();
		const auto nf_other = other.getNeuralField();
		/*bool isKernelEqual = false;
		switch (parameters.type)
		{
		case FieldGeneType::INPUT:
		{
			const auto gk = std::dynamic_pointer_cast<GaussKernel>(kernel);
			const auto gk_other = std::dynamic_pointer_cast<GaussKernel>(k_other);
			isKernelEqual = gk->getParameters() == gk_other->getParameters();
		}
		break;
		case FieldGeneType::OUTPUT:
		{
			const auto li = std::dynamic_pointer_cast<LateralInteractions>(kernel);
			const auto li_other = std::dynamic_pointer_cast<LateralInteractions>(k_other);
			isKernelEqual = li->getParameters() == li_other->getParameters();
		}
		break;
		case FieldGeneType::HIDDEN:
		{
			const auto gk = std::dynamic_pointer_cast<GaussKernel>(kernel);
			const auto gk_other = std::dynamic_pointer_cast<GaussKernel>(k_other);
			isKernelEqual = gk->getParameters() == gk_other->getParameters();
		}
		break;
		}*/

		/*const NeuralFieldParameters nfp = neuralField->getParameters();
		const NeuralFieldParameters nfp_other = nf_other->getParameters();*/
		//const bool isNeuralFieldEqual = nfp == nfp_other;

		return parameters == other.parameters && neuralField == nf_other && kernel == k_other;
	}

	std::string FieldGene::toString() const
	{
		std::string result = "FieldGene: ";
		result += parameters.toString();

		std::stringstream addr_nf;
		addr_nf << neuralField.get();
		result += "NeuralField: " + addr_nf.str() + '\n';
		result += neuralField->toString();

		std::stringstream addr_k;
		addr_k << kernel.get();
		result += "Kernel: " + addr_k.str() + '\n';
		result += kernel->toString();

		return result;
	}

	void FieldGene::print() const
	{
		tools::logger::log(tools::logger::INFO, toString());
	}

	FieldGene FieldGene::clone() const
	{
		const auto nf = neuralField->clone();
		const auto k = kernel->clone();
		
		const auto nf_ = std::dynamic_pointer_cast<dnf_composer::element::NeuralField>(nf);
		const auto k_ = std::dynamic_pointer_cast<dnf_composer::element::Kernel>(k);

		FieldGene clone{ parameters, nf_, k_ };
		return clone;
	}

}