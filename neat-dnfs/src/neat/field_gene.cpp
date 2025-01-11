#include "neat/field_gene.h"

namespace neat_dnfs
{
	FieldGene::FieldGene(const FieldGeneParameters& parameters, const dnf_composer::element::ElementDimensions& dimensions)
		: parameters(parameters)
	{
		switch (parameters.type)
		{
		case FieldGeneType::INPUT:
			setAsInput(dimensions);
			break;
		case FieldGeneType::OUTPUT:
			setAsOutput(dimensions);
			break;
		case FieldGeneType::HIDDEN:
			setAsHidden(dimensions);
			break;
		}
	}

	FieldGene::FieldGene(const FieldGeneParameters& parameters, const NeuralFieldPtr& neuralField, 
		const KernelPtr& kernel)
		: parameters(parameters), neuralField(neuralField), kernel(kernel)
	{}

	void FieldGene::setAsInput(const dnf_composer::element::ElementDimensions& dimensions)
	{
		parameters.type = FieldGeneType::INPUT;
		initializeNeuralField(dimensions);
		initializeKernel(dimensions);
	}

	void FieldGene::setAsOutput(const dnf_composer::element::ElementDimensions& dimensions)
	{
		parameters.type = FieldGeneType::OUTPUT;
		initializeNeuralField(dimensions);
		initializeKernel(dimensions);
	}

	void FieldGene::setAsHidden(const dnf_composer::element::ElementDimensions& dimensions)
	{
		parameters.type = FieldGeneType::HIDDEN;
		initializeNeuralField(dimensions);
		initializeKernel(dimensions);
	}

	void FieldGene::mutate()
	{
		using namespace dnf_composer::element;
		using namespace neat_dnfs::tools::utils;

		// discover kernel type
		const ElementLabel label = kernel->getLabel();
		const int signal = generateRandomSignal();
		switch(label)
		{
		case ElementLabel::GAUSS_KERNEL:
			{
				const auto gaussKernel = std::dynamic_pointer_cast<GaussKernel>(kernel);
				GaussKernelParameters gkp = std::dynamic_pointer_cast<GaussKernel>(kernel)->getParameters();
				ElementDimensions gkd = kernel->getElementCommonParameters().dimensionParameters;

				const int mutationSelection = generateRandomInt(0, 3); // number of mutable parameters + change type
				switch(mutationSelection)
				{
				case 0: // width
					gkp.width = std::clamp(gkp.width + GaussKernelConstants::widthStep * signal,
						GaussKernelConstants::widthMinVal, 
						GaussKernelConstants::widthMaxVal);
					break;
				case 1: // amplitude
					gkp.amplitude = std::clamp(gkp.amplitude + GaussKernelConstants::ampStep * signal,
						GaussKernelConstants::ampMinVal,
						GaussKernelConstants::ampMaxVal);
					break;
				case 2: // amplitude global
					gkp.amplitudeGlobal = std::clamp(gkp.amplitudeGlobal + GaussKernelConstants::ampStep * signal,
						GaussKernelConstants::ampGlobalMinVal,
						GaussKernelConstants::ampGlobalMaxVal);
					break;
				case 3: // change type
					{
						const int typeSelection = generateRandomInt(0, 1); // number of (other) kernel types
						switch(typeSelection)
						{
							case 0:
								initializeMexicanHatKernel(gkd);
							return;
							case 1:
								initializeOscillatoryKernel(gkd);
							return;
							default:
								break;
						}
					}
					return; // this is a return because the kernel type has changed
				default:
					break;
				}
				std::dynamic_pointer_cast<GaussKernel>(kernel)->setParameters(gkp);
			}

			break;
		case ElementLabel::MEXICAN_HAT_KERNEL:
			{
				const auto mexicanHatKernel = std::dynamic_pointer_cast<MexicanHatKernel>(kernel);
				MexicanHatKernelParameters mhkp = std::dynamic_pointer_cast<MexicanHatKernel>(kernel)->getParameters();
				ElementDimensions mhkd = kernel->getElementCommonParameters().dimensionParameters;

				const int mutationSelection = generateRandomInt(0, 5); // number of mutable parameters + change type
				switch (mutationSelection)
				{
				case 0: // width exc
					mhkp.widthExc = std::clamp(mhkp.widthExc + MexicanHatKernelConstants::widthExcStep * signal,
						MexicanHatKernelConstants::widthExcMinVal,
						MexicanHatKernelConstants::widthExcMaxVal);
					break;
				case 1: // amplitude exc
					mhkp.amplitudeExc = std::clamp(mhkp.amplitudeExc + MexicanHatKernelConstants::ampExcStep * signal,
						MexicanHatKernelConstants::ampExcMinVal,
						MexicanHatKernelConstants::ampExcMaxVal);
					break;
				case 2: // width inh
					mhkp.widthInh = std::clamp(mhkp.widthInh + MexicanHatKernelConstants::widthInhStep * signal,
						MexicanHatKernelConstants::widthInhMinVal,
						MexicanHatKernelConstants::widthInhMaxVal);
					break;
				case 3: // amplitude inh
					mhkp.amplitudeInh = std::clamp(mhkp.amplitudeInh + MexicanHatKernelConstants::ampInhStep * signal,
						MexicanHatKernelConstants::ampInhMinVal,
						MexicanHatKernelConstants::ampInhMaxVal);
					break;
				case 4: // amplitude global
					mhkp.amplitudeGlobal = std::clamp(mhkp.amplitudeGlobal + MexicanHatKernelConstants::ampGlobStep * signal,
						MexicanHatKernelConstants::ampGlobMin,
						MexicanHatKernelConstants::ampGlobMax);
					break;
				case 5: // change type
					{
						const int typeSelection = generateRandomInt(0, 1); // number of (other) kernel types
						switch (typeSelection)
						{
						case 0:
							initializeGaussKernel(mhkd);
							break;
						case 1:
							initializeOscillatoryKernel(mhkd);
							break;
						default:
							break;
						}
					}
					return; // this is a return because the kernel type has changed
				default:
					break;
				}
				std::dynamic_pointer_cast<MexicanHatKernel>(kernel)->setParameters(mhkp);
			}
			break;
		case ElementLabel::OSCILLATORY_KERNEL:
		{
			const auto oscillatoryKernel = std::dynamic_pointer_cast<OscillatoryKernel>(kernel);
			OscillatoryKernelParameters okp = std::dynamic_pointer_cast<OscillatoryKernel>(kernel)->getParameters();
			ElementDimensions okd = kernel->getElementCommonParameters().dimensionParameters;

			const int mutationSelection = generateRandomInt(0, 4); // number of mutable parameters + change type
			switch (mutationSelection)
			{
				case 0: // amplitude
					okp.amplitude = std::clamp(okp.amplitude + OscillatoryKernelConstants::amplitudeStep * signal,
													OscillatoryKernelConstants::amplitudeMinVal,
													OscillatoryKernelConstants::amplitudeMaxVal);
					break;
				case 1: // decay
					okp.decay = std::clamp(okp.decay + OscillatoryKernelConstants::decayStep * signal,
																		OscillatoryKernelConstants::decayMinVal,
																		OscillatoryKernelConstants::decayMaxVal);
					break;
				case 2: // zero crossings
					okp.zeroCrossings = std::clamp(okp.zeroCrossings + OscillatoryKernelConstants::zeroCrossingsStep * signal,
																				OscillatoryKernelConstants::zeroCrossingsMinVal,
																				OscillatoryKernelConstants::zeroCrossingsMaxVal);
					break;
				case 3: // amplitude global
					okp.amplitudeGlobal = std::clamp(okp.amplitudeGlobal + OscillatoryKernelConstants::ampGlobStep * signal,
																				OscillatoryKernelConstants::ampGlobMin,
																				OscillatoryKernelConstants::ampGlobMax);
					break;
				case 4: // change type
					{
						const int typeSelection = generateRandomInt(0, 1); // number of (other) kernel types
						switch (typeSelection)
						{
							case 0:
								initializeGaussKernel(okd);
								break;
							case 1:
								initializeMexicanHatKernel(okd);
								break;
							default:
								break;
						}
					}
					return; // this is a return because the kernel type has changed
				default:
					break;
			}
			std::dynamic_pointer_cast<OscillatoryKernel>(kernel)->setParameters(okp);
		}
		break;
		default:
			tools::logger::log(tools::logger::FATAL, "FieldGene::mutate() - Kernel type not recognized.");
			throw std::runtime_error("FieldGene::mutate() - Kernel type not recognized.");
			break;
		}
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

	void FieldGene::initializeNeuralField(const dnf_composer::element::ElementDimensions& dimensions)
	{
		using namespace dnf_composer::element;
		using namespace neat_dnfs::tools::utils;

		const double tau = generateRandomDouble(NeuralFieldConstants::tauMinVal, NeuralFieldConstants::tauMaxVal);
		const double restingLevel = generateRandomDouble(NeuralFieldConstants::restingLevelMinVal, NeuralFieldConstants::restingLevelMaxVal);

		const NeuralFieldParameters nfp{ tau, restingLevel,
											NeuralFieldConstants::activationFunction};
		const ElementCommonParameters nfcp{ NeuralFieldConstants::namePrefix + std::to_string(parameters.id),
						dimensions};
		neuralField = std::make_shared<NeuralField>(nfcp, nfp);
	}

	void FieldGene::initializeKernel(const dnf_composer::element::ElementDimensions& dimensions)
	{
		using namespace neat_dnfs::tools::utils;

		// randomly select kernel type
		switch (generateRandomInt(0, 2))
		{
		case 0:
			initializeGaussKernel(dimensions);
		break;
		case 1:
			initializeMexicanHatKernel(dimensions);
		break;
		case 2:
			initializeOscillatoryKernel(dimensions);
		default:
			break;
		}
	}

	void FieldGene::initializeGaussKernel(const dnf_composer::element::ElementDimensions& dimensions)
	{
		using namespace dnf_composer::element;
		using namespace neat_dnfs::tools::utils;

		const double width = generateRandomDouble(GaussKernelConstants::widthMinVal, GaussKernelConstants::widthMaxVal);
		const double amplitude = generateRandomDouble(GaussKernelConstants::ampMinVal, GaussKernelConstants::ampMaxVal);
		const double amplitudeGlobal = generateRandomDouble(GaussKernelConstants::ampGlobalMinVal, GaussKernelConstants::ampGlobalMaxVal);
		const GaussKernelParameters gkp{ width,
										amplitude,
											amplitudeGlobal,
									KernelConstants::circularity,
									KernelConstants::normalization
		};
		const ElementCommonParameters gkcp{ GaussKernelConstants::namePrefix + std::to_string(parameters.id),
						dimensions};
		kernel = std::make_shared<GaussKernel>(gkcp, gkp);
	}

	void FieldGene::initializeMexicanHatKernel(const dnf_composer::element::ElementDimensions& dimensions)
	{
		using namespace dnf_composer::element;
		using namespace neat_dnfs::tools::utils;

		const double widthExc = generateRandomDouble(MexicanHatKernelConstants::widthExcMinVal, MexicanHatKernelConstants::widthExcMaxVal);
		const double amplitudeExc = generateRandomDouble(MexicanHatKernelConstants::ampExcMinVal, MexicanHatKernelConstants::ampExcMaxVal);
		const double widthInh = generateRandomDouble(MexicanHatKernelConstants::widthInhMinVal, MexicanHatKernelConstants::widthInhMaxVal);
		const double amplitudeInh = generateRandomDouble(MexicanHatKernelConstants::ampInhMinVal, MexicanHatKernelConstants::ampInhMaxVal);
		const double amplitudeGlobal = generateRandomDouble(MexicanHatKernelConstants::ampGlobMin, MexicanHatKernelConstants::ampGlobMax);
		const MexicanHatKernelParameters mhkp{ widthExc,
								amplitudeExc,
								widthInh,
								amplitudeInh,
								amplitudeGlobal,
								KernelConstants::circularity,
								KernelConstants::normalization
		};
		const ElementCommonParameters mhcp{ MexicanHatKernelConstants::namePrefix + std::to_string(parameters.id), dimensions
		};
		kernel = std::make_shared<MexicanHatKernel>(mhcp, mhkp);
	}

	void FieldGene::initializeOscillatoryKernel(const dnf_composer::element::ElementDimensions& dimensions)
	{
		using namespace dnf_composer::element;
		using namespace neat_dnfs::tools::utils;

		const double amplitude = generateRandomDouble(OscillatoryKernelConstants::amplitudeMinVal, OscillatoryKernelConstants::amplitudeMaxVal);
		const double decay = generateRandomDouble(OscillatoryKernelConstants::decayMinVal, OscillatoryKernelConstants::decayMaxVal);
		const double zeroCrossings = generateRandomDouble(OscillatoryKernelConstants::zeroCrossingsMinVal, OscillatoryKernelConstants::zeroCrossingsMaxVal);
		const double amplitudeGlobal = generateRandomDouble(OscillatoryKernelConstants::ampGlobMin, OscillatoryKernelConstants::ampGlobMax);
		const OscillatoryKernelParameters okp{ amplitude, decay, zeroCrossings, amplitudeGlobal,
											KernelConstants::circularity,
											false
				};
		const ElementCommonParameters okcp{ OscillatoryKernelConstants::namePrefix + std::to_string(parameters.id), dimensions
				};
		kernel = std::make_shared<OscillatoryKernel>(okcp, okp);
	}


}