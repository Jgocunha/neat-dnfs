#include "neat/connection_gene.h"


namespace neat_dnfs
{
	ConnectionGene::ConnectionGene(ConnectionTuple connectionTuple)
		: parameters(connectionTuple)
	{
		initializeKernel({ DimensionConstants::xSize, DimensionConstants::dx });
	}

	ConnectionGene::ConnectionGene(ConnectionTuple connectionTuple,
		const dnf_composer::element::GaussKernelParameters& gkp)
		: parameters(connectionTuple)
	{
		using namespace dnf_composer::element;

		const std::string elementName = GaussKernelConstants::namePrefixConnectionGene +
			std::to_string(connectionTuple.inFieldGeneId) +
			" - " + std::to_string(connectionTuple.outFieldGeneId) + " " +
			std::to_string(parameters.innovationNumber);
		const ElementCommonParameters gkcp{ elementName,
			{DimensionConstants::xSize, DimensionConstants::dx} };
		kernel = std::make_unique<GaussKernel>(gkcp, gkp);
	}

	ConnectionGene::ConnectionGene(ConnectionTuple connectionTuple,
		const dnf_composer::element::MexicanHatKernelParameters& mhkp)
		: parameters(connectionTuple)
	{
		using namespace dnf_composer::element;

		const std::string elementName = MexicanHatKernelConstants::namePrefixConnectionGene +
			std::to_string(connectionTuple.inFieldGeneId) +
			" - " + std::to_string(connectionTuple.outFieldGeneId) + " " +
			std::to_string(parameters.innovationNumber);
		const ElementCommonParameters mhkcp{ elementName,
			{DimensionConstants::xSize, DimensionConstants::dx} };
		kernel = std::make_unique<MexicanHatKernel>(mhkcp, mhkp);
	}

	ConnectionGene::ConnectionGene(ConnectionTuple connectionTuple,
		const dnf_composer::element::OscillatoryKernelParameters& osckp)
		: parameters(connectionTuple)
	{
		using namespace dnf_composer::element;

		const std::string elementName = OscillatoryKernelConstants::namePrefixConnectionGene +
			std::to_string(parameters.connectionTuple.inFieldGeneId) +
			" - " + std::to_string(parameters.connectionTuple.outFieldGeneId) + " " +
			std::to_string(parameters.innovationNumber);
		const ElementCommonParameters osckcp{ elementName,
			{DimensionConstants::xSize, DimensionConstants::dx} };
		kernel = std::make_unique<OscillatoryKernel>(osckcp, osckp);
	}

	ConnectionGene::ConnectionGene(const ConnectionGeneParameters& parameters,
		const dnf_composer::element::GaussKernelParameters& gkp)
		: parameters(parameters)
	{
		using namespace dnf_composer::element;

		const std::string elementName = GaussKernelConstants::namePrefixConnectionGene +
			std::to_string(parameters.connectionTuple.inFieldGeneId) +
			" - " + std::to_string(parameters.connectionTuple.outFieldGeneId) + " " +
			std::to_string(parameters.innovationNumber);
		const ElementCommonParameters gkcp{ elementName,
			{DimensionConstants::xSize, DimensionConstants::dx} };
		kernel = std::make_unique<GaussKernel>(gkcp, gkp);
	}

	ConnectionGene::ConnectionGene(const ConnectionGeneParameters& parameters,
				const dnf_composer::element::MexicanHatKernelParameters& mhkp)
		: parameters(parameters)
	{
				using namespace dnf_composer::element;

		const std::string elementName = MexicanHatKernelConstants::namePrefixConnectionGene +
			std::to_string(parameters.connectionTuple.inFieldGeneId) +
			" - " + std::to_string(parameters.connectionTuple.outFieldGeneId) + " " +
			std::to_string(parameters.innovationNumber);
		const ElementCommonParameters mhkcp{ elementName,
					{DimensionConstants::xSize, DimensionConstants::dx} };
		kernel = std::make_unique<MexicanHatKernel>(mhkcp, mhkp);
	}

	ConnectionGene::ConnectionGene(const ConnectionGeneParameters& parameters,
						const dnf_composer::element::OscillatoryKernelParameters& osckp)
		: parameters(parameters)
	{
						using namespace dnf_composer::element;

		const std::string elementName = OscillatoryKernelConstants::namePrefixConnectionGene +
			std::to_string(parameters.connectionTuple.inFieldGeneId) +
			" - " + std::to_string(parameters.connectionTuple.outFieldGeneId) + " " +
			std::to_string(parameters.innovationNumber);
		const ElementCommonParameters osckcp{ elementName,
							{DimensionConstants::xSize, DimensionConstants::dx} };
		kernel = std::make_unique<OscillatoryKernel>(osckcp, osckp);
	}

	ConnectionGene::ConnectionGene(ConnectionTuple connectionTuple, const KernelPtr& kernel)
		: parameters(connectionTuple), kernel(kernel)
	{}

	void ConnectionGene::mutate()
	{
		using namespace dnf_composer::element;

		if (!kernel)
		{
			const std::string message = "Calling mutate() on ConnectionGene with ConnectionTuple: " +
				std::to_string(parameters.connectionTuple.inFieldGeneId) + " - " +
				std::to_string(parameters.connectionTuple.outFieldGeneId) + " but kernel is nullptr.";
			tools::logger::log(tools::logger::FATAL, message);
			throw std::runtime_error(message);
		}

		using namespace neat_dnfs::tools::utils;

		// discover kernel type
		const ElementLabel label = kernel->getLabel();
		const int signal = generateRandomSignal();
		switch (label)
		{
		case ElementLabel::GAUSS_KERNEL:
		{
			const auto gaussKernel = std::dynamic_pointer_cast<GaussKernel>(kernel);
			GaussKernelParameters gkp = std::dynamic_pointer_cast<GaussKernel>(kernel)->getParameters();
			ElementDimensions gkd = kernel->getElementCommonParameters().dimensionParameters;
			const int amp_sign = gkp.amplitude < 0 ? -1 : 1;

			const int mutationSelection = generateRandomInt(0, 2 
				//+ ConnectionGeneConstants::allowAllKernelTypes 
				+ ConnectionGeneConstants::allowInhibitoryConnections
			); // number of mutable parameters + change type + inhibitory connections
			switch (mutationSelection)
			{
			case 0: // width
				gkp.width = std::clamp(gkp.width + GaussKernelConstants::widthStep * signal,
					GaussKernelConstants::widthMinVal,
					GaussKernelConstants::widthMaxVal);
				break;
			case 1: // amplitude
				gkp.amplitude = amp_sign * std::clamp(gkp.amplitude + GaussKernelConstants::ampStep * signal,
					GaussKernelConstants::ampMinVal,
					GaussKernelConstants::ampMaxVal);
				break;
			case 2: // amplitude global
				gkp.amplitudeGlobal = std::clamp(gkp.amplitudeGlobal + GaussKernelConstants::ampStep * signal,
					GaussKernelConstants::ampGlobalMinVal,
					GaussKernelConstants::ampGlobalMaxVal);
				break;
			//case 3: // change type
			//{
			//	const int typeSelection = generateRandomInt(0, 1); // number of (other) kernel types
			//	switch (typeSelection)
			//	{
			//	case 0:
			//		initializeMexicanHatKernel(gkd);
			//		return;
			//	case 1:
			//		initializeOscillatoryKernel(gkd);
			//		return;
			//	default:
			//		break;
			//	}
			//}
			//return; // this is a return because the kernel type has changed
			case 3: // toggle signal
				gkp.amplitude = -gkp.amplitude;
				break;
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
			const int amp_sign = mhkp.amplitudeExc < 0 ? -1 : 1;

			const int mutationSelection = generateRandomInt(0, 4
				+ ConnectionGeneConstants::allowAllKernelTypes
				+ ConnectionGeneConstants::allowInhibitoryConnections
			); // number of mutable parameters + change type + inhibitory connections
			switch (mutationSelection)
			{
			case 0: // width exc
				mhkp.widthExc = std::clamp(mhkp.widthExc + MexicanHatKernelConstants::widthExcStep * signal,
					MexicanHatKernelConstants::widthExcMinVal,
					MexicanHatKernelConstants::widthExcMaxVal);
				break;
			case 1: // amplitude exc
				mhkp.amplitudeExc = amp_sign * std::clamp(mhkp.amplitudeExc + MexicanHatKernelConstants::ampExcStep * signal,
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
			case 6: // toggle signal
				mhkp.amplitudeExc = -mhkp.amplitudeExc;
				break;
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
			const int amp_sign = okp.amplitude < 0 ? -1 : 1;

			const int mutationSelection = generateRandomInt(0, 3
				+ ConnectionGeneConstants::allowAllKernelTypes
				+ ConnectionGeneConstants::allowInhibitoryConnections
			); // number of mutable parameters + change type + inhibitory connections
			switch (mutationSelection)
			{
			case 0: // amplitude
				okp.amplitude = amp_sign * std::clamp(okp.amplitude + OscillatoryKernelConstants::amplitudeStep * signal,
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
			case 5: // toggle signal
				okp.amplitude = -okp.amplitude;
				break;
			default:
				break;
			}
			std::dynamic_pointer_cast<OscillatoryKernel>(kernel)->setParameters(okp);
		}
		break;
		default:
			tools::logger::log(tools::logger::FATAL, "ConnectionGene::mutate() - Kernel type not recognized.");
			throw std::runtime_error("ConnectionGene::mutate() - Kernel type not recognized.");
			break;
		}
	}

	void ConnectionGene::disable()
	{
		parameters.enabled = false;
	}

	void ConnectionGene::toggle()
	{
		parameters.enabled = !parameters.enabled;
	}

	bool ConnectionGene::isEnabled() const
	{
		return parameters.enabled;
	}

	void ConnectionGene::setInnovationNumber(uint16_t innovationNumber)
	{
		parameters.innovationNumber = innovationNumber;
	}

	ConnectionGeneParameters ConnectionGene::getParameters() const
	{
		return parameters;
	}

	KernelPtr ConnectionGene::getKernel() const
	{
		return kernel;
	}

	uint16_t ConnectionGene::getInnovationNumber() const
	{
		return parameters.innovationNumber;
	}

	uint16_t ConnectionGene::getInFieldGeneId() const
	{
		return parameters.connectionTuple.inFieldGeneId;
	}

	uint16_t ConnectionGene::getOutFieldGeneId() const
	{
		return parameters.connectionTuple.outFieldGeneId;
	}

	double ConnectionGene::getKernelAmplitude() const
	{
		using namespace dnf_composer::element;
		switch (kernel->getLabel())
		{
			case GAUSS_KERNEL:
				return std::dynamic_pointer_cast<GaussKernel>(kernel)->getParameters().amplitude;
			case MEXICAN_HAT_KERNEL:
				return std::dynamic_pointer_cast<MexicanHatKernel>(kernel)->getParameters().amplitudeExc;
			case OSCILLATORY_KERNEL:
				return std::dynamic_pointer_cast<OscillatoryKernel>(kernel)->getParameters().amplitude;
			default:
				break;
		}
		throw std::runtime_error("ConnectionGene::getKernelAmplitude() - Kernel type not recognized.");
	}

	double ConnectionGene::getKernelWidth() const
	{
		using namespace dnf_composer::element;
		switch (kernel->getLabel())
		{
			case GAUSS_KERNEL:
				return std::dynamic_pointer_cast<GaussKernel>(kernel)->getParameters().width;
			case MEXICAN_HAT_KERNEL:
				return std::dynamic_pointer_cast<MexicanHatKernel>(kernel)->getParameters().widthExc;
			case OSCILLATORY_KERNEL:
				return std::dynamic_pointer_cast<OscillatoryKernel>(kernel)->getParameters().decay;
			default:
				break;
		}
		throw std::runtime_error("ConnectionGene::getKernelWidth() - Kernel type not recognized.");
	}


	bool ConnectionGene::operator==(const ConnectionGene& other) const
	{
		return parameters.innovationNumber == other.parameters.innovationNumber;
	}

	bool ConnectionGene::isCloneOf(const ConnectionGene& other) const
	{
		using namespace dnf_composer::element;

		const auto k_other = other.getKernel();
		return parameters == other.parameters && kernel == k_other;
	}

	std::string ConnectionGene::toString() const
	{
		std::string result = "ConnectionGene: ";
		result += parameters.toString();

		std::stringstream address;
		address << kernel.get();
		result += "Kernel address: " + address.str() + '\n';
		result += kernel->toString();
		return result;
	}

	void ConnectionGene::print() const
	{
		tools::logger::log(tools::logger::INFO, toString());
	}

	ConnectionGene ConnectionGene::clone() const
	{
		using namespace dnf_composer::element;

		switch (kernel->getLabel())
		{
			case GAUSS_KERNEL:
				return ConnectionGene{ parameters, std::dynamic_pointer_cast<GaussKernel>(kernel)->getParameters() };
			case MEXICAN_HAT_KERNEL:
				return ConnectionGene{ parameters, std::dynamic_pointer_cast<MexicanHatKernel>(kernel)->getParameters() };
			case OSCILLATORY_KERNEL:
				return ConnectionGene{ parameters, std::dynamic_pointer_cast<OscillatoryKernel>(kernel)->getParameters() };
			default:
				break;
		}

		throw std::runtime_error("ConnectionGene::clone() - Kernel type not recognized.");
	}

	void ConnectionGene::initializeKernel(const dnf_composer::element::ElementDimensions& dimensions)
	{
		using namespace neat_dnfs::tools::utils;

		if (ConnectionGeneConstants::allowAllKernelTypes)
		{
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
		else
		{
			initializeGaussKernel(dimensions);
		}
	}

	void ConnectionGene::initializeGaussKernel(const dnf_composer::element::ElementDimensions& dimensions)
	{
		using namespace dnf_composer::element;
		using namespace neat_dnfs::tools::utils;

		const int amplitude_sign = generateRandomSignal();
		const double width = generateRandomDouble(GaussKernelConstants::widthMinVal, GaussKernelConstants::widthMaxVal);
		const double amplitude = amplitude_sign * generateRandomDouble(GaussKernelConstants::ampMinVal, GaussKernelConstants::ampMaxVal);
		const double amplitudeGlobal = generateRandomDouble(GaussKernelConstants::ampGlobalMinVal, GaussKernelConstants::ampGlobalMaxVal);
		const GaussKernelParameters gkp{ width,
										amplitude,
											amplitudeGlobal,
									KernelConstants::circularity,
									KernelConstants::normalization
		};
		const std::string elementName = GaussKernelConstants::namePrefixConnectionGene + std::to_string(parameters.connectionTuple.inFieldGeneId) +
			" - " + std::to_string(parameters.connectionTuple.outFieldGeneId) + " " +
			std::to_string(parameters.innovationNumber);
		const ElementCommonParameters gkcp{ elementName, dimensions };
		kernel = std::make_shared<GaussKernel>(gkcp, gkp);
	}

	void ConnectionGene::initializeMexicanHatKernel(const dnf_composer::element::ElementDimensions& dimensions)
	{
		using namespace dnf_composer::element;
		using namespace neat_dnfs::tools::utils;

		const int amplitude_sign = generateRandomSignal();
		const double widthExc = generateRandomDouble(MexicanHatKernelConstants::widthExcMinVal, MexicanHatKernelConstants::widthExcMaxVal);
		const double amplitudeExc = amplitude_sign * generateRandomDouble(MexicanHatKernelConstants::ampExcMinVal, MexicanHatKernelConstants::ampExcMaxVal);
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
		const std::string elementName = MexicanHatKernelConstants::namePrefixConnectionGene + std::to_string(parameters.connectionTuple.inFieldGeneId) +
			" - " + std::to_string(parameters.connectionTuple.outFieldGeneId) + " " +
			std::to_string(parameters.innovationNumber);
		const ElementCommonParameters mhcp{ elementName, dimensions};
		kernel = std::make_shared<MexicanHatKernel>(mhcp, mhkp);
	}

	void ConnectionGene::initializeOscillatoryKernel(const dnf_composer::element::ElementDimensions& dimensions)
	{
		using namespace dnf_composer::element;
		using namespace neat_dnfs::tools::utils;

		const int amplitude_sign = generateRandomSignal();
		const double amplitude = amplitude_sign * generateRandomDouble(OscillatoryKernelConstants::amplitudeMinVal, OscillatoryKernelConstants::amplitudeMaxVal);
		const double decay = generateRandomDouble(OscillatoryKernelConstants::decayMinVal, OscillatoryKernelConstants::decayMaxVal);
		const double zeroCrossings = generateRandomDouble(OscillatoryKernelConstants::zeroCrossingsMinVal, OscillatoryKernelConstants::zeroCrossingsMaxVal);
		const double amplitudeGlobal = generateRandomDouble(OscillatoryKernelConstants::ampGlobMin, OscillatoryKernelConstants::ampGlobMax);
		const OscillatoryKernelParameters okp{ amplitude, decay, zeroCrossings, amplitudeGlobal,
											KernelConstants::circularity,
											KernelConstants::normalization
		};
		const std::string elementName = OscillatoryKernelConstants::namePrefixConnectionGene + std::to_string(parameters.connectionTuple.inFieldGeneId) +
			" - " + std::to_string(parameters.connectionTuple.outFieldGeneId) + " " +
			std::to_string(parameters.innovationNumber);
		const ElementCommonParameters okcp{ elementName, dimensions};
		kernel = std::make_shared<OscillatoryKernel>(okcp, okp);
	}
}