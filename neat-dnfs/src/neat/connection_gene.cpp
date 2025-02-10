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


		static constexpr double totalProbability = MutationConstants::mutateConnectionGeneKernelProbability +
			MutationConstants::mutateConnectionGeneConnectionSignalProbability +
			MutationConstants::mutateConnectionGeneKernelTypeProbability;

		constexpr double epsilon = 1e-6;
		if (std::abs(totalProbability - 1.0) > epsilon)
			throw std::runtime_error("Mutation probabilities in connection gene mutation must sum up to 1.");

		const double randomValue = tools::utils::generateRandomDouble(0.0, 1.0);
		if (randomValue < MutationConstants::mutateConnectionGeneKernelProbability)
		{
			mutateKernel();
		}
		else if (randomValue < MutationConstants::mutateConnectionGeneKernelProbability +
						MutationConstants::mutateConnectionGeneConnectionSignalProbability)
		{
			mutateConnectionSignal();
		}
		else
		{
			mutateKernelType();
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

		static constexpr double totalProbability = ConnectionGeneConstants::gaussKernelProbability +
			ConnectionGeneConstants::mexicanHatKernelProbability +
			ConnectionGeneConstants::oscillatoryKernelProbability;

		constexpr double epsilon = 1e-6;
		if (std::abs(totalProbability - 1.0) > epsilon)
			throw std::runtime_error("Kernel probabilities in connection gene initialization must sum up to 1.");

		const double randomValue = generateRandomDouble(0.0, 1.0);
		if (randomValue < ConnectionGeneConstants::gaussKernelProbability)
		{
			initializeGaussKernel(dimensions);
		}
		else if (randomValue < ConnectionGeneConstants::gaussKernelProbability + ConnectionGeneConstants::mexicanHatKernelProbability)
		{
			initializeMexicanHatKernel(dimensions);
		}
		else
		{
			initializeOscillatoryKernel(dimensions);
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

	void ConnectionGene::mutateKernel() const
	{
		switch (kernel->getLabel())
		{
		case dnf_composer::element::ElementLabel::GAUSS_KERNEL:
			mutateGaussKernel();
			break;
		case dnf_composer::element::ElementLabel::MEXICAN_HAT_KERNEL:
			mutateMexicanHatKernel();
			break;
		case dnf_composer::element::ElementLabel::OSCILLATORY_KERNEL:
			mutateOscillatoryKernel();
			break;
		default:
			tools::logger::log(tools::logger::FATAL, "ConnectionGene::mutate() - Kernel type not recognized.");
			throw std::runtime_error("ConnectionGene::mutate() - Kernel type not recognized.");
		}
	}

	void ConnectionGene::mutateKernelType()
	{
		using namespace dnf_composer::element;
		using namespace neat_dnfs::tools::utils;

		const auto dimensions = kernel->getElementCommonParameters().dimensionParameters;

		constexpr double totalProbability = FieldGeneConstants::gaussKernelProbability +
			FieldGeneConstants::mexicanHatKernelProbability +
			FieldGeneConstants::oscillatoryKernelProbability;

		constexpr double epsilon = 1e-6;
		if (std::abs(totalProbability - 1.0) > epsilon)
			throw std::runtime_error("Mutation probabilities in connection gene kernel type mutation must sum up to 1.");

		const double randomValue = generateRandomDouble(0.0, 1.0);

		if (randomValue < FieldGeneConstants::gaussKernelProbability)
		{
			initializeGaussKernel(dimensions);
		}
		else if (randomValue < FieldGeneConstants::gaussKernelProbability + FieldGeneConstants::mexicanHatKernelProbability)
		{
			initializeMexicanHatKernel(dimensions);
		}
		else
		{
			initializeOscillatoryKernel(dimensions);
		}
	}

	void ConnectionGene::mutateGaussKernel() const
	{
		using namespace dnf_composer::element;
		using namespace neat_dnfs::tools::utils;

		const int signal = generateRandomSignal(); // +/- to add or sum a step

		const auto gaussKernel = std::dynamic_pointer_cast<GaussKernel>(kernel);
		GaussKernelParameters gkp = std::dynamic_pointer_cast<GaussKernel>(kernel)->getParameters();
		const int amp_sign = gkp.amplitude < 0 ? -1 : 1;

		constexpr double totalProbability = MutationConstants::mutateGaussKernelWidthProbability +
			MutationConstants::mutateGaussKernelAmplitudeProbability +
			MutationConstants::mutateGaussKernelGlobalAmplitudeProbability;

		constexpr double epsilon = 1e-6;
		if (std::abs(totalProbability - 1.0) > epsilon)
			throw std::runtime_error("Mutation probabilities in field gene gauss kernel mutation must sum up to 1.");


		const double mutationSelection = generateRandomDouble(0.0, 1.0);

		if (mutationSelection < MutationConstants::mutateGaussKernelWidthProbability)
		{
			gkp.width = std::clamp(gkp.width + GaussKernelConstants::widthStep * signal,
				GaussKernelConstants::widthMinVal,
				GaussKernelConstants::widthMaxVal);
		}
		else if (mutationSelection < MutationConstants::mutateGaussKernelWidthProbability +
			MutationConstants::mutateGaussKernelAmplitudeProbability)
		{
			gkp.amplitude = amp_sign * std::clamp(gkp.amplitude + GaussKernelConstants::ampStep * signal,
				GaussKernelConstants::ampMinVal,
				GaussKernelConstants::ampMaxVal);
		}
		else
		{
			gkp.amplitudeGlobal = std::clamp(gkp.amplitudeGlobal + GaussKernelConstants::ampStep * signal,
				GaussKernelConstants::ampGlobalMinVal,
				GaussKernelConstants::ampGlobalMaxVal);
		}
		std::dynamic_pointer_cast<GaussKernel>(kernel)->setParameters(gkp);
	}

	void ConnectionGene::mutateMexicanHatKernel() const
	{
		using namespace dnf_composer::element;
		using namespace neat_dnfs::tools::utils;

		const int signal = generateRandomSignal();  // +/- to add or sum a step

		const auto mexicanHatKernel = std::dynamic_pointer_cast<MexicanHatKernel>(kernel);
		MexicanHatKernelParameters mhkp = std::dynamic_pointer_cast<MexicanHatKernel>(kernel)->getParameters();
		const int amp_sign = mhkp.amplitudeExc < 0 ? -1 : 1;

		constexpr double totalProbability = MutationConstants::mutateMexicanHatKernelAmplitudeExcProbability +
			MutationConstants::mutateMexicanHatKernelWidthExcProbability +
			MutationConstants::mutateMexicanHatKernelAmplitudeInhProbability +
			MutationConstants::mutateMexicanHatKernelWidthInhProbability +
			MutationConstants::mutateMexicanHatKernelGlobalAmplitudeProbability;

		constexpr double epsilon = 1e-6;
		if (std::abs(totalProbability - 1.0) > epsilon)
			throw std::runtime_error("Mutation probabilities in field gene mexican-hat kernel mutation must sum up to 1.");


		const double mutationSelection = generateRandomDouble(0.0, 1.0);
		if (mutationSelection < MutationConstants::mutateMexicanHatKernelAmplitudeExcProbability)
		{
			mhkp.amplitudeExc = amp_sign * std::clamp(mhkp.amplitudeExc + MexicanHatKernelConstants::ampExcStep * signal,
				MexicanHatKernelConstants::ampExcMinVal,
				MexicanHatKernelConstants::ampExcMaxVal);
		}
		else if (mutationSelection < MutationConstants::mutateMexicanHatKernelAmplitudeExcProbability +
			MutationConstants::mutateMexicanHatKernelWidthExcProbability)
		{
			mhkp.widthExc = std::clamp(mhkp.widthExc + MexicanHatKernelConstants::widthExcStep * signal,
				MexicanHatKernelConstants::widthExcMinVal,
				MexicanHatKernelConstants::widthExcMaxVal);
		}
		else if (mutationSelection < MutationConstants::mutateMexicanHatKernelAmplitudeExcProbability +
			MutationConstants::mutateMexicanHatKernelWidthExcProbability +
			MutationConstants::mutateMexicanHatKernelAmplitudeInhProbability)
		{
			mhkp.amplitudeInh = std::clamp(mhkp.amplitudeInh + MexicanHatKernelConstants::ampInhStep * signal,
				MexicanHatKernelConstants::ampInhMinVal,
				MexicanHatKernelConstants::ampInhMaxVal);
		}
		else if (mutationSelection < MutationConstants::mutateMexicanHatKernelAmplitudeExcProbability +
			MutationConstants::mutateMexicanHatKernelWidthExcProbability +
			MutationConstants::mutateMexicanHatKernelAmplitudeInhProbability +
			MutationConstants::mutateMexicanHatKernelWidthInhProbability)
		{
			mhkp.widthInh = std::clamp(mhkp.widthInh + MexicanHatKernelConstants::widthInhStep * signal,
				MexicanHatKernelConstants::widthInhMinVal,
				MexicanHatKernelConstants::widthInhMaxVal);
		}
		else
		{
			mhkp.amplitudeGlobal = std::clamp(mhkp.amplitudeGlobal + MexicanHatKernelConstants::ampGlobStep * signal,
				MexicanHatKernelConstants::ampGlobMin,
				MexicanHatKernelConstants::ampGlobMax);
		}
		std::dynamic_pointer_cast<MexicanHatKernel>(kernel)->setParameters(mhkp);
	}

	void ConnectionGene::mutateOscillatoryKernel() const
	{
		using namespace dnf_composer::element;
		using namespace neat_dnfs::tools::utils;

		const int signal = generateRandomSignal();  // +/- to add or sum a step

		const auto oscillatoryKernel = std::dynamic_pointer_cast<OscillatoryKernel>(kernel);
		OscillatoryKernelParameters okp = std::dynamic_pointer_cast<OscillatoryKernel>(kernel)->getParameters();
		const int amp_sign = okp.amplitude < 0 ? -1 : 1;

		constexpr double totalProbability = MutationConstants::mutateOscillatoryKernelAmplitudeProbability +
			MutationConstants::mutateOscillatoryKernelDecayProbability +
			MutationConstants::mutateOscillatoryKernelZeroCrossingsProbability +
			MutationConstants::mutateOscillatoryKernelGlobalAmplitudeProbability;

		constexpr double epsilon = 1e-6;
		if (std::abs(totalProbability - 1.0) > epsilon)
			throw std::runtime_error("Mutation probabilities in field gene oscillatory kernel mutation must sum up to 1.");


		const double mutationSelection = generateRandomDouble(0.0, 1.0);


		if (mutationSelection < MutationConstants::mutateOscillatoryKernelAmplitudeProbability)
		{
			okp.amplitude = amp_sign * std::clamp(okp.amplitude + OscillatoryKernelConstants::amplitudeStep * signal,
				OscillatoryKernelConstants::amplitudeMinVal,
				OscillatoryKernelConstants::amplitudeMaxVal);
		}
		else if (mutationSelection < MutationConstants::mutateOscillatoryKernelAmplitudeProbability +
			MutationConstants::mutateOscillatoryKernelDecayProbability)
		{
			okp.decay = std::clamp(okp.decay + OscillatoryKernelConstants::decayStep * signal,
				OscillatoryKernelConstants::decayMinVal,
				OscillatoryKernelConstants::decayMaxVal);
		}
		else if (mutationSelection < MutationConstants::mutateOscillatoryKernelAmplitudeProbability +
			MutationConstants::mutateOscillatoryKernelDecayProbability +
			MutationConstants::mutateOscillatoryKernelZeroCrossingsProbability)
		{
			okp.zeroCrossings = std::clamp(okp.zeroCrossings + OscillatoryKernelConstants::zeroCrossingsStep * signal,
				OscillatoryKernelConstants::zeroCrossingsMinVal,
				OscillatoryKernelConstants::zeroCrossingsMaxVal);
		}
		else
		{
			okp.amplitudeGlobal = std::clamp(okp.amplitudeGlobal + OscillatoryKernelConstants::ampGlobStep * signal,
				OscillatoryKernelConstants::ampGlobMin,
				OscillatoryKernelConstants::ampGlobMax);
		}
		std::dynamic_pointer_cast<OscillatoryKernel>(kernel)->setParameters(okp);
	}

	void ConnectionGene::mutateConnectionSignal() const
	{
		using namespace dnf_composer::element;

		switch (kernel->getLabel())
		{
		case GAUSS_KERNEL:
			{
				const auto gaussKernel = std::dynamic_pointer_cast<GaussKernel>(kernel);
				GaussKernelParameters gkp = std::dynamic_pointer_cast<GaussKernel>(kernel)->getParameters();
				gkp.amplitude = -gkp.amplitude;
				gaussKernel->setParameters(gkp);
			}
			break;
		case MEXICAN_HAT_KERNEL:
			{
				const auto mexicanHatKernel = std::dynamic_pointer_cast<MexicanHatKernel>(kernel);
				MexicanHatKernelParameters mhkp = std::dynamic_pointer_cast<MexicanHatKernel>(kernel)->getParameters();
				mhkp.amplitudeExc = -mhkp.amplitudeExc;
				mexicanHatKernel->setParameters(mhkp);
			}
			break;
		case OSCILLATORY_KERNEL:
			{
				const auto oscillatoryKernel = std::dynamic_pointer_cast<OscillatoryKernel>(kernel);
				OscillatoryKernelParameters okp = std::dynamic_pointer_cast<OscillatoryKernel>(kernel)->getParameters();
				okp.amplitude = -okp.amplitude;
				oscillatoryKernel->setParameters(okp);
			}
			break;
		default:
			tools::logger::log(tools::logger::FATAL, "ConnectionGene::mutate() - Kernel type not recognized.");
			throw std::runtime_error("ConnectionGene::mutate() - Kernel type not recognized.");
		}
	}
}