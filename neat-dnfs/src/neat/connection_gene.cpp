#include "neat/connection_gene.h"
#include <format>

namespace neat_dnfs
{
	ConnectionTuple::ConnectionTuple(const int inFieldGeneId, const int outFieldGeneId)
		: inFieldGeneId(inFieldGeneId), outFieldGeneId(outFieldGeneId)
	{}

	bool ConnectionTuple::operator==(const ConnectionTuple& other) const
	{
		return inFieldGeneId == other.inFieldGeneId && outFieldGeneId == other.outFieldGeneId;
	}

	bool ConnectionTuple::operator<(const ConnectionTuple& other) const {
		if (inFieldGeneId == other.inFieldGeneId)
		{
			return outFieldGeneId < other.outFieldGeneId;
		}
		return inFieldGeneId < other.inFieldGeneId;
	}

	std::string ConnectionTuple::toString() const
	{
		return std::format("{}-{}", inFieldGeneId, outFieldGeneId);
	}

	void ConnectionTuple::print() const
	{
		tools::logger::log(tools::logger::INFO, toString());
	}

	ConnectionGeneParameters::ConnectionGeneParameters(const ConnectionTuple connectionTuple, const int innov)
		: connectionTuple(connectionTuple), innovationNumber(innov), enabled(true)
	{}

	ConnectionGeneParameters::ConnectionGeneParameters(const int inFieldGeneId, const int outFieldGeneId, const int innov)
		: connectionTuple(inFieldGeneId, outFieldGeneId), innovationNumber(innov), enabled(true)
	{}

	bool ConnectionGeneParameters::operator==(const ConnectionGeneParameters& other) const
	{
		return connectionTuple == other.connectionTuple &&
			innovationNumber == other.innovationNumber;
	}

	std::string ConnectionGeneParameters::toString() const
	{
		return std::format("{}, innov: {}, enabled: {}",
			connectionTuple.toString(),
			innovationNumber,
			enabled ? "true" : "false");
	}

	void ConnectionGeneParameters::print() const
	{
		tools::logger::log(tools::logger::INFO, toString());
	}

	ConnectionGene::ConnectionGene(const ConnectionTuple connectionTuple, const int innov)
		: parameters(connectionTuple, innov)
	{
		initializeKernel(dnf_composer::element::ElementDimensions{ DimensionConstants::xSize, DimensionConstants::dx });
	}

	ConnectionGene::ConnectionGene(const ConnectionTuple connectionTuple, const int innov,
		const dnf_composer::element::GaussKernelParameters& gkp)
		: parameters(connectionTuple, innov)
	{
		using namespace dnf_composer::element;

		const std::string elementName = std::format("{}{}-{} {}",
			GaussKernelConstants::namePrefixConnectionGene,
			connectionTuple.inFieldGeneId,
			connectionTuple.outFieldGeneId,
			innov);
		const ElementCommonParameters gkcp{ elementName,
			dnf_composer::element::ElementDimensions{DimensionConstants::xSize, DimensionConstants::dx} };
		kernel = std::make_unique<GaussKernel>(gkcp, gkp);
	}

	ConnectionGene::ConnectionGene(const ConnectionTuple connectionTuple, const int innov,
		const dnf_composer::element::MexicanHatKernelParameters& mhkp)
		: parameters(connectionTuple, innov)
	{
		using namespace dnf_composer::element;

		const std::string elementName = std::format("{}{}-{} {}",
			MexicanHatKernelConstants::namePrefixConnectionGene,
			connectionTuple.inFieldGeneId,
			connectionTuple.outFieldGeneId,
			innov);
		const ElementCommonParameters mhkcp{ elementName,
			dnf_composer::element::ElementDimensions{DimensionConstants::xSize, DimensionConstants::dx} };
		kernel = std::make_unique<MexicanHatKernel>(mhkcp, mhkp);
	}

	ConnectionGene::ConnectionGene(const ConnectionGeneParameters& parameters,
		const dnf_composer::element::GaussKernelParameters& gkp)
		: parameters(parameters)
	{
		using namespace dnf_composer::element;

		const std::string elementName = std::format("{}{}-{} {}",
			GaussKernelConstants::namePrefixConnectionGene,
			parameters.connectionTuple.inFieldGeneId,
			parameters.connectionTuple.outFieldGeneId,
			parameters.innovationNumber);
		const ElementCommonParameters gkcp{ elementName,
			dnf_composer::element::ElementDimensions{DimensionConstants::xSize, DimensionConstants::dx} };
		kernel = std::make_unique<GaussKernel>(gkcp, gkp);
	}

	ConnectionGene::ConnectionGene(const ConnectionGeneParameters& parameters,
				const dnf_composer::element::MexicanHatKernelParameters& mhkp)
		: parameters(parameters)
	{
		using namespace dnf_composer::element;

		const std::string elementName = std::format("{}{}-{} {}",
			MexicanHatKernelConstants::namePrefixConnectionGene,
			parameters.connectionTuple.inFieldGeneId,
			parameters.connectionTuple.outFieldGeneId,
			parameters.innovationNumber);
		const ElementCommonParameters mhkcp{ elementName,
					dnf_composer::element::ElementDimensions{DimensionConstants::xSize, DimensionConstants::dx} };
		kernel = std::make_unique<MexicanHatKernel>(mhkcp, mhkp);
	}

	ConnectionGene::ConnectionGene(const ConnectionTuple connectionTuple, const int innov, KernelPtr kernel)
		: parameters(connectionTuple, innov), kernel(std::move(kernel))
	{
		if (!kernel)
		{
			throw std::invalid_argument("Cannot create ConnectionGene with null kernel");
		}
	}

	void ConnectionGene::mutate()
	{
		using namespace dnf_composer::element;

		if (!kernel)
		{
			const std::string message = std::format("Calling mutate() on ConnectionGene with ConnectionTuple: {}-{} but kernel is nullptr.",
                                                parameters.connectionTuple.inFieldGeneId,
                                                parameters.connectionTuple.outFieldGeneId);
			tools::logger::log(tools::logger::FATAL, message);
			throw std::runtime_error(message);
		}

		static constexpr double totalProbability = ConnectionGeneConstants::mutateConnectionGeneKernelProbability +
			ConnectionGeneConstants::mutateConnectionGeneConnectionSignalProbability +
			ConnectionGeneConstants::mutateConnectionGeneKernelTypeProbability;

		constexpr double epsilon = 1e-6;
		if (std::abs(totalProbability - 1.0) > epsilon)
		{
			throw std::runtime_error("Mutation probabilities in connection gene mutation must sum up to 1.");
		}

		const double randomValue = tools::utils::generateRandomDouble(0.0, 1.0);
		if (randomValue < ConnectionGeneConstants::mutateConnectionGeneKernelProbability)
		{
			mutateKernel();
		}
		else if (randomValue < ConnectionGeneConstants::mutateConnectionGeneKernelProbability +
			ConnectionGeneConstants::mutateConnectionGeneConnectionSignalProbability)
		{
			mutateConnectionSignal();
		}
		else
		{
			mutateKernelType();
		}
	}

	void ConnectionGene::clearLastMutations()
	{
		mutationsInLastGeneration = "";
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

	void ConnectionGene::setInnovationNumber(int innovationNumber)
	{
		parameters.innovationNumber = innovationNumber;
	}

	ConnectionGeneParameters ConnectionGene::getParameters() const
	{
		return parameters;
	}

	std::string ConnectionGene::getMutationsInLastGeneration() const
	{
		return mutationsInLastGeneration;
	}

	KernelPtr ConnectionGene::getKernel() const
	{
		return kernel;
	}

	int ConnectionGene::getInnovationNumber() const
	{
		return parameters.innovationNumber;
	}

	int ConnectionGene::getInFieldGeneId() const
	{
		return parameters.connectionTuple.inFieldGeneId;
	}

	int ConnectionGene::getOutFieldGeneId() const
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
		// cg (innov, tuple, enabled)
		std::string result = "cg (";
		result += parameters.toString();
		result += ")";
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
			default:
				break;
		}

		throw std::runtime_error("ConnectionGene::clone() - Kernel type not recognized.");
	}

	void ConnectionGene::initializeKernel(const dnf_composer::element::ElementDimensions& dimensions)
	{
		using namespace neat_dnfs::tools::utils;

		static constexpr double totalProbability = ConnectionGeneConstants::gaussKernelProbability +
			ConnectionGeneConstants::mexicanHatKernelProbability;

		constexpr double epsilon = 1e-6;
		if (std::abs(totalProbability - 1.0) > epsilon)
		{
			throw std::runtime_error("Kernel probabilities in connection gene initialization must sum up to 1.");
		}

		const double randomValue = generateRandomDouble(0.0, 1.0);
		if (randomValue < ConnectionGeneConstants::gaussKernelProbability)
		{
			initializeGaussKernel(dimensions);
		}
		else
		{
			initializeMexicanHatKernel(dimensions);
		}
	}

	void ConnectionGene::initializeGaussKernel(const dnf_composer::element::ElementDimensions& dimensions)
	{
		using namespace dnf_composer::element;
		using namespace neat_dnfs::tools::utils;

		const int amplitude_sign = generateRandomSignal();
		const double width = generateRandomDouble(GaussKernelConstants::widthMinVal, GaussKernelConstants::widthMaxVal);
		const double amplitude = amplitude_sign * generateRandomDouble(GaussKernelConstants::ampMinVal, GaussKernelConstants::ampMaxVal);
		constexpr double amplitudeGlobal = 0.0;
		const GaussKernelParameters gkp{ width,
										amplitude,
											amplitudeGlobal,
									KernelConstants::circularity,
									KernelConstants::normalization
		};
		const std::string elementName = std::format("{}{}-{} {}",
                                                GaussKernelConstants::namePrefixConnectionGene,
                                                parameters.connectionTuple.inFieldGeneId,
                                                parameters.connectionTuple.outFieldGeneId,
                                                parameters.innovationNumber);
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
		constexpr double amplitudeGlobal = 0.0;
		const MexicanHatKernelParameters mhkp{ widthExc,
								amplitudeExc,
								widthInh,
								amplitudeInh,
								amplitudeGlobal,
								KernelConstants::circularity,
								KernelConstants::normalization
		};
		const std::string elementName = std::format("{}{}-{} {}",
                                                MexicanHatKernelConstants::namePrefixConnectionGene,
                                                parameters.connectionTuple.inFieldGeneId,
                                                parameters.connectionTuple.outFieldGeneId,
                                                parameters.innovationNumber);
		const ElementCommonParameters mhcp{ elementName, dimensions};
		kernel = std::make_shared<MexicanHatKernel>(mhcp, mhkp);
	}

	void ConnectionGene::mutateKernel()
	{
		switch (kernel->getLabel())
		{
		case dnf_composer::element::ElementLabel::GAUSS_KERNEL:
			mutateGaussKernel();
			break;
		case dnf_composer::element::ElementLabel::MEXICAN_HAT_KERNEL:
			mutateMexicanHatKernel();
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
			FieldGeneConstants::mexicanHatKernelProbability;

		constexpr double epsilon = 1e-6;
		if (std::abs(totalProbability - 1.0) > epsilon)
		{
			throw std::runtime_error("Mutation probabilities in connection gene kernel type mutation must sum up to 1.");
		}

		const double randomValue = generateRandomDouble(0.0, 1.0);

		if (randomValue < FieldGeneConstants::gaussKernelProbability)
		{
			initializeGaussKernel(dimensions);
			mutationsInLastGeneration += "(cg to gk)";
		}
		else
		{
			initializeMexicanHatKernel(dimensions);
			mutationsInLastGeneration += "(cg to mhk)";
		}
	}

	void ConnectionGene::mutateGaussKernel()
	{
		using namespace dnf_composer::element;
		using namespace neat_dnfs::tools::utils;

		const int signal = generateRandomSignal(); // +/- to add or sum a step

		const auto gaussKernel = std::dynamic_pointer_cast<GaussKernel>(kernel);
		GaussKernelParameters gkp = std::dynamic_pointer_cast<GaussKernel>(kernel)->getParameters();
		const int amp_sign = gkp.amplitude < 0 ? -1 : 1;

		if (generateRandomDouble(0.0, 1.0) < ConnectionGeneConstants::mutateConnectionGeneGaussKernelWidthProbability)
		{
			gkp.width = std::clamp(gkp.width + GaussKernelConstants::widthStep * signal,
				GaussKernelConstants::widthMinVal,
				GaussKernelConstants::widthMaxVal);
			mutationsInLastGeneration += std::format("(cg gk width {})", GaussKernelConstants::widthStep * signal);
		}

		if (generateRandomDouble(0.0, 1.0) < ConnectionGeneConstants::mutateConnectionGeneGaussKernelAmplitudeProbability)
		{
			gkp.amplitude = amp_sign * std::clamp(std::abs(gkp.amplitude) + GaussKernelConstants::ampStep * signal,
				GaussKernelConstants::ampMinVal,
				GaussKernelConstants::ampMaxVal);
			mutationsInLastGeneration += std::format("(cg gk amp. {})", GaussKernelConstants::ampStep * signal);
		}

		if (generateRandomDouble(0.0, 1.0) < ConnectionGeneConstants::mutateConnectionGeneGaussKernelGlobalAmplitudeProbability)
		{
			gkp.amplitudeGlobal = std::clamp(gkp.amplitudeGlobal + GaussKernelConstants::ampGlobalStep * signal,
				GaussKernelConstants::ampGlobalMinVal,
				GaussKernelConstants::ampGlobalMaxVal);
			mutationsInLastGeneration += std::format("(cg gk amp. glob. {})", GaussKernelConstants::ampGlobalStep * signal);
		}
		std::dynamic_pointer_cast<GaussKernel>(kernel)->setParameters(gkp);
	}

	void ConnectionGene::mutateMexicanHatKernel()
	{
		using namespace dnf_composer::element;
		using namespace neat_dnfs::tools::utils;

		const int signal = generateRandomSignal();  // +/- to add or sum a step

		const auto mexicanHatKernel = std::dynamic_pointer_cast<MexicanHatKernel>(kernel);
		MexicanHatKernelParameters mhkp = std::dynamic_pointer_cast<MexicanHatKernel>(kernel)->getParameters();
		const int amp_sign = mhkp.amplitudeExc < 0 ? -1 : 1;

		if (generateRandomDouble(0.0, 1.0) < ConnectionGeneConstants::mutateConnectionGeneMexicanHatKernelAmplitudeExcProbability)
		{
			mhkp.amplitudeExc = amp_sign * std::clamp(std::abs(mhkp.amplitudeExc) + MexicanHatKernelConstants::ampExcStep * signal,
				MexicanHatKernelConstants::ampExcMinVal,
				MexicanHatKernelConstants::ampExcMaxVal);
				mutationsInLastGeneration += std::format("(cg mhk amp. exc. {})", MexicanHatKernelConstants::ampExcStep * signal);
		}

		if (generateRandomDouble(0.0, 1.0) < ConnectionGeneConstants::mutateConnectionGeneMexicanHatKernelWidthExcProbability)
		{
			mhkp.widthExc = std::clamp(mhkp.widthExc + MexicanHatKernelConstants::widthExcStep * signal,
				MexicanHatKernelConstants::widthExcMinVal,
				MexicanHatKernelConstants::widthExcMaxVal);
			mutationsInLastGeneration += std::format("(cg mhk width exc. {})", MexicanHatKernelConstants::widthExcStep * signal);
		}

		if (generateRandomDouble(0.0, 1.0) < ConnectionGeneConstants::mutateConnectionGeneMexicanHatKernelAmplitudeInhProbability)
		{
			mhkp.amplitudeInh = std::clamp(mhkp.amplitudeInh + MexicanHatKernelConstants::ampInhStep * signal,
				MexicanHatKernelConstants::ampInhMinVal,
				MexicanHatKernelConstants::ampInhMaxVal);
			mutationsInLastGeneration += std::format("(cg mhk amp. inh. {})", MexicanHatKernelConstants::ampInhStep * signal);
		}

		if (generateRandomDouble(0.0, 1.0) < ConnectionGeneConstants::mutateConnectionGeneMexicanHatKernelWidthInhProbability)
		{
			mhkp.widthInh = std::clamp(mhkp.widthInh + MexicanHatKernelConstants::widthInhStep * signal,
				MexicanHatKernelConstants::widthInhMinVal,
				MexicanHatKernelConstants::widthInhMaxVal);
			mutationsInLastGeneration += std::format("(cg mhk width inh. {})", MexicanHatKernelConstants::widthInhStep * signal);
		}

		if (generateRandomDouble(0.0, 1.0) < ConnectionGeneConstants::mutateConnectionGeneMexicanHatKernelGlobalAmplitudeProbability)
		{
			mhkp.amplitudeGlobal = std::clamp(mhkp.amplitudeGlobal + MexicanHatKernelConstants::ampGlobStep * signal,
				MexicanHatKernelConstants::ampGlobMin,
				MexicanHatKernelConstants::ampGlobMax);
			mutationsInLastGeneration += std::format("(cg mhk amp. glob. {})", MexicanHatKernelConstants::ampGlobStep * signal);
		}
		std::dynamic_pointer_cast<MexicanHatKernel>(kernel)->setParameters(mhkp);
	}

	void ConnectionGene::mutateConnectionSignal()
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
				const bool amp_sign = gkp.amplitude >= 0;
				mutationsInLastGeneration += std::format("(cg to {})", amp_sign ? "excitatory" : "inhibitory");
			}
			break;
		case MEXICAN_HAT_KERNEL:
			{
				const auto mexicanHatKernel = std::dynamic_pointer_cast<MexicanHatKernel>(kernel);
				MexicanHatKernelParameters mhkp = std::dynamic_pointer_cast<MexicanHatKernel>(kernel)->getParameters();
				mhkp.amplitudeExc = -mhkp.amplitudeExc;
				mexicanHatKernel->setParameters(mhkp);
				const bool amp_sign = mhkp.amplitudeExc >=0;
				mutationsInLastGeneration += std::format("(cg to {})", amp_sign ? "excitatory" : "inhibitory");
			}
			break;
		default:
			tools::logger::log(tools::logger::FATAL, "ConnectionGene::mutate() - Kernel type not recognized.");
			throw std::runtime_error("ConnectionGene::mutate() - Kernel type not recognized.");
		}
	}
}