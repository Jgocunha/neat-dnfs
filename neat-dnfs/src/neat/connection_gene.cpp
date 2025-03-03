#include "neat/connection_gene.h"


namespace neat_dnfs
{
	ConnectionGeneStatistics ConnectionGene::statistics;

	ConnectionTuple::ConnectionTuple(int inFieldGeneId, int outFieldGeneId)
		: inFieldGeneId(inFieldGeneId), outFieldGeneId(outFieldGeneId)
	{}

	bool ConnectionTuple::operator==(const ConnectionTuple& other) const
	{
		return inFieldGeneId == other.inFieldGeneId && outFieldGeneId == other.outFieldGeneId;
	}

	bool ConnectionTuple::operator<(const ConnectionTuple& other) const {
		if (inFieldGeneId == other.inFieldGeneId)
			return outFieldGeneId < other.outFieldGeneId;
		return inFieldGeneId < other.inFieldGeneId;
	}

	std::string ConnectionTuple::toString() const
	{
		return std::to_string(inFieldGeneId) + "-" + std::to_string(outFieldGeneId);
	}

	void ConnectionTuple::print() const
	{
		tools::logger::log(tools::logger::INFO, toString());
	}

	ConnectionGeneParameters::ConnectionGeneParameters(ConnectionTuple connectionTuple, int innov)
		: connectionTuple(connectionTuple), innovationNumber(innov), enabled(true)
	{}

	ConnectionGeneParameters::ConnectionGeneParameters(int inFieldGeneId, int outFieldGeneId, int innov)
		: connectionTuple(inFieldGeneId, outFieldGeneId), innovationNumber(innov), enabled(true)
	{}

	bool ConnectionGeneParameters::operator==(const ConnectionGeneParameters& other) const
	{
		return connectionTuple == other.connectionTuple &&
			innovationNumber == other.innovationNumber;
	}

	std::string ConnectionGeneParameters::toString() const
	{
		return connectionTuple.toString() +
			", innov: " + std::to_string(innovationNumber) +
			", enabled: " + (enabled ? "true" : "false");
	}

	void ConnectionGeneParameters::print() const
	{
		tools::logger::log(tools::logger::INFO, toString());
	}

	void ConnectionGeneStatistics::resetPerGenerationStatistics()
	{
		numKernelMutationsPerGeneration = 0;
		numKernelTypeMutationsPerGeneration = 0;
		numGaussKernelMutationsPerGeneration = 0;
		numMexicanHatKernelMutationsPerGeneration = 0;
		numOscillatoryKernelMutationsPerGeneration = 0;
		numConnectionSignalMutationsPerGeneration = 0;
	}

	std::string ConnectionGeneStatistics::toString() const
	{
		std::string result = "ConnectionGeneStatistics: {";
		result += "Kernel mutations per generation: " + std::to_string(numKernelMutationsPerGeneration);
		result += ", Kernel type mutations per generation: " + std::to_string(numKernelTypeMutationsPerGeneration);
		result += ", Gauss kernel mutations per generation: " + std::to_string(numGaussKernelMutationsPerGeneration);
		result += ", Mexican hat kernel mutations per generation: " + std::to_string(numMexicanHatKernelMutationsPerGeneration);
		result += ", Oscillatory kernel mutations per generation: " + std::to_string(numOscillatoryKernelMutationsPerGeneration);
		result += ", Connection signal mutations per generation: " + std::to_string(numConnectionSignalMutationsPerGeneration);
		result += ", Total kernel mutations: " + std::to_string(numKernelMutationsTotal);
		result += ", Total kernel type mutations: " + std::to_string(numKernelTypeMutationsTotal);
		result += ", Total Gauss kernel mutations: " + std::to_string(numGaussKernelMutationsTotal);
		result += ", Total Mexican hat kernel mutations: " + std::to_string(numMexicanHatKernelMutationsTotal);
		result += ", Total Oscillatory kernel mutations: " + std::to_string(numOscillatoryKernelMutationsTotal);
		result += ", Total connection signal mutations: " + std::to_string(numConnectionSignalMutationsTotal);
		result += " }";
		return result;
	}

	void ConnectionGeneStatistics::print() const
	{
		tools::logger::log(tools::logger::INFO, toString());
	}

	void ConnectionGeneStatistics::savePerGeneration(const std::string& directory) const
	{
		std::ofstream logFile(directory + "connection_gene_statistics_per_generation.txt", std::ios::app);
		if (logFile.is_open())
		{
			logFile << "Kernel mutations per generation: " + std::to_string(numKernelMutationsPerGeneration);
			logFile << ", Kernel type mutations per generation: " + std::to_string(numKernelTypeMutationsPerGeneration);
			logFile << ", Gauss kernel mutations per generation: " + std::to_string(numGaussKernelMutationsPerGeneration);
			logFile << ", Mexican hat kernel mutations per generation: " + std::to_string(numMexicanHatKernelMutationsPerGeneration);
			logFile << ", Oscillatory kernel mutations per generation: " + std::to_string(numOscillatoryKernelMutationsPerGeneration);
			logFile << ", Connection signal mutations per generation: " + std::to_string(numConnectionSignalMutationsPerGeneration);
			logFile << "\n";
			logFile.close();
		}
		else
		{
			tools::logger::log(tools::logger::LogLevel::ERROR, "Failed to open log file for field gene per generation statistics.");
		}
	}

	void ConnectionGeneStatistics::saveTotal(const std::string& directory) const
	{
		std::ofstream logFile(directory + "field_gene_statistics_total.txt", std::ios::app);
		if (logFile.is_open())
		{
			logFile << ", Total kernel mutations: " + std::to_string(numKernelMutationsTotal);
			logFile << ", Total kernel type mutations: " + std::to_string(numKernelTypeMutationsTotal);
			logFile << ", Total Gauss kernel mutations: " + std::to_string(numGaussKernelMutationsTotal);
			logFile << ", Total Mexican hat kernel mutations: " + std::to_string(numMexicanHatKernelMutationsTotal);
			logFile << ", Total Oscillatory kernel mutations: " + std::to_string(numOscillatoryKernelMutationsTotal);
			logFile << ", Total connection signal mutations: " + std::to_string(numConnectionSignalMutationsTotal);
			logFile.close();
		}
		else
		{
			tools::logger::log(tools::logger::LogLevel::ERROR, "Failed to open log file for field gene total statistics.");
		}
	}

	ConnectionGene::ConnectionGene(ConnectionTuple connectionTuple, int innov)
		: parameters(connectionTuple, innov)
	{
		initializeKernel({ DimensionConstants::xSize, DimensionConstants::dx });
	}

	ConnectionGene::ConnectionGene(ConnectionTuple connectionTuple, int innov,
		const dnf_composer::element::GaussKernelParameters& gkp)
		: parameters(connectionTuple, innov)
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

	ConnectionGene::ConnectionGene(ConnectionTuple connectionTuple, int innov,
		const dnf_composer::element::MexicanHatKernelParameters& mhkp)
		: parameters(connectionTuple, innov)
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

	ConnectionGene::ConnectionGene(ConnectionTuple connectionTuple, int innov,
		const dnf_composer::element::OscillatoryKernelParameters& osckp)
		: parameters(connectionTuple, innov)
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

	ConnectionGene::ConnectionGene(ConnectionTuple connectionTuple, int innov, KernelPtr kernel)
		: parameters(connectionTuple, innov), kernel(std::move(kernel))
	{
		if (!kernel)
			throw std::invalid_argument("Cannot create ConnectionGene with null kernel");
	}

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

		static constexpr double totalProbability = ConnectionGeneConstants::mutateConnectionGeneKernelProbability +
			ConnectionGeneConstants::mutateConnectionGeneConnectionSignalProbability +
			ConnectionGeneConstants::mutateConnectionGeneKernelTypeProbability;

		constexpr double epsilon = 1e-6;
		if (std::abs(totalProbability - 1.0) > epsilon)
			throw std::runtime_error("Mutation probabilities in connection gene mutation must sum up to 1.");

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

	void ConnectionGene::resetMutationStatisticsPerGeneration()
	{
		statistics.resetPerGenerationStatistics();
	}

	ConnectionGeneStatistics ConnectionGene::getStatistics()
	{
		return statistics;
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
		constexpr double amplitudeGlobal = 0.0f;
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
		constexpr double amplitudeGlobal = 0.0f;
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
		constexpr double amplitudeGlobal = 0.0f;
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
		statistics.numKernelMutationsPerGeneration++;
		statistics.numKernelMutationsTotal++;
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
		statistics.numKernelTypeMutationsPerGeneration++;
		statistics.numKernelTypeMutationsTotal++;
	}

	void ConnectionGene::mutateGaussKernel() const
	{
		using namespace dnf_composer::element;
		using namespace neat_dnfs::tools::utils;

		const int signal = generateRandomSignal(); // +/- to add or sum a step

		const auto gaussKernel = std::dynamic_pointer_cast<GaussKernel>(kernel);
		GaussKernelParameters gkp = std::dynamic_pointer_cast<GaussKernel>(kernel)->getParameters();
		const int amp_sign = gkp.amplitude < 0 ? -1 : 1;

		constexpr double totalProbability = ConnectionGeneConstants::mutateConnectionGeneGaussKernelWidthProbability +
			ConnectionGeneConstants::mutateConnectionGeneGaussKernelAmplitudeProbability +
			ConnectionGeneConstants::mutateConnectionGeneGaussKernelGlobalAmplitudeProbability;

		constexpr double epsilon = 1e-6;
		if (std::abs(totalProbability - 1.0) > epsilon)
			throw std::runtime_error("Mutation probabilities in field gene gauss kernel mutation must sum up to 1.");


		const double mutationSelection = generateRandomDouble(0.0, 1.0);

		if (mutationSelection < ConnectionGeneConstants::mutateConnectionGeneGaussKernelWidthProbability)
		{
			gkp.width = std::clamp(gkp.width + GaussKernelConstants::widthStep * signal,
				GaussKernelConstants::widthMinVal,
				GaussKernelConstants::widthMaxVal);
		}
		else if (mutationSelection < ConnectionGeneConstants::mutateConnectionGeneGaussKernelWidthProbability +
			ConnectionGeneConstants::mutateConnectionGeneGaussKernelAmplitudeProbability)
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
		statistics.numGaussKernelMutationsPerGeneration++;
		statistics.numGaussKernelMutationsTotal++;
	}

	void ConnectionGene::mutateMexicanHatKernel() const
	{
		using namespace dnf_composer::element;
		using namespace neat_dnfs::tools::utils;

		const int signal = generateRandomSignal();  // +/- to add or sum a step

		const auto mexicanHatKernel = std::dynamic_pointer_cast<MexicanHatKernel>(kernel);
		MexicanHatKernelParameters mhkp = std::dynamic_pointer_cast<MexicanHatKernel>(kernel)->getParameters();
		const int amp_sign = mhkp.amplitudeExc < 0 ? -1 : 1;

		constexpr double totalProbability = ConnectionGeneConstants::mutateConnectionGeneMexicanHatKernelAmplitudeExcProbability +
			ConnectionGeneConstants::mutateConnectionGeneMexicanHatKernelWidthExcProbability +
			ConnectionGeneConstants::mutateConnectionGeneMexicanHatKernelAmplitudeInhProbability +
			ConnectionGeneConstants::mutateConnectionGeneMexicanHatKernelWidthInhProbability +
			ConnectionGeneConstants::mutateConnectionGeneMexicanHatKernelGlobalAmplitudeProbability;

		constexpr double epsilon = 1e-6;
		if (std::abs(totalProbability - 1.0) > epsilon)
			throw std::runtime_error("Mutation probabilities in field gene mexican-hat kernel mutation must sum up to 1.");


		const double mutationSelection = generateRandomDouble(0.0, 1.0);
		if (mutationSelection < ConnectionGeneConstants::mutateConnectionGeneMexicanHatKernelAmplitudeExcProbability)
		{
			mhkp.amplitudeExc = amp_sign * std::clamp(mhkp.amplitudeExc + MexicanHatKernelConstants::ampExcStep * signal,
				MexicanHatKernelConstants::ampExcMinVal,
				MexicanHatKernelConstants::ampExcMaxVal);
		}
		else if (mutationSelection < ConnectionGeneConstants::mutateConnectionGeneMexicanHatKernelAmplitudeExcProbability +
			ConnectionGeneConstants::mutateConnectionGeneMexicanHatKernelWidthExcProbability)
		{
			mhkp.widthExc = std::clamp(mhkp.widthExc + MexicanHatKernelConstants::widthExcStep * signal,
				MexicanHatKernelConstants::widthExcMinVal,
				MexicanHatKernelConstants::widthExcMaxVal);
		}
		else if (mutationSelection < ConnectionGeneConstants::mutateConnectionGeneMexicanHatKernelAmplitudeExcProbability +
			ConnectionGeneConstants::mutateConnectionGeneMexicanHatKernelWidthExcProbability +
			ConnectionGeneConstants::mutateConnectionGeneMexicanHatKernelAmplitudeInhProbability)
		{
			mhkp.amplitudeInh = std::clamp(mhkp.amplitudeInh + MexicanHatKernelConstants::ampInhStep * signal,
				MexicanHatKernelConstants::ampInhMinVal,
				MexicanHatKernelConstants::ampInhMaxVal);
		}
		else if (mutationSelection < ConnectionGeneConstants::mutateConnectionGeneMexicanHatKernelAmplitudeExcProbability +
			ConnectionGeneConstants::mutateConnectionGeneMexicanHatKernelWidthExcProbability +
			ConnectionGeneConstants::mutateConnectionGeneMexicanHatKernelAmplitudeInhProbability +
			ConnectionGeneConstants::mutateConnectionGeneMexicanHatKernelWidthInhProbability)
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
		statistics.numMexicanHatKernelMutationsPerGeneration++;
		statistics.numMexicanHatKernelMutationsTotal++;
	}

	void ConnectionGene::mutateOscillatoryKernel() const
	{
		using namespace dnf_composer::element;
		using namespace neat_dnfs::tools::utils;

		const int signal = generateRandomSignal();  // +/- to add or sum a step

		const auto oscillatoryKernel = std::dynamic_pointer_cast<OscillatoryKernel>(kernel);
		OscillatoryKernelParameters okp = std::dynamic_pointer_cast<OscillatoryKernel>(kernel)->getParameters();
		const int amp_sign = okp.amplitude < 0 ? -1 : 1;

		constexpr double totalProbability = ConnectionGeneConstants::mutateConnectionGeneOscillatoryKernelAmplitudeProbability +
			ConnectionGeneConstants::mutateConnectionGeneOscillatoryKernelDecayProbability +
			ConnectionGeneConstants::mutateConnectionGeneOscillatoryKernelZeroCrossingsProbability +
			ConnectionGeneConstants::mutateConnectionGeneOscillatoryKernelGlobalAmplitudeProbability;

		constexpr double epsilon = 1e-6;
		if (std::abs(totalProbability - 1.0) > epsilon)
			throw std::runtime_error("Mutation probabilities in field gene oscillatory kernel mutation must sum up to 1.");


		const double mutationSelection = generateRandomDouble(0.0, 1.0);


		if (mutationSelection < ConnectionGeneConstants::mutateConnectionGeneOscillatoryKernelAmplitudeProbability)
		{
			okp.amplitude = amp_sign * std::clamp(okp.amplitude + OscillatoryKernelConstants::amplitudeStep * signal,
				OscillatoryKernelConstants::amplitudeMinVal,
				OscillatoryKernelConstants::amplitudeMaxVal);
		}
		else if (mutationSelection < ConnectionGeneConstants::mutateConnectionGeneOscillatoryKernelAmplitudeProbability +
			ConnectionGeneConstants::mutateConnectionGeneOscillatoryKernelDecayProbability)
		{
			okp.decay = std::clamp(okp.decay + OscillatoryKernelConstants::decayStep * signal,
				OscillatoryKernelConstants::decayMinVal,
				OscillatoryKernelConstants::decayMaxVal);
		}
		else if (mutationSelection < ConnectionGeneConstants::mutateConnectionGeneOscillatoryKernelAmplitudeProbability +
			ConnectionGeneConstants::mutateConnectionGeneOscillatoryKernelDecayProbability +
			ConnectionGeneConstants::mutateConnectionGeneOscillatoryKernelZeroCrossingsProbability)
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
		statistics.numOscillatoryKernelMutationsPerGeneration++;
		statistics.numOscillatoryKernelMutationsTotal++;
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
		statistics.numConnectionSignalMutationsPerGeneration++;
		statistics.numConnectionSignalMutationsTotal++;
	}
}