#include "neat/field_gene.h"

namespace neat_dnfs
{
	FieldGeneStatistics FieldGene::statistics;

	FieldGeneParameters::FieldGeneParameters(FieldGeneType type, int id)
		: type(type), id(id)
	{}

	bool FieldGeneParameters::operator==(const FieldGeneParameters& other) const
	{
		return type == other.type && id == other.id;
	}

	std::string FieldGeneParameters::toString() const
	{
		std::string typeStr;
		switch (type)
		{
		case FieldGeneType::INPUT:
			typeStr = "INPUT";
			break;
		case FieldGeneType::OUTPUT:
			typeStr = "OUTPUT";
			break;
		case FieldGeneType::HIDDEN:
			typeStr = "HIDDEN";
			break;
		}
		return "id: " + std::to_string(id) + ", type: " + typeStr;
	}

	void FieldGeneParameters::print() const
	{
		tools::logger::log(tools::logger::INFO, toString());
	}

	void FieldGeneStatistics::resetPerGenerationStatistics()
	{
		numNeuralFieldMutationsPerGeneration = 0;
		numKernelMutationsPerGeneration = 0;
		numKernelTypeMutationsPerGeneration = 0;
		numGaussKernelMutationsPerGeneration = 0;
		numMexicanHatKernelMutationsPerGeneration = 0;
		numOscillatoryKernelMutationsPerGeneration = 0;
	}

	std::string FieldGeneStatistics::toString() const
	{
		std::string result = "FieldGeneStatistics: {";
		result += "Kernel mutations per generation: " + std::to_string(numKernelMutationsPerGeneration);
		result += ", Kernel type mutations per generation: " + std::to_string(numKernelTypeMutationsPerGeneration);
		result += ", Gauss kernel mutations per generation: " + std::to_string(numGaussKernelMutationsPerGeneration);
		result += ", Mexican hat kernel mutations per generation: " + std::to_string(numMexicanHatKernelMutationsPerGeneration);
		result += ", Oscillatory kernel mutations per generation: " + std::to_string(numOscillatoryKernelMutationsPerGeneration);
		result += ", Neural field mutations per generation: " + std::to_string(numNeuralFieldMutationsPerGeneration);
		result += ", Total kernel mutations: " + std::to_string(numKernelMutationsTotal);
		result += ", Total kernel type mutations: " + std::to_string(numKernelTypeMutationsTotal);
		result += ", Total Gauss kernel mutations: " + std::to_string(numGaussKernelMutationsTotal);
		result += ", Total Mexican hat kernel mutations: " + std::to_string(numMexicanHatKernelMutationsTotal);
		result += ", Total Oscillatory kernel mutations: " + std::to_string(numOscillatoryKernelMutationsTotal);
		result += ", Total neural field mutations: " + std::to_string(numNeuralFieldMutationsTotal);
		result += " }";
		return result;
	}

	void FieldGeneStatistics::print() const
	{
		tools::logger::log(tools::logger::INFO, toString());
	}

	void FieldGeneStatistics::savePerGeneration(const std::string& directory) const
	{
		std::ofstream logFile(directory + "field_gene_statistics_per_generation.txt", std::ios::app);
		if (logFile.is_open())
		{
			logFile << "Kernel mutations per generation: " + std::to_string(numKernelMutationsPerGeneration);
			logFile << ", Kernel type mutations per generation: " + std::to_string(numKernelTypeMutationsPerGeneration);
			logFile << ", Gauss kernel mutations per generation: " + std::to_string(numGaussKernelMutationsPerGeneration);
			logFile << ", Mexican hat kernel mutations per generation: " + std::to_string(numMexicanHatKernelMutationsPerGeneration);
			logFile << ", Oscillatory kernel mutations per generation: " + std::to_string(numOscillatoryKernelMutationsPerGeneration);
			logFile << ", Neural field mutations per generation: " + std::to_string(numNeuralFieldMutationsPerGeneration);
			logFile << "\n";
			logFile.close();
		}
		else
		{
			tools::logger::log(tools::logger::LogLevel::ERROR, "Failed to open log file for field gene per generation statistics.");
		}
	}

	void FieldGeneStatistics::saveTotal(const std::string& directory) const
	{
		std::ofstream logFile(directory + "field_gene_statistics_total.txt", std::ios::app);
		if (logFile.is_open())
		{
			logFile << ", Total kernel mutations: " + std::to_string(numKernelMutationsTotal);
			logFile << ", Total kernel type mutations: " + std::to_string(numKernelTypeMutationsTotal);
			logFile << ", Total Gauss kernel mutations: " + std::to_string(numGaussKernelMutationsTotal);
			logFile << ", Total Mexican hat kernel mutations: " + std::to_string(numMexicanHatKernelMutationsTotal);
			logFile << ", Total Oscillatory kernel mutations: " + std::to_string(numOscillatoryKernelMutationsTotal);
			logFile << ", Total neural field mutations: " + std::to_string(numNeuralFieldMutationsTotal);
			logFile.close();
		}
		else
		{
			tools::logger::log(tools::logger::LogLevel::ERROR, "Failed to open log file for field gene total statistics.");
		}
	}

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

	FieldGene::FieldGene(const FieldGeneParameters& parameters, const NeuralFieldPtr& neuralField, KernelPtr kernel)
		: parameters(parameters), neuralField(neuralField), kernel(std::move(kernel))
	{
		initializeNoise(neuralField->getElementCommonParameters().dimensionParameters);
	}

	FieldGene::FieldGene(const FieldGeneParameters& parameters, const FieldGene& other)
		: parameters(parameters)
	{
		using namespace dnf_composer::element;
		const ElementDimensions dimensions = other.getNeuralField()->getElementCommonParameters().dimensionParameters;

		const std::shared_ptr<NeuralField> nf = other.getNeuralField();
		const ElementCommonParameters nfcp{ NeuralFieldConstants::namePrefix + std::to_string(parameters.id), dimensions };
		neuralField = std::make_shared<NeuralField>(nfcp, nf->getParameters());

		const auto k = other.getKernel();
		switch (k->getLabel())
		{
			case ElementLabel::GAUSS_KERNEL:
				{
					const auto gkp = std::dynamic_pointer_cast<GaussKernel>(k)->getParameters();
					const ElementCommonParameters gkcp{ GaussKernelConstants::namePrefix + std::to_string(parameters.id), dimensions };
					kernel = std::make_shared<GaussKernel>(gkcp, gkp);
				}
				break;
			case ElementLabel::MEXICAN_HAT_KERNEL:
				{
					const auto mhkp = std::dynamic_pointer_cast<MexicanHatKernel>(k)->getParameters();
					const ElementCommonParameters mhcp{ MexicanHatKernelConstants::namePrefix + std::to_string(parameters.id), dimensions };
					kernel = std::make_shared<MexicanHatKernel>(mhcp, mhkp);
				}
				break;
			case ElementLabel::OSCILLATORY_KERNEL:
				{
					const auto oskp = std::dynamic_pointer_cast<OscillatoryKernel>(k)->getParameters();
					const ElementCommonParameters oskcp{ OscillatoryKernelConstants::namePrefix + std::to_string(parameters.id), dimensions };
					kernel = std::make_shared<OscillatoryKernel>(oskcp, oskp);
				}
				break;
			default:				
				tools::logger::log(tools::logger::FATAL, "FieldGene::FieldGene() - Kernel type not recognized.");
				throw std::runtime_error("FieldGene::FieldGene() - Kernel type not recognized.");
		}

		const std::shared_ptr<NormalNoise> nn = other.getNoise();
		const ElementCommonParameters nncp{ NoiseConstants::namePrefix + std::to_string(parameters.id), dimensions };
		noise = std::make_shared<NormalNoise>(nncp, nn->getParameters());
	}

	void FieldGene::setAsInput(const dnf_composer::element::ElementDimensions& dimensions)
	{
		parameters.type = FieldGeneType::INPUT;
		initializeNeuralField(dimensions);
		initializeKernel(dimensions);
		initializeNoise(dimensions);
	}

	void FieldGene::setAsOutput(const dnf_composer::element::ElementDimensions& dimensions)
	{
		parameters.type = FieldGeneType::OUTPUT;
		initializeNeuralField(dimensions);
		initializeKernel(dimensions);
		initializeNoise(dimensions);
	}

	void FieldGene::setAsHidden(const dnf_composer::element::ElementDimensions& dimensions)
	{
		parameters.type = FieldGeneType::HIDDEN;

		initializeNeuralField(dimensions);
		initializeKernel(dimensions);
		initializeNoise(dimensions);
	}

	void FieldGene::mutate()
	{
		static constexpr double totalProbability = FieldGeneConstants::mutateFielGeneKernelProbability +
			FieldGeneConstants::mutateFieldGeneNeuralFieldProbability +
			FieldGeneConstants::mutateFielGeneKernelTypeProbability;

		constexpr double epsilon = 1e-6;
		if (std::abs(totalProbability - 1.0) > epsilon)
			throw std::runtime_error("Mutation probabilities in field gene mutation must sum up to 1.");

		const double randomValue = tools::utils::generateRandomDouble(0.0, 1.0);
		if (randomValue < FieldGeneConstants::mutateFielGeneKernelProbability)
		{
			mutateKernel();
		}
		else if (randomValue < FieldGeneConstants::mutateFielGeneKernelProbability +
			FieldGeneConstants::mutateFieldGeneNeuralFieldProbability)
		{
			mutateNeuralField();
		}
		else
		{
			mutateKernelType();
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

	std::shared_ptr<dnf_composer::element::NormalNoise> FieldGene::getNoise() const
	{
		return noise;
	}

	void FieldGene::resetMutationStatisticsPerGeneration()
	{
		statistics.resetPerGenerationStatistics();
	}

	FieldGeneStatistics FieldGene::getStatistics()
	{
		return statistics;
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
		// fg (id, type)
		std::string result = "fg (";
		result += parameters.toString();
		result += ")";
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

		if(FieldGeneConstants::variableParameters)
		{
			const double tau = generateRandomDouble(NeuralFieldConstants::tauMinVal, NeuralFieldConstants::tauMaxVal);
			const double restingLevel = generateRandomDouble(NeuralFieldConstants::restingLevelMinVal, NeuralFieldConstants::restingLevelMaxVal);

			const NeuralFieldParameters nfp{ tau, restingLevel, NeuralFieldConstants::activationFunction };
			const ElementCommonParameters nfcp{ NeuralFieldConstants::namePrefix + std::to_string(parameters.id), dimensions };
			neuralField = std::make_shared<NeuralField>(nfcp, nfp);
		}
		else
		{
			constexpr double tau = NeuralFieldConstants::tau;
			constexpr double restingLevel = NeuralFieldConstants::restingLevel;
			const NeuralFieldParameters nfp{ tau, restingLevel, NeuralFieldConstants::activationFunction };
			const ElementCommonParameters nfcp{ NeuralFieldConstants::namePrefix + std::to_string(parameters.id), dimensions };
			neuralField = std::make_shared<NeuralField>(nfcp, nfp);
		}
		neuralField->setThresholdForStability(NeuralFieldConstants::stabilityThreshold);
	}

	void FieldGene::initializeKernel(const dnf_composer::element::ElementDimensions& dimensions)
	{
		using namespace neat_dnfs::tools::utils;

		static constexpr double totalProbability = FieldGeneConstants::gaussKernelProbability +
			FieldGeneConstants::mexicanHatKernelProbability +
			FieldGeneConstants::oscillatoryKernelProbability;

		constexpr double epsilon = 1e-6;
		if (std::abs(totalProbability - 1.0) > epsilon)
			throw std::runtime_error("Kernel probabilities in field gene initialization must sum up to 1.");

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

	void FieldGene::initializeGaussKernel(const dnf_composer::element::ElementDimensions& dimensions)
	{
		using namespace dnf_composer::element;
		using namespace neat_dnfs::tools::utils;

		if (FieldGeneConstants::variableParameters)
		{
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
							dimensions };
			kernel = std::make_shared<GaussKernel>(gkcp, gkp);
		}
		else
		{
			constexpr double width = GaussKernelConstants::width;
			constexpr double amplitude = GaussKernelConstants::amplitude;
			constexpr double amplitudeGlobal = GaussKernelConstants::amplitudeGlobal;
			const GaussKernelParameters gkp{ width, amplitude, amplitudeGlobal,
										KernelConstants::circularity,
										KernelConstants::normalization
			};
			const ElementCommonParameters gkcp{ GaussKernelConstants::namePrefix + std::to_string(parameters.id),
							dimensions };
			kernel = std::make_shared<GaussKernel>(gkcp, gkp);
		}
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

	void FieldGene::initializeNoise(const dnf_composer::element::ElementDimensions& dimensions)
	{
		using namespace dnf_composer::element;
		using namespace neat_dnfs::tools::utils;

		const NormalNoiseParameters nnp{ NoiseConstants::amplitude };
		const ElementCommonParameters nncp{  NoiseConstants::namePrefix + std::to_string(parameters.id), dimensions };
		noise = std::make_shared<NormalNoise>(nncp, nnp);
	}

	void FieldGene::mutateKernel() const
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
			tools::logger::log(tools::logger::FATAL, "FieldGene::mutate() - Kernel type not recognized.");
			throw std::runtime_error("FieldGene::mutate() - Kernel type not recognized.");
		}
		statistics.numKernelMutationsPerGeneration++;
		statistics.numKernelMutationsTotal++;
	}

	void FieldGene::mutateGaussKernel() const
	{
		using namespace dnf_composer::element;
		using namespace neat_dnfs::tools::utils;

		const int signal = generateRandomSignal(); // +/- to add or sum a step

		const auto gaussKernel = std::dynamic_pointer_cast<GaussKernel>(kernel);
		GaussKernelParameters gkp = std::dynamic_pointer_cast<GaussKernel>(kernel)->getParameters();

		constexpr double totalProbability = FieldGeneConstants::mutateFieldGeneGaussKernelWidthProbability +
			FieldGeneConstants::mutateFieldGeneGaussKernelAmplitudeProbability +
			FieldGeneConstants::mutateFieldGeneGaussKernelGlobalAmplitudeProbability;

		constexpr double epsilon = 1e-6;
		if (std::abs(totalProbability - 1.0) > epsilon)
			throw std::runtime_error("Mutation probabilities in field gene gauss kernel mutation must sum up to 1.");


		const double mutationSelection = generateRandomDouble(0.0, 1.0);

		if (mutationSelection < FieldGeneConstants::mutateFieldGeneGaussKernelWidthProbability)
		{
			gkp.width = std::clamp(gkp.width + GaussKernelConstants::widthStep * signal,
								GaussKernelConstants::widthMinVal,
								GaussKernelConstants::widthMaxVal);
		}
		else if (mutationSelection < FieldGeneConstants::mutateFieldGeneGaussKernelWidthProbability +
			FieldGeneConstants::mutateFieldGeneGaussKernelAmplitudeProbability)
		{
			gkp.amplitude = std::clamp(gkp.amplitude + GaussKernelConstants::ampStep * signal,
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

	void FieldGene::mutateMexicanHatKernel() const
	{
		using namespace dnf_composer::element;
		using namespace neat_dnfs::tools::utils;

		const int signal = generateRandomSignal();  // +/- to add or sum a step

		const auto mexicanHatKernel = std::dynamic_pointer_cast<MexicanHatKernel>(kernel);
		MexicanHatKernelParameters mhkp = std::dynamic_pointer_cast<MexicanHatKernel>(kernel)->getParameters();

		constexpr double totalProbability = FieldGeneConstants::mutateFieldGeneMexicanHatKernelAmplitudeExcProbability +
			FieldGeneConstants::mutateFieldGeneMexicanHatKernelWidthExcProbability +
			FieldGeneConstants::mutateFieldGeneMexicanHatKernelAmplitudeInhProbability +
			FieldGeneConstants::mutateFieldGeneMexicanHatKernelWidthInhProbability +
			FieldGeneConstants::mutateFieldGeneMexicanHatKernelGlobalAmplitudeProbability;

		constexpr double epsilon = 1e-6;
		if (std::abs(totalProbability - 1.0) > epsilon)
			throw std::runtime_error("Mutation probabilities in field gene mexican-hat kernel mutation must sum up to 1.");


		const double mutationSelection = generateRandomDouble(0.0, 1.0);
		if(mutationSelection < FieldGeneConstants::mutateFieldGeneMexicanHatKernelAmplitudeExcProbability)
		{
			mhkp.amplitudeExc = std::clamp(mhkp.amplitudeExc + MexicanHatKernelConstants::ampExcStep * signal,
								MexicanHatKernelConstants::ampExcMinVal,
								MexicanHatKernelConstants::ampExcMaxVal);
		}
		else if (mutationSelection < FieldGeneConstants::mutateFieldGeneMexicanHatKernelAmplitudeExcProbability +
			FieldGeneConstants::mutateFieldGeneMexicanHatKernelWidthExcProbability)
		{
			mhkp.widthExc = std::clamp(mhkp.widthExc + MexicanHatKernelConstants::widthExcStep * signal,
												MexicanHatKernelConstants::widthExcMinVal,
												MexicanHatKernelConstants::widthExcMaxVal);
		}
		else if (mutationSelection < FieldGeneConstants::mutateFieldGeneMexicanHatKernelAmplitudeExcProbability +
			FieldGeneConstants::mutateFieldGeneMexicanHatKernelWidthExcProbability +
			FieldGeneConstants::mutateFieldGeneMexicanHatKernelAmplitudeInhProbability)
		{
			mhkp.amplitudeInh = std::clamp(mhkp.amplitudeInh + MexicanHatKernelConstants::ampInhStep * signal,
																MexicanHatKernelConstants::ampInhMinVal,
																MexicanHatKernelConstants::ampInhMaxVal);
		}
		else if (mutationSelection < FieldGeneConstants::mutateFieldGeneMexicanHatKernelAmplitudeExcProbability +
						FieldGeneConstants::mutateFieldGeneMexicanHatKernelWidthExcProbability +
						FieldGeneConstants::mutateFieldGeneMexicanHatKernelAmplitudeInhProbability +
						FieldGeneConstants::mutateFieldGeneMexicanHatKernelWidthInhProbability)
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

	void FieldGene::mutateOscillatoryKernel() const
	{
		using namespace dnf_composer::element;
		using namespace neat_dnfs::tools::utils;

		const int signal = generateRandomSignal();  // +/- to add or sum a step

		const auto oscillatoryKernel = std::dynamic_pointer_cast<OscillatoryKernel>(kernel);
		OscillatoryKernelParameters okp = std::dynamic_pointer_cast<OscillatoryKernel>(kernel)->getParameters();

		constexpr double totalProbability = FieldGeneConstants::mutateFieldGeneOscillatoryKernelAmplitudeProbability +
			FieldGeneConstants::mutateFieldGeneOscillatoryKernelDecayProbability +
			FieldGeneConstants::mutateFieldGeneOscillatoryKernelZeroCrossingsProbability +
			FieldGeneConstants::mutateFieldGeneOscillatoryKernelGlobalAmplitudeProbability;

		constexpr double epsilon = 1e-6;
		if (std::abs(totalProbability - 1.0) > epsilon)
			throw std::runtime_error("Mutation probabilities in field gene oscillatory kernel mutation must sum up to 1.");


		const double mutationSelection = generateRandomDouble(0.0, 1.0);


		if ( mutationSelection < FieldGeneConstants::mutateFieldGeneOscillatoryKernelAmplitudeProbability)
		{
			okp.amplitude = std::clamp(okp.amplitude + OscillatoryKernelConstants::amplitudeStep * signal,
											OscillatoryKernelConstants::amplitudeMinVal,
											OscillatoryKernelConstants::amplitudeMaxVal);
		}
		else if (mutationSelection < FieldGeneConstants::mutateFieldGeneOscillatoryKernelAmplitudeProbability +
			FieldGeneConstants::mutateFieldGeneOscillatoryKernelDecayProbability)
		{
			okp.decay = std::clamp(okp.decay + OscillatoryKernelConstants::decayStep * signal,
											OscillatoryKernelConstants::decayMinVal,
											OscillatoryKernelConstants::decayMaxVal);
		}
		else if (mutationSelection < FieldGeneConstants::mutateFieldGeneOscillatoryKernelAmplitudeProbability +
			FieldGeneConstants::mutateFieldGeneOscillatoryKernelDecayProbability +
			FieldGeneConstants::mutateFieldGeneOscillatoryKernelZeroCrossingsProbability)
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

	void FieldGene::mutateKernelType()
	{
		using namespace dnf_composer::element;
		using namespace neat_dnfs::tools::utils;

		const auto dimensions = neuralField->getElementCommonParameters().dimensionParameters;

		constexpr double totalProbability = FieldGeneConstants::gaussKernelProbability +
			FieldGeneConstants::mexicanHatKernelProbability +
			FieldGeneConstants::oscillatoryKernelProbability;

		constexpr double epsilon = 1e-6;
		if (std::abs(totalProbability - 1.0) > epsilon)
			throw std::runtime_error("Mutation probabilities in field gene kernel type mutation must sum up to 1.");

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

	void FieldGene::mutateNeuralField()
	{
		static constexpr double totalProbability = FieldGeneConstants::mutateFieldGeneNeuralFieldTauProbability +
			FieldGeneConstants::mutateFieldGeneNeuralFieldRestingLevelProbability + 
			FieldGeneConstants::mutateFieldGeneNeuralFieldRandomlyProbability;

		constexpr double epsilon = 1e-6;
		if (std::abs(totalProbability - 1.0) > epsilon)
			throw std::runtime_error("Mutation probabilities in field gene neural field mutation must sum up to 1.");

		const double signal = tools::utils::generateRandomSignal();
		dnf_composer::element::NeuralFieldParameters nfp = neuralField->getParameters();

		const double mutationSelection = tools::utils::generateRandomDouble(0.0, 1.0);
		if (mutationSelection < FieldGeneConstants::mutateFieldGeneNeuralFieldTauProbability)
		{
			const double tau = neuralField->getParameters().tau;
			nfp.tau = std::clamp(tau + NeuralFieldConstants::tauStep * signal,
												NeuralFieldConstants::tauMinVal,
												NeuralFieldConstants::tauMaxVal);
			neuralField->setParameters(nfp);
		}
		else if (mutationSelection < FieldGeneConstants::mutateFieldGeneNeuralFieldTauProbability +
			FieldGeneConstants::mutateFieldGeneNeuralFieldRestingLevelProbability)
		{
			const double restingLevel = neuralField->getParameters().startingRestingLevel;
			nfp.startingRestingLevel = std::clamp(restingLevel + NeuralFieldConstants::restingLevelStep * signal,
												NeuralFieldConstants::restingLevelMinVal,
												NeuralFieldConstants::restingLevelMaxVal);
			neuralField->setParameters(nfp);
		}
		else
		{
			dnf_composer::element::ElementCommonParameters nfcp = neuralField->getElementCommonParameters();
			initializeNeuralField(nfcp.dimensionParameters);
		}
		statistics.numNeuralFieldMutationsPerGeneration++;
		statistics.numNeuralFieldMutationsTotal++;
	}
}