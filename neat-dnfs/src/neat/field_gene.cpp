#include "neat/field_gene.h"
#include <format> 

namespace neat_dnfs
{
	FieldGeneParameters::FieldGeneParameters(const FieldGeneType type, const int id)
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
		return std::format("id: {}, type: {}", id, typeStr);
	}

	void FieldGeneParameters::print() const
	{
		tools::logger::log(tools::logger::INFO, toString());
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
		const ElementCommonParameters nfcp{ std::format("{}{}", NeuralFieldConstants::namePrefix, parameters.id), dimensions };
		neuralField = std::make_shared<NeuralField>(nfcp, nf->getParameters());

		const auto k = other.getKernel();
		switch (k->getLabel())
		{
			case ElementLabel::GAUSS_KERNEL:
				{
					const auto gkp = std::dynamic_pointer_cast<GaussKernel>(k)->getParameters();
					const ElementCommonParameters gkcp{ std::format("{}{}", GaussKernelConstants::namePrefix, parameters.id), dimensions };
					kernel = std::make_shared<GaussKernel>(gkcp, gkp);
				}
				break;
			case ElementLabel::MEXICAN_HAT_KERNEL:
				{
					const auto mhkp = std::dynamic_pointer_cast<MexicanHatKernel>(k)->getParameters();
					const ElementCommonParameters mhcp{ std::format("{}{}", MexicanHatKernelConstants::namePrefix, parameters.id), dimensions };
					kernel = std::make_shared<MexicanHatKernel>(mhcp, mhkp);
				}
				break;
			default:				
				tools::logger::log(tools::logger::FATAL, "FieldGene::FieldGene() - Kernel type not recognized.");
				throw std::runtime_error("FieldGene::FieldGene() - Kernel type not recognized.");
		}

		const std::shared_ptr<NormalNoise> nn = other.getNoise();
		const ElementCommonParameters nncp{ std::format("{}{}", NoiseConstants::namePrefix, parameters.id), dimensions };
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
		static constexpr double totalProbability = FieldGeneConstants::mutateFieldGeneKernelProbability +
			FieldGeneConstants::mutateFieldGeneNeuralFieldProbability +
			FieldGeneConstants::mutateFieldGeneKernelTypeProbability;

		constexpr double epsilon = 1e-6;
		if (std::abs(totalProbability - 1.0) > epsilon)
		{
			throw std::runtime_error("Mutation probabilities in field gene mutation must sum up to 1.");
		}

		const double randomValue = tools::utils::generateRandomDouble(0.0, 1.0);
		if (randomValue < FieldGeneConstants::mutateFieldGeneKernelProbability)
		{
			mutateKernel();
		}
		else if (randomValue < FieldGeneConstants::mutateFieldGeneKernelProbability +
			FieldGeneConstants::mutateFieldGeneNeuralFieldProbability)
		{
			mutateNeuralField();
		}
		else
		{
			mutateKernelType();
		}
	}

	void FieldGene::clearLastMutations()
	{
		mutationsInLastGeneration = "";
	}

	FieldGeneParameters FieldGene::getParameters() const
	{
		return parameters;
	}

	std::string FieldGene::getMutationsInLastGeneration() const
	{
		return mutationsInLastGeneration;
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
			const ElementCommonParameters nfcp{ std::format("{}{}", NeuralFieldConstants::namePrefix, parameters.id), dimensions };
			neuralField = std::make_shared<NeuralField>(nfcp, nfp);
		}
		else
		{
			constexpr double tau = NeuralFieldConstants::tau;
			constexpr double restingLevel = NeuralFieldConstants::restingLevel;
			const NeuralFieldParameters nfp{ tau, restingLevel, NeuralFieldConstants::activationFunction };
			const ElementCommonParameters nfcp{ std::format("{}{}", NeuralFieldConstants::namePrefix, parameters.id), dimensions };
			neuralField = std::make_shared<NeuralField>(nfcp, nfp);
		}
	}

	void FieldGene::initializeKernel(const dnf_composer::element::ElementDimensions& dimensions)
	{
		using namespace neat_dnfs::tools::utils;

		static constexpr double totalProbability = FieldGeneConstants::gaussKernelProbability +
			FieldGeneConstants::mexicanHatKernelProbability;

		constexpr double epsilon = 1e-6;
		if (std::abs(totalProbability - 1.0) > epsilon)
		{
			throw std::runtime_error("Kernel probabilities in field gene initialization must sum up to 1.");
		}

		const double randomValue = generateRandomDouble(0.0, 1.0);
		if (randomValue < FieldGeneConstants::gaussKernelProbability)
		{
			initializeGaussKernel(dimensions);
		}
		else
		{
			initializeMexicanHatKernel(dimensions);
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
			const ElementCommonParameters gkcp{ std::format("{}{}", GaussKernelConstants::namePrefix, parameters.id), dimensions };
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
			const ElementCommonParameters gkcp{ std::format("{}{}", GaussKernelConstants::namePrefix, parameters.id), dimensions };
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
		const ElementCommonParameters mhcp{ std::format("{}{}", MexicanHatKernelConstants::namePrefix, parameters.id), dimensions };
		kernel = std::make_shared<MexicanHatKernel>(mhcp, mhkp);
	}

	void FieldGene::initializeNoise(const dnf_composer::element::ElementDimensions& dimensions)
	{
		using namespace dnf_composer::element;
		using namespace neat_dnfs::tools::utils;

		const NormalNoiseParameters nnp{ NoiseConstants::amplitude };
		const ElementCommonParameters nncp{ std::format("{}{}", NoiseConstants::namePrefix, parameters.id), dimensions };
		noise = std::make_shared<NormalNoise>(nncp, nnp);
	}

	void FieldGene::mutateKernel()
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
			tools::logger::log(tools::logger::FATAL, "FieldGene::mutate() - Kernel type not recognized.");
			throw std::runtime_error("FieldGene::mutate() - Kernel type not recognized.");
		}
	}

	void FieldGene::mutateGaussKernel()
	{
		using namespace dnf_composer::element;
		using namespace neat_dnfs::tools::utils;

		const int signal = generateRandomSignal(); // +/- to add or sum a step

		const auto gaussKernel = std::dynamic_pointer_cast<GaussKernel>(kernel);
		GaussKernelParameters gkp = std::dynamic_pointer_cast<GaussKernel>(kernel)->getParameters();

		if (generateRandomDouble(0.0, 1.0) < FieldGeneConstants::mutateFieldGeneGaussKernelWidthProbability)
		{
			gkp.width = std::clamp(gkp.width + GaussKernelConstants::widthStep * signal,
								GaussKernelConstants::widthMinVal,
								GaussKernelConstants::widthMaxVal);
			mutationsInLastGeneration += std::format("(fg gk width {})", GaussKernelConstants::widthStep * signal);
		}

		if (generateRandomDouble(0.0, 1.0) < FieldGeneConstants::mutateFieldGeneGaussKernelAmplitudeProbability)
		{
			gkp.amplitude = std::clamp(gkp.amplitude + GaussKernelConstants::ampStep * signal,
								GaussKernelConstants::ampMinVal,
								GaussKernelConstants::ampMaxVal);
			mutationsInLastGeneration += std::format("(fg gk amp. {})", GaussKernelConstants::ampStep * signal);
		}

		if (generateRandomDouble(0.0, 1.0) < FieldGeneConstants::mutateFieldGeneGaussKernelGlobalAmplitudeProbability)
		{
			gkp.amplitudeGlobal = std::clamp(gkp.amplitudeGlobal + GaussKernelConstants::ampGlobalStep * signal,
								GaussKernelConstants::ampGlobalMinVal,
								GaussKernelConstants::ampGlobalMaxVal);
			mutationsInLastGeneration += std::format("(fg gk amp. glob. {})", GaussKernelConstants::ampGlobalStep * signal);
		}
		std::dynamic_pointer_cast<GaussKernel>(kernel)->setParameters(gkp);

	}

	void FieldGene::mutateMexicanHatKernel()
	{
		using namespace dnf_composer::element;
		using namespace neat_dnfs::tools::utils;

		const int signal = generateRandomSignal();  // +/- to add or sum a step

		const auto mexicanHatKernel = std::dynamic_pointer_cast<MexicanHatKernel>(kernel);
		MexicanHatKernelParameters mhkp = std::dynamic_pointer_cast<MexicanHatKernel>(kernel)->getParameters();

		if(generateRandomDouble(0.0, 1.0) < FieldGeneConstants::mutateFieldGeneMexicanHatKernelAmplitudeExcProbability)
		{
			mhkp.amplitudeExc = std::clamp(mhkp.amplitudeExc + MexicanHatKernelConstants::ampExcStep * signal,
								MexicanHatKernelConstants::ampExcMinVal,
								MexicanHatKernelConstants::ampExcMaxVal);
			mutationsInLastGeneration += std::format("(fg mhk amp. exc. {})", MexicanHatKernelConstants::ampExcStep * signal);
		}

		if (generateRandomDouble(0.0, 1.0) < FieldGeneConstants::mutateFieldGeneMexicanHatKernelWidthExcProbability)
		{
			mhkp.widthExc = std::clamp(mhkp.widthExc + MexicanHatKernelConstants::widthExcStep * signal,
												MexicanHatKernelConstants::widthExcMinVal,
												MexicanHatKernelConstants::widthExcMaxVal);
			mutationsInLastGeneration += std::format("(fg mhk width exc. {})", MexicanHatKernelConstants::widthExcStep * signal);
		}

		if (generateRandomDouble(0.0, 1.0) < FieldGeneConstants::mutateFieldGeneMexicanHatKernelAmplitudeInhProbability)
		{
			mhkp.amplitudeInh = std::clamp(mhkp.amplitudeInh + MexicanHatKernelConstants::ampInhStep * signal,
																MexicanHatKernelConstants::ampInhMinVal,
																MexicanHatKernelConstants::ampInhMaxVal);
			mutationsInLastGeneration += std::format("(fg mhk amp. inh. {})", MexicanHatKernelConstants::ampInhStep * signal);
		}

		if (generateRandomDouble(0.0, 1.0) < FieldGeneConstants::mutateFieldGeneMexicanHatKernelWidthInhProbability)
		{
			mhkp.widthInh = std::clamp(mhkp.widthInh + MexicanHatKernelConstants::widthInhStep * signal,
															MexicanHatKernelConstants::widthInhMinVal,
															MexicanHatKernelConstants::widthInhMaxVal);
			mutationsInLastGeneration += std::format("(fg mhk width inh. {})", MexicanHatKernelConstants::widthInhStep * signal); 
		}

		if (generateRandomDouble(0.0, 1.0) < FieldGeneConstants::mutateFieldGeneMexicanHatKernelGlobalAmplitudeProbability)
		{
			mhkp.amplitudeGlobal = std::clamp(mhkp.amplitudeGlobal + MexicanHatKernelConstants::ampGlobStep * signal,
															MexicanHatKernelConstants::ampGlobMin,
															MexicanHatKernelConstants::ampGlobMax);
			mutationsInLastGeneration += std::format("(fg mhk amp. glob. {})", MexicanHatKernelConstants::ampGlobStep * signal);
		}
		std::dynamic_pointer_cast<MexicanHatKernel>(kernel)->setParameters(mhkp);
	}

	void FieldGene::mutateKernelType()
	{
		using namespace dnf_composer::element;
		using namespace neat_dnfs::tools::utils;

		const auto dimensions = neuralField->getElementCommonParameters().dimensionParameters;

		constexpr double totalProbability = FieldGeneConstants::gaussKernelProbability +
			FieldGeneConstants::mexicanHatKernelProbability;

		constexpr double epsilon = 1e-6;
		if (std::abs(totalProbability - 1.0) > epsilon)
		{
			throw std::runtime_error("Mutation probabilities in field gene kernel type mutation must sum up to 1.");
		}

		const double randomValue = generateRandomDouble(0.0, 1.0);

		if (randomValue < FieldGeneConstants::gaussKernelProbability)
		{
			initializeGaussKernel(dimensions);
			mutationsInLastGeneration += "(mhk to gk)";
		}
		else if (randomValue < FieldGeneConstants::gaussKernelProbability + FieldGeneConstants::mexicanHatKernelProbability)
		{
			initializeMexicanHatKernel(dimensions);
			mutationsInLastGeneration += "(gk to mhk)";
		}
	}

	void FieldGene::mutateNeuralField()
	{
		static constexpr double totalProbability = FieldGeneConstants::mutateFieldGeneNeuralFieldParametersProbability +
			FieldGeneConstants::mutateFieldGeneNeuralFieldGenerateRandomParametersProbability;

		constexpr double epsilon = 1e-6;
		if (std::abs(totalProbability - 1.0) > epsilon)
		{
			throw std::runtime_error("Mutation probabilities in field gene neural field mutation must sum up to 1.");
		}

		const double signal = tools::utils::generateRandomSignal();
		dnf_composer::element::NeuralFieldParameters nfp = neuralField->getParameters();

		const double mutationSelection = tools::utils::generateRandomDouble(0.0, 1.0);
		if (mutationSelection < FieldGeneConstants::mutateFieldGeneNeuralFieldParametersProbability)
		{
			if (tools::utils::generateRandomDouble(0.0, 1.0) <
				FieldGeneConstants::mutateFieldGeneNeuralFieldParametersTauProbability)
			{
				const double tau = neuralField->getParameters().tau;
				nfp.tau = std::clamp(tau + NeuralFieldConstants::tauStep * signal,
													NeuralFieldConstants::tauMinVal,
													NeuralFieldConstants::tauMaxVal);
				neuralField->setParameters(nfp);
				mutationsInLastGeneration += std::format("(fg nf tau {})", NeuralFieldConstants::tauStep * signal);
			}

			if (tools::utils::generateRandomDouble(0.0, 1.0) <
				FieldGeneConstants::mutateFieldGeneNeuralFieldParametersRestingLevelProbability)
			{
				const double restingLevel = neuralField->getParameters().startingRestingLevel;
				nfp.startingRestingLevel = std::clamp(restingLevel + NeuralFieldConstants::restingLevelStep * signal,
													NeuralFieldConstants::restingLevelMinVal,
													NeuralFieldConstants::restingLevelMaxVal);
				neuralField->setParameters(nfp);
				mutationsInLastGeneration += std::format("(fg nf rest. lvl. {})", NeuralFieldConstants::restingLevelStep * signal);
			}
		}
		else
		{
			dnf_composer::element::ElementCommonParameters nfcp = neuralField->getElementCommonParameters();
			initializeNeuralField(nfcp.dimensionParameters);
			mutationsInLastGeneration += "(fg nf rand.)";
		}
	}
}