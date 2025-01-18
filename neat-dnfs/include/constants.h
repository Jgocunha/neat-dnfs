#pragma once

#include <elements/element_factory.h>

namespace neat_dnfs
{
	typedef std::shared_ptr<dnf_composer::element::NeuralField> NeuralFieldPtr;
	typedef std::shared_ptr<dnf_composer::element::Kernel> KernelPtr;
	typedef std::shared_ptr<dnf_composer::element::FieldCoupling> FieldCouplingPtr;
	typedef std::shared_ptr<dnf_composer::element::NormalNoise> NormalNoisePtr;

	struct SimulationConstants
	{
		inline static std::string name = "solution ";
		static constexpr double deltaT = 10.0;
		static constexpr size_t maxSimulationSteps = 300;
	};

	struct DimensionConstants
	{
		static constexpr int xSize = 100;
		static constexpr double dx = 1.0;
	};

	struct NoiseConstants
	{
		inline static std::string namePrefix = "nn ";
		static constexpr double amplitude = 0.2;
	};

	struct GaussStimulusConstants
	{
		inline static std::string namePrefix = "gs ";
		static constexpr double width = 5.0;
		static constexpr double amplitude = 15.0;
		static constexpr bool circularity = true;
		static constexpr bool normalization = false;
	};

	struct NeuralFieldConstants
	{
		inline static std::string namePrefix = "nf ";
		static constexpr double stabilityThreshold = 0.90;
		static constexpr double tau = 25;
		static constexpr double restingLevel = -5;
		inline static dnf_composer::element::HeavisideFunction activationFunction{ 0.0 };
		static constexpr double tauMinVal = 15.0;
		static constexpr double tauMaxVal = 300.0;
		static constexpr double tauStep = 5.0;
		static constexpr double restingLevelMinVal = -20.0;
		static constexpr double restingLevelMaxVal = -1.0;
		static constexpr double restingLevelStep = 0.5;
	};

	struct KernelConstants
	{
		static constexpr bool circularity = true;
		static constexpr bool normalization = true;
	};

	struct GaussKernelConstants
	{
		inline static std::string namePrefix = "gk ";
		inline static std::string namePrefixConnectionGene = "gk cg ";

		static constexpr double width = 5;
		static constexpr double amplitude = 2;
		static constexpr double amplitudeGlobal = -0.01;

		static constexpr double widthMinVal = 3.0; // 3
		static constexpr double widthMaxVal = 10.0; // 5.0
		static constexpr double widthStep = 0.5; // 0.5
		static constexpr double ampMinVal = 3.0; // 0.5
		static constexpr double ampMaxVal = 25.0; // 8.0
		static constexpr double ampStep = 0.5; // 0.5
		static constexpr double ampGlobalMinVal = -5.0; // -0.5
		static constexpr double ampGlobalMaxVal = 0.0; // -0.01
		static constexpr double ampGlobalStep = 0.05; // 0.01	 
	};

	struct MexicanHatKernelConstants
	{
		inline static std::string namePrefix = "mhk ";
		inline static std::string namePrefixConnectionGene = "mhk cg ";

		static constexpr double widthExc = 2.5; // 2.5
		static constexpr double widthInh = 5.0; // 5.0
		static constexpr double amplitudeExc = 11.0;  // 11.0
		static constexpr double amplitudeInh = 15.0;  // 15.0
		static constexpr double amplitudeGlobal = -0.01; // -0.1

		static constexpr double widthExcMinVal = 1.0; // 2.0
		static constexpr double widthExcMaxVal = 10.0; // 9.0
		static constexpr double widthExcStep = 0.5; // 0.5

		static constexpr double widthInhMinVal = 5.0; // 5.0
		static constexpr double widthInhMaxVal = 30.0; // 30.0
		static constexpr double widthInhStep = 0.5; // 0.5

		static constexpr double ampExcMinVal = 1.0; // 8.0
		static constexpr double ampExcMaxVal = 20.0; // 25.0
		static constexpr double ampExcStep = 0.5; // 0.5

		static constexpr double ampInhMinVal = 15.0; // 12.0
		static constexpr double ampInhMaxVal = 25.0; // 30.0
		static constexpr double ampInhStep = 0.5; // 0.5

		static constexpr double ampGlobMin = -5.0; // -0.5
		static constexpr double ampGlobMax = 0.0; // -0.01
		static constexpr double ampGlobStep = 0.05; // 0.01
	};

	struct OscillatoryKernelConstants
	{
		inline static std::string namePrefix = "osck ";
		inline static std::string namePrefixConnectionGene = "osck cg ";

		static constexpr double amplitude = 1.0;
		static constexpr double decay = 0.08;
		static constexpr double zeroCrossings = 0.3;
		static constexpr double amplitudeGlobal = -0.01;

		static constexpr double amplitudeMinVal = 1;
		static constexpr double amplitudeMaxVal = 25.0;
		static constexpr double amplitudeStep = 0.5;

		static constexpr double decayMinVal = 0.01;
		static constexpr double decayMaxVal = 1.0;
		static constexpr double decayStep = 0.05;

		static constexpr double zeroCrossingsMinVal = 0.1;
		static constexpr double zeroCrossingsMaxVal = 1.0;
		static constexpr double zeroCrossingsStep = 0.5;

		static constexpr double ampGlobMin = -5.0;
		static constexpr double ampGlobMax = 0.0;
		static constexpr double ampGlobStep = 0.05;
	};

	struct MutationConstants
	{
		static constexpr double addFieldGeneProbability = 0.00;
		static constexpr double mutateFieldGeneProbability = 0.45;
		static constexpr double addConnectionGeneProbability = 0.15;
		static constexpr double mutateConnectionGeneProbability = 0.40;
		static constexpr double toggleConnectionGeneProbability = 0.00;
	};

	struct CompatibilityCoefficients
	{
		static constexpr double excessCoefficient = 1.0;
		static constexpr double disjointCoefficient = 1.0;
		static constexpr double amplitudeDifferenceCoefficient = 0.8;
		static constexpr double widthDifferenceCoefficient = 0.2;
	};

	struct ConnectionGeneConstants
	{
		static constexpr int allowAllKernelTypes = false;
		// set to true to allow all kernel types, if false only gauss kernel is allowed
		static constexpr int allowInhibitoryConnections = false;
		// set to true to allow inhibitory connections, if false only excitatory connections are allowed
	};

	struct SolutionConstants
	{
		static constexpr uint8_t minInitialInputGenes = 1;
		static constexpr uint8_t minInitialOutputGenes = 1;
		static constexpr double initialConnectionProbability = 1.0;
	};

	struct SpeciesConstants
	{
		static constexpr double compatibilityThreshold = 5.0;
		static constexpr double excessGenesCompatibilityWeight = 0.5;
		static constexpr double disjointGenesCompatibilityWeight = 0.5;
		static constexpr double averageConnectionDifferenceCompatibilityWeight = 0.5;
	};

	struct PopulationConstants
	{
		static constexpr double killRatio = 0.8;
		static constexpr bool validateUniqueSolutions = true;
		static constexpr bool validatePopulationSize = true;
		static constexpr bool validateElitism = true;
		static constexpr bool validateUniqueGenesInGenomes = true;
		static constexpr bool validateUniqueKernelAndNeuralFieldPtrs = true;
		static constexpr bool logs = false;
	};

}