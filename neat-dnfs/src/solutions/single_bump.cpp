#include "solutions/single_bump.h"

namespace neat_dnfs
{
	SingleBumpSolution::SingleBumpSolution(const SolutionTopology& topology)
		: Solution(topology)
	{}

	SolutionPtr SingleBumpSolution::clone() const
	{
		SingleBumpSolution solution(initialTopology);
		auto clonedSolution = std::make_shared<SingleBumpSolution>(solution);

		return clonedSolution;
	}

	void SingleBumpSolution::testPhenotype()
	{
		const dnf_composer::element::GaussStimulusParameters stimParams = {
			5.0,
			15.0,
			50.0,
			false,
			false
		};
		addGaussianStimulus("nf 1", stimParams);
		initSimulation();

		const auto field_in = std::dynamic_pointer_cast<dnf_composer::element::NeuralField>(phenotype.getElement("nf 1"));
		const auto field_out = std::dynamic_pointer_cast<dnf_composer::element::NeuralField>(phenotype.getElement("nf 2"));
		double u_max_field_in = field_in->getHighestActivation();
		double u_max_field_out = field_out->getHighestActivation();
		if (u_max_field_in != 0.0)
			log(tools::logger::LogLevel::WARNING, "u_max at field in is not 0.0 after init() it is: " + std::to_string(u_max_field_in));
		if (u_max_field_out != 0.0)
			log(tools::logger::LogLevel::WARNING, "u_max at field out is not 0.0 after init() it is: " + std::to_string(u_max_field_out));

		//runSimulationUntilFieldStable("nf 1");
		//runSimulationUntilFieldStable("nf 2");
		runSimulation(100);

		u_max_field_in = field_in->getHighestActivation();
		u_max_field_out = field_out->getHighestActivation();
		if (u_max_field_in != 9.9859044930311960)
			log(tools::logger::LogLevel::WARNING, "u_max at field in is not 9.9859044930311960 after steps() it is: " + std::to_string(u_max_field_in));

		updateFitness();
		removeGaussianStimuli();

		//runSimulationUntilFieldStable("nf 1");
		//runSimulationUntilFieldStable("nf 2");
		runSimulation(10);

		u_max_field_in = field_in->getHighestActivation();
		u_max_field_out = field_out->getHighestActivation();

		if (u_max_field_in != -10.0)
			log(tools::logger::LogLevel::WARNING, "u_max at field in is not -10.0 after close() it is: " + std::to_string(u_max_field_in));
		if (u_max_field_out != -10.0)
			log(tools::logger::LogLevel::WARNING, "u_max at field out is not -10.0 after close() it is: " + std::to_string(u_max_field_out));
		//stopSimulation();

		// Check if fitness decreased.
		if (parameters.fitness == 0.0 && analysis.previousFitness > 0.0)
		{
			std::stringstream address;
			address << this;
			log(tools::logger::LogLevel::ERROR, "Fitness is: " + std::to_string(parameters.fitness) 
				+ " and changed from: " + std::to_string(analysis.previousFitness) + " at address: " + address.str());
		}
		// Check if is an elite
		if(analysis.previousFitness > 0.00)
		{
			if (parameters.fitness == 0.0)
				log(tools::logger::LogLevel::FATAL, "Fitness is 0.0 but previous fitness is greater than 0.0.");
			// log if previousFitness is greater than 0.0 than i am an elite
			std::stringstream address;
			address << this;
			log(tools::logger::LogLevel::INFO, "Address is: " + address.str() + " and previousFitness is: " 
				+ std::to_string(analysis.previousFitness) + " and fitness is: " + std::to_string(parameters.fitness));
			// Check if fitness decreased
			if (parameters.fitness < analysis.previousFitness)
			{
				log(tools::logger::LogLevel::ERROR, "Fitness decreased but previous solution should not have not changed "
										"because it is an elite.");
				// compare bumps
				log(tools::logger::LogLevel::WARNING, "Bump is : " + std::to_string(parameters.bumps.front().centroid) 
					+ " and previous bump is: " + std::to_string(analysis.previousBump.centroid));
				// compare the genome to see if it is the same.
				if (genome != analysis.previousGenome)
				{
					log(tools::logger::LogLevel::ERROR, "Genome is not the same as previous genome.");
				}
				const auto element = phenotype.getElement("nf 2")->getInputs()[1];
				const auto kernel = std::dynamic_pointer_cast<dnf_composer::element::GaussKernel>(element);
				if (analysis.pgkps != kernel->getParameters())
				{
  					log(tools::logger::LogLevel::ERROR, "Gauss kernel parameters are not the same as previous gauss kernel parameters.");
				}
			}
		}

		analysis.previousFitness = parameters.fitness;
		analysis.previousGenome = genome;
		if (!parameters.bumps.empty())
		{
			analysis.previousBump = parameters.bumps.front();
			const auto inputs = phenotype.getElement("nf 2")->getInputs();
			for(const auto& element : inputs)
			{
				if (element->getLabel() == dnf_composer::element::ElementLabel::GAUSS_KERNEL)
				{
					const auto kernel = std::dynamic_pointer_cast<dnf_composer::element::GaussKernel>(element);
					analysis.pgkps = kernel->getParameters();
					analysis.pgkps.width = kernel->getParameters().width;
					analysis.pgkps.amplitude = kernel->getParameters().amplitude;
				}
			}

			const auto kernel_ = std::dynamic_pointer_cast<dnf_composer::element::GaussKernel>(genome.getConnectionGenes()[0].getKernel());
			
			if (kernel_->getParameters().amplitude != analysis.pgkps.amplitude || kernel_->getParameters().width != analysis.pgkps.width)
			{
				log(tools::logger::LogLevel::ERROR, "Kernel is not the same as the kernel in the genome.");
			}
		}
	}

	void SingleBumpSolution::updateFitness()
	{
		using namespace dnf_composer::element;

		static constexpr double expectedBumpPosition = 50.0;
		static constexpr double expectedBumpWidth = 10.0;
		static constexpr double expectedBumpAmplitude = 10.0;

		const auto field =
			std::dynamic_pointer_cast<NeuralField>(phenotype.getElement("nf 2"));
		const auto fieldBumps = field->getBumps();

		parameters.bumps = fieldBumps;
		if (fieldBumps.empty())
		{
			parameters.fitness = 0.0;
			return;
		}

		const auto bump = fieldBumps.front();
		const double centroidDifference = std::abs(bump.centroid - expectedBumpPosition);
		const double widthDifference = std::abs(bump.width - expectedBumpWidth);
		const double amplitudeDifference = std::abs(bump.amplitude - expectedBumpAmplitude);

		const double positionFitness = 0.5 / (1.0 + centroidDifference);
		const double widthFitness = 0.25 / (1.0 + widthDifference);
		const double amplitudeFitness = 0.25 / (1.0 + amplitudeDifference);

		parameters.fitness = positionFitness + widthFitness + amplitudeFitness;

		{
		//	// For debugging purposes.
		//if (parameters.fitness > 0.95)
		//{
		//	log(tools::logger::LogLevel::INFO, "Fitness is: " + std::to_string(parameters.fitness) + " centroidDifference: " + std::to_string(centroidDifference) + " widthDifference: " + std::to_string(widthDifference) + " amplitudeDifference: " + std::to_string(amplitudeDifference));
		//	log(tools::logger::LogLevel::INFO, "Position contribution: " + std::to_string(positionFitness) + " Width contribution: " + std::to_string(widthFitness) + " Amplitude contribution: " + std::to_string(amplitudeFitness));
		//	log(tools::logger::LogLevel::INFO, "Centroid: " + std::to_string(bump.centroid) + " width: " + std::to_string(bump.width) + " amplitude: " + std::to_string(bump.amplitude));
		//}

		// This evaluation is faulty (I think because of the precarious values of amplitude and width).
		// Fitness decreases but solution is present in the population.
		// bsAddress: 0000021F3D3F1A20 bsFitness: 0.5
		// opbsAddress: 0000021F3D3F1A21 opbsFitness: 0.66
		// npbsAddress: 0000021F3D3F1A21 npbsFitness: 0.43
		// pbsf > bsf > npbsf
		// pbsAddr = npbsAddr != bsAddr
		// Or:
		// Fitness decreased but previous best solution is in the population.
		// bsAddress: 0000021F3D3F1A20 bsFitness: 0.1667
		// opbsAddress: 0000021F3D3F1A20 opbsFitness: 0.1667
		// npbsAddress: 0000021F3D3F1A20 npbsFitness: 0.333
		// pbsf > npbsf = bsf
		// pbsAddr = npbsAddr = bsAddr
		// And:
		// Fitness decreased and previous best solution is not in the population.
		// Should just probably mean the previous best solution was so poorly evaluated
		// that it was not included as an elite in the population.
		// Meaning, the same solutions are being evaluated differently in evolutionary runs.
		// Part of the problem was the generation of non-stable solutions due to the lik parameters.
		// This still occurs tho.
		}
		
	}
}
