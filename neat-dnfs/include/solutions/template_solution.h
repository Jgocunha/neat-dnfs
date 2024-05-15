#pragma once

#include "neat/solution.h"
#include "tools/logger.h"

namespace neat_dnfs
{
	class TemplateSolution : public Solution
	{
		std::vector<std::map<std::string, std::vector<dnf_composer::element::NeuralFieldBump>>> outputFieldsBumps;

	public:
		TemplateSolution(const SolutionTopology& topology);

		void evaluate() override;
		SolutionPtr clone() const override;
	private:
		void evaluatePhenotype() override; 
		void createRandomInitialConnectionGenes() override;


		void addStimulus(const std::string& name, const double& position);
		void runSimulation();
		void removeStimulus();
		/*void recordSimulationResults();*/
		void updateFitness(double expectedOutput);
	};
}