#pragma once

#include "neat/solution.h"
#include "tools/logger.h"

namespace neat_dnfs
{
	class TemplateSolution : public Solution
	{
		std::vector<std::map<std::string, std::vector<dnf_composer::element::NeuralFieldBump>>> outputFieldsBumps;

	public:
		TemplateSolution(const SolutionTopology& initialTopology);

		void evaluate() override;
		SolutionPtr clone() const override;
	private:
		void evaluatePhenotype(); 
		void addStimulus(const std::string& name, const double& position);
		void removeStimulus();
		void runSimulation();
		void recordSimulationResults();
		void updateFitness();
	};
}