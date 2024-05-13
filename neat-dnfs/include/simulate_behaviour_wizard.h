#pragma once

#include <elements/neural_field.h>
#include <simulation/simulation.h>

namespace neat_dnfs
{
	typedef dnf_composer::Simulation Phenotype;

	class SimulateBehaviourWizard
	{
	private:
		std::shared_ptr<Phenotype> phenotype;
		std::map<std::string, std::vector<std::vector<double>>> inputFieldsBumpPositions;
		std::vector<std::string> outputFieldsNames;
		std::vector<std::map<std::string, std::vector<dnf_composer::element::NeuralFieldBump>>> outputFieldsBumps;
	public:
		SimulateBehaviourWizard(const Phenotype& phenotype);
		void setOutputFieldsNames(const std::vector<std::string>& names);
		void addInputFieldBumps(const std::string& name, const std::vector<std::vector<double>>& bump);
		std::vector<std::map<std::string, std::vector<dnf_composer::element::NeuralFieldBump>>> getResultingBumps();
		void simulate();
	private:
		void addStimulus(const std::string& name, const double& position);
		void removeStimulus();
		void runSimulation();
		void recordOutputFieldBumps();
	};
}
