//#pragma once
//
//#include <thread>
//#include <imgui-platform-kit/user_interface_window.h>
//#include <user_interface/simulation_window.h>
//#include <user_interface/plot_window.h>
//#include <user_interface/element_window.h>
//#include <user_interface/field_metrics_window.h>
//
//#include "neat/population.h"
//#include "solutions/single_bump.h"
//
//namespace neat_dnfs
//{
//	class PopulationControlWindow : public imgui_kit::UserInterfaceWindow
//	{
//	private:
//		std::shared_ptr<std::thread> evolveThread;
//		std::shared_ptr<std::thread> simulationThread;

//		std::shared_ptr<Population> population;
//		PopulationParameters populationParameters;
//		SolutionTopology solutionParameters;
//		SolutionPtr bestSolution;
//	public:
//		PopulationControlWindow(const std::shared_ptr<Population>& population);
//		void render() override;
//		~PopulationControlWindow() override;
//		std::shared_ptr<Population> getPopulation() const {return population;}
//	private:
//		void renderPopulationInfo();
//		void renderSolutionInfo();
//		void renderPopulationMethods();
//		void renderShowBestSolution();
//
//		void runSimulation(const std::shared_ptr<dnf_composer::Simulation>& simulation);
//	};
//}
