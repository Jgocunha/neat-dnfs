
// This is a personal academic project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: https://pvs-studio.com

#include <exception>
#include <iostream>

#include "neat/population.h"
#include "solutions/template_solution.h"
#include "application/application.h"
#include "ui_windows/population_control_window.h"

int main(int argc, char* argv[])
{
	try
	{
		using namespace neat_dnfs;

		const SolutionTopology initialTopology{3, 1};
		TemplateSolution solution(initialTopology);
		Population population(100, std::make_shared<TemplateSolution>(solution));
		constexpr int numGenerations = 1000;
/*
		population.initialize();
		population.evolve(numGenerations);
		SolutionPtr bestSolution = population.getBestSolution();*/

		Application app;
		app.addWindow<imgui_kit::LogWindow>();
		app.addWindow<PopulationControlWindow>(std::make_shared<Population>(population));
		app.initialize();
		do
		{
			app.render();
		} while (!app.hasCloseBeenRequested());

		return 0;
	}
	catch (const std::exception& ex)
	{
		std::cerr << ex.what() << std::endl;
		return 1;
	}
	catch (...)
	{
		std::cerr << "Unknown exception" << std::endl;
		return 1;
	}
}
