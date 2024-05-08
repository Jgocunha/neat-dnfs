
// This is a personal academic project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: https://pvs-studio.com

#include <exception>
#include <iostream>

#include "neat/population.h"
#include "solutions/template_solution.h"
#include "application/application.h"

int main(int argc, char* argv[])
{
	try
	{
		using namespace neat_dnfs;

		const Application app;
		app.initialize();
		do { app.render(); } while (!app.hasCloseBeenRequested());

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
