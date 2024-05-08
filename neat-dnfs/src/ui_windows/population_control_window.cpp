#include "ui_windows/population_control_window.h"

namespace neat_dnfs
{
	PopulationControlWindow::PopulationControlWindow(const std::shared_ptr<Population>& population)
		: population(population)
	{
	}

	void PopulationControlWindow::render()
	{
		if(ImGui::Begin("Population control"))
		{
			ImGui::Text("Population size");
		}
		ImGui::End();
	}


}