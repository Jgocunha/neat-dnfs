#pragma once

#include <imgui-platform-kit/user_interface_window.h>

#include "neat/population.h"

namespace neat_dnfs
{
	class PopulationControlWindow : public imgui_kit::UserInterfaceWindow
	{
	private:
		std::shared_ptr<Population> population;
	public:
		PopulationControlWindow(const std::shared_ptr<Population>& population);
		void render() override;
	};
}