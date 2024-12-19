#pragma once

#include <thread>
#include <windows.h>//Sleep
#include <cstdio>

#include <imgui-platform-kit/user_interface_window.h>

#include "neat/population.h"

namespace neat_dnfs
{
	class SetupWindow : public imgui_kit::UserInterfaceWindow
	{
	private:
		std::shared_ptr<Population> population;
	public:
		SetupWindow(const std::shared_ptr<Population>& population);
		void render() override;
		~SetupWindow() = default;
	private:
		void renderPopulationInfo();
		void renderSolutionInfo();
		void renderPopulationMethods();
		void renderSolutionMethods();
		void renderShowBestSolution();
	};
}
