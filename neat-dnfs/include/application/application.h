#pragma once

#include <imgui-platform-kit/user_interface.h>

#include "neat/population.h"
#include "ui_windows/population_control_window.h"

namespace neat_dnfs
{
	class Application
	{
	private:
		std::shared_ptr<Population> population;
		std::shared_ptr<imgui_kit::UserInterface> ui;
		imgui_kit::UserInterfaceParameters uiParameters;
		bool activateUI;
	public:
		Application(bool activateUI = true);

		void initialize() const;
		void render() const;
		void shutdown() const;

		bool hasCloseBeenRequested() const;
	};
}
