#pragma once

#include <imgui-platform-kit/user_interface.h>

#include "neat/population.h"
#include "user_interface/setup_window.h"

namespace neat_dnfs
{
	class Application
	{
	private:
		std::shared_ptr<Population> population;
		std::shared_ptr<imgui_kit::UserInterface> gui;
		bool guiActive;
	public:
		Application(const std::shared_ptr<Population>& population = nullptr);

		void init() const;
		void step() const;
		void close() const;

		bool hasGUIBeenClosed() const;
	};
}
