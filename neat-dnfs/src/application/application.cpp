#include "application/application.h"


namespace neat_dnfs
{
	Application::Application(const std::shared_ptr<Population>& population)
		: population(population), guiActive(true)
	{
		using namespace imgui_kit;
		const WindowParameters winParams{ "neat dnfs" };
		const FontParameters fontParams({ {std::string(PROJECT_DIR) + "/resources/fonts/JetBrainsMono-Regular.ttf", 16},
										{std::string(PROJECT_DIR) + "/resources/fonts/JetBrainsMono-Thin.ttf", 16},
										{std::string(PROJECT_DIR) + "/resources/fonts/JetBrainsMono-Medium.ttf", 16},
										{std::string(PROJECT_DIR) + "/resources/fonts/JetBrainsMono-Bold.ttf", 18},
										{std::string(PROJECT_DIR) + "/resources/fonts/JetBrainsMono-Italic.ttf", 16},
										{std::string(PROJECT_DIR) + "/resources/fonts/JetBrainsMono-Light.ttf", 16},
			}
		);
		const StyleParameters styleParams{ Theme::ClassicSteam };
		const IconParameters iconParams{ std::string(PROJECT_DIR) + "/resources/icons/icon.ico" };
		const BackgroundImageParameters bgParams{ std::string(PROJECT_DIR) + "/resources/images/background.png", ImageFitType::ZOOM_TO_FIT};
		const UserInterfaceParameters guiParameters{ winParams, fontParams, styleParams, iconParams, bgParams };

		gui = std::make_shared<UserInterface>(guiParameters);
	}

	void Application::init() const
	{
		gui->addWindow<SetupWindow>(population );
		gui->addWindow<imgui_kit::LogWindow>();

		if(population)
			population->initialize();

		gui->initialize();
		log(tools::logger::LogLevel::INFO, "Application initialized successfully.");
	}

	void Application::step() const
	{
		if(population->isInitialized() && !population->endConditionMet())
			population->evolutionaryStep();
		if (guiActive)
			gui->render();
	}

	void Application::close() const
	{
		if (guiActive)
			gui->shutdown();
	}

	bool Application::hasGUIBeenClosed() const
	{
		if (guiActive)
			return gui->isShutdownRequested();
		return false;
	}

}
