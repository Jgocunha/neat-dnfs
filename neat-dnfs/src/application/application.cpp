#include "application/application.h"

namespace neat_dnfs
{
	Application::Application(bool activateUI)
		: ui {nullptr}, activateUI{ activateUI }
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
		uiParameters = UserInterfaceParameters{ winParams, fontParams, styleParams, iconParams, bgParams };

		ui = std::make_shared<UserInterface>(uiParameters);
	}

	void Application::initialize() const
	{
		if (activateUI)
		{
			ui->addWindow<imgui_kit::LogWindow>();
			//ui->addWindow<MainWindow>();
			ui->initialize();
		}
	}

	void Application::render() const
	{
		if (activateUI)
			ui->render();
	}

	void Application::shutdown() const
	{
		if (activateUI)
			ui->shutdown();
	}

	bool Application::hasCloseBeenRequested() const
	{
		return ui->isShutdownRequested();
	}

}