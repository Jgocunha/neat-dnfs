#pragma once

#include <atomic>
#include <iostream>
#include <thread>

#include "neat/population.h"
#include "neat_tools/logger.h"

namespace neat_dnfs
{
	/// @brief Console key listener that drives a Population's pause/resume/stop API.
	///
	/// Owns a worker thread that reads stdin and calls @c stop() / @c pause() / @c resume()
	/// on the given Population. The thread is joined on destruction, so the listener must
	/// not outlive the Population it references. Construct it in the same scope as evolve().
	class KeyListener
	{
	public:
		explicit KeyListener(Population& population)
			: population(population)
		{
			std::cout << R"(
        _             _            _                 _                    _            _             _         _
        /\ \     _    /\ \         / /\              /\ \                 /\ \         /\ \     _    /\ \      / /\
       /  \ \   /\_\ /  \ \       / /  \             \_\ \               /  \ \____   /  \ \   /\_\ /  \ \    / /  \
      / /\ \ \_/ / // /\ \ \     / / /\ \            /\__ \             / /\ \_____\ / /\ \ \_/ / // /\ \ \  / / /\ \__
     / / /\ \___/ // / /\ \_\   / / /\ \ \          / /_ \ \   ____    / / /\/___  // / /\ \___/ // / /\ \_\/ / /\ \___\
    / / /  \/____// /_/_ \/_/  / / /  \ \ \        / / /\ \ \/\____/\ / / /   / / // / /  \/____// /_/_ \/_/\ \ \ \/___/
   / / /    / / // /____/\    / / /___/ /\ \      / / /  \/_/\/____\// / /   / / // / /    / / // /____/\    \ \ \
  / / /    / / // /\____\/   / / /_____/ /\ \    / / /              / / /   / / // / /    / / // /\____\/_    \ \ \
 / / /    / / // / /______  / /_________/\ \ \  / / /               \ \ \__/ / // / /    / / // / /     /_/\__/ / /
/ / /    / / // / /_______\/ / /_       __\ \_\/_/ /                 \ \___\/ // / /    / / // / /      \ \/___/ /
\/_/     \/_/ \/__________/\_\___\     /____/_/\_\/                   \/_____/ \/_/     \/_/ \/_/        \_____\/

)"			<< '\n';
			std::cout << "Press 's' and 'Enter' to stop the current run." << '\n';
			std::cout << "Press 'p' and 'Enter' to pause the current run." << '\n';
			std::cout << "Press 'r' and 'Enter' to resume the current run." << "\n\n" << std::flush;

			worker = std::thread([this]() {
				while (!done)
				{
					const int key = std::cin.get();
					if (key == 's')
					{
						this->population.stop();
						tools::logger::log(tools::logger::LogLevel::INFO,
							"Stopping evolution after the current run...");
						return;
					}
					if (key == 'p')
					{
						this->population.pause();
						tools::logger::log(tools::logger::LogLevel::INFO,
							"Pausing evolution...");
					}
					if (key == 'r')
					{
						this->population.resume();
						tools::logger::log(tools::logger::LogLevel::INFO,
							"Resuming evolution...");
					}
				}
				});
		}

		~KeyListener()
		{
			done = true;
			if (worker.joinable())
				worker.detach();
		}

		KeyListener(const KeyListener&) = delete;
		KeyListener& operator=(const KeyListener&) = delete;

	private:
		Population& population;
		std::thread worker;
		std::atomic<bool> done{ false };
	};
}
