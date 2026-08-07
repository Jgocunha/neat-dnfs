#pragma once

#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

#if defined(_WIN32)
#include <conio.h>
#else
#include <sys/select.h>
#include <unistd.h>
#endif

#include "neat/population.h"
#include "neat_tools/logger.h"

namespace neat_dnfs
{
	/// @brief Console key listener that drives a Population's pause/resume/stop API.
	///
	/// Owns a worker thread that watches stdin for 's' (stop), 'p' (pause) and
	/// 'r' (resume) and calls the matching method on the given Population.
	///
	/// The worker never calls a blocking read directly. Instead it polls for
	/// input in short (pollInterval) slices -- @c select() with a timeout on
	/// POSIX, @c _kbhit() on Windows -- re-checking the shutdown flag between
	/// slices. That bound is what makes the destructor's @c join() actually
	/// terminate: it sets the flag and waits for the worker to notice it and
	/// return, so the thread never outlives the KeyListener (or the
	/// Population&) it reads from. Construct it in the same scope as
	/// evolve(), and let it go out of scope before the Population does.
	///
	/// Platform note: on POSIX this preserves the original behaviour exactly
	/// (terminal canonical mode + echo, so a keystroke needs Enter to reach
	/// the process). On Windows, @c _kbhit()/@c _getch() read the console's
	/// raw input queue directly, bypassing line editing, so a bare keystroke
	/// (no Enter) is enough -- unavoidable, since Windows has no equivalent
	/// of POSIX's select()-respects-canonical-mode blocking behaviour that
	/// would let a bounded wait coexist with line-buffered input. An
	/// optional trailing Enter is simply ignored.
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

			worker = std::thread([this]() { run(); });
		}

		~KeyListener()
		{
			done = true;
			if (worker.joinable())
				worker.join();
		}

		KeyListener(const KeyListener&) = delete;
		KeyListener& operator=(const KeyListener&) = delete;

		/// @brief Applies the action bound to a single keystroke.
		///
		/// Pulled out of the polling loop so the key -> action mapping can be
		/// unit-tested without driving it through stdin.
		/// @param key Character code read from the input source.
		/// @return true if this key requests the listener to stop (key == 's');
		///         false otherwise (including for keys with no binding).
		bool handleKey(int key)
		{
			if (key == 's')
			{
				population.stop();
				tools::logger::log(tools::logger::LogLevel::INFO,
					"Stopping evolution after the current run...");
				return true;
			}
			if (key == 'p')
			{
				population.pause();
				tools::logger::log(tools::logger::LogLevel::INFO,
					"Pausing evolution...");
			}
			if (key == 'r')
			{
				population.resume();
				tools::logger::log(tools::logger::LogLevel::INFO,
					"Resuming evolution...");
			}
			return false;
		}

	private:
		void run()
		{
			while (!done)
			{
				const int key = pollKey();
				if (key < 0)
					continue; // nothing arrived within pollInterval; re-check done

				if (done) // destructor may have started while we were polling
					break;

#if defined(_WIN32)
				// _getch() bypasses the console's line editor, so nothing echoes
				// the keystroke on screen; do it ourselves. POSIX doesn't need
				// this -- the terminal's own canonical-mode echo already did it.
				std::cout << static_cast<char>(key) << std::endl;
#endif
				if (handleKey(key))
					break;
			}
		}

		// Waits up to pollInterval for a keystroke; returns it, or -1 if none
		// arrived in time. Bounding the wait -- rather than blocking forever
		// on a raw read -- is what lets run() re-check `done` and lets the
		// destructor's join() actually return.
#if defined(_WIN32)
		int pollKey()
		{
			const auto deadline = std::chrono::steady_clock::now() + pollInterval;
			while (std::chrono::steady_clock::now() < deadline)
			{
				if (_kbhit())
					return _getch();
				std::this_thread::sleep_for(std::chrono::milliseconds(5));
			}
			return -1;
		}
#else
		int pollKey()
		{
			if (eofReached)
			{
				// stdin is closed/exhausted (e.g. redirected from an empty
				// source); nothing will ever arrive, so idle instead of
				// spinning select()/read() at 100% CPU.
				std::this_thread::sleep_for(pollInterval);
				return -1;
			}

			fd_set readSet;
			FD_ZERO(&readSet);
			FD_SET(STDIN_FILENO, &readSet);

			const auto micros =
				std::chrono::duration_cast<std::chrono::microseconds>(pollInterval).count();
			timeval tv{};
			tv.tv_sec = static_cast<decltype(tv.tv_sec)>(micros / 1'000'000);
			tv.tv_usec = static_cast<decltype(tv.tv_usec)>(micros % 1'000'000);

			const int ready = ::select(STDIN_FILENO + 1, &readSet, nullptr, nullptr, &tv);
			if (ready <= 0)
				return -1; // timeout or interrupted; try again next loop

			unsigned char ch = 0;
			const ssize_t bytesRead = ::read(STDIN_FILENO, &ch, 1);
			if (bytesRead <= 0)
			{
				eofReached = true;
				return -1;
			}
			return ch;
		}
#endif

		Population& population;
		std::thread worker;
		std::atomic<bool> done{ false };
		static constexpr std::chrono::milliseconds pollInterval{ 100 };
#if !defined(_WIN32)
		bool eofReached = false;
#endif
	};
}
