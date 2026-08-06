#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <sstream>

#include "neat/population.h"
#include "neat_tools/key_listener.h"
#include "test_helpers.h"
#include "test_stub_solution.h"

using namespace neat_dnfs;
using namespace neat_dnfs::test;

namespace
{
	// KeyListener's constructor prints a banner; keep test output clean the
	// same way test_logger.cpp does, by swapping std::cout's buffer for the
	// duration of the call.
	template <typename Fn>
	void withSuppressedCout(Fn&& fn)
	{
		std::ostringstream discarded;
		std::streambuf* const originalCoutBuffer = std::cout.rdbuf(discarded.rdbuf());
		try
		{
			fn();
		}
		catch (...)
		{
			std::cout.rdbuf(originalCoutBuffer);
			throw;
		}
		std::cout.rdbuf(originalCoutBuffer);
	}
}

// The blocking-stdin path itself (the original bug: a detached worker
// dereferencing a freed KeyListener/Population from inside std::cin.get())
// can't be driven from a unit test without hanging it on real stdin. What is
// testable, and what actually exercises the fix:
//   1. the key -> action mapping (handleKey), pulled out of the polling loop
//      specifically so it doesn't require stdin at all;
//   2. that constructing then destroying a KeyListener completes promptly
//      with no input ever arriving -- proving the destructor's join() really
//      returns instead of detaching a thread that outlives the object.

TEST_CASE("KeyListener::handleKey('s') stops the Population and reports so", "[KeyListener]")
{
	resetGlobalState();

	const PopulationParameters parameters(4, 1000, 0.99);
	Population population{ parameters, std::make_shared<FixedFitnessSolution>(makeTopology(1, 1), 0.1), false };
	population.initialize();

	withSuppressedCout([&]()
		{
			KeyListener keyListener{ population };
			REQUIRE(keyListener.handleKey('s'));
		});

	// Population::evolve() is a do-while loop that only checks the stop flag
	// after running a generation, so a Population that was told to stop
	// before evolve() was even called still runs exactly one generation --
	// not the 1000 configured -- which is what proves handleKey('s') reached
	// Population::stop() rather than being a no-op.
	population.evolve();
	REQUIRE(population.getCurrentGeneration() == 1);
}

TEST_CASE("KeyListener::handleKey only 's' requests stop", "[KeyListener]")
{
	resetGlobalState();

	const PopulationParameters parameters(4, 5, 0.99);
	Population population{ parameters, std::make_shared<FixedFitnessSolution>(makeTopology(1, 1), 0.1), false };
	population.initialize();

	withSuppressedCout([&]()
		{
			KeyListener keyListener{ population };
			REQUIRE_FALSE(keyListener.handleKey('p'));
			REQUIRE_FALSE(keyListener.handleKey('r'));
			REQUIRE_FALSE(keyListener.handleKey('x'));
			REQUIRE(keyListener.handleKey('s'));
		});
}

TEST_CASE("KeyListener destructor joins the worker instead of detaching it", "[KeyListener]")
{
	resetGlobalState();

	const PopulationParameters parameters(4, 5, 0.99);
	Population population{ parameters, std::make_shared<FixedFitnessSolution>(makeTopology(1, 1), 0.1), false };
	population.initialize();

	const auto start = std::chrono::steady_clock::now();

	withSuppressedCout([&]()
		{
			KeyListener keyListener{ population };
			// No key is ever supplied. Under the original bug this scope exiting
			// would `detach()` a thread left blocked in std::cin.get(); here the
			// worker polls in bounded slices, so destruction below actually
			// completes (join() returns) instead of hanging the test process.
		});

	const auto elapsed = std::chrono::steady_clock::now() - start;
	// Polling runs in 100ms slices; a couple of seconds is a generous bound
	// that only a genuinely stuck (i.e. still-bugged) join would exceed.
	REQUIRE(elapsed < std::chrono::seconds(3));
}
