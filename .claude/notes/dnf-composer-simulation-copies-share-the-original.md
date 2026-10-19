# A copied dnf-composer Simulation still points at the original's elements

`dnf_composer::Simulation`'s copy constructor clones every element, but each clone is a plain copy of its element, so
its inputs and outputs still point at the *original* simulation's elements. `~Simulation()` calls
`disconnectAllElements()`, which strips the inputs and outputs of every one of those originals.

So a copy is only safe to read while the original is alive. Decoding a copy after the original has gone reads
disconnected elements. `Solution::translatePhenotypeToGenome()` finds a field's own kernel as the kernel whose
inputs equal its outputs, which every disconnected kernel satisfies, so it picks one at random. Found on 19 October
2026: a test helper that returned `*simulation` from a function, letting the `shared_ptr` die, made the evolved-b2
fixture score 0.24 instead of 0.98 in 12 of 15 runs.

The apps are fine: they keep the loaded `shared_ptr<Simulation>` alive while `makeFromTemplate` decodes it, and
`Population` copies from its starting solution's live phenotype. In tests, `loadFixtureSimulation` in
`tests/test_solutions_tasks.cpp` returns the `shared_ptr` for this reason.

The behaviour is in dnf-composer's `src/simulation/simulation.cpp` and the elements' `clone()`.
