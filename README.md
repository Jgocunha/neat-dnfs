# neat-dnfs

## A NeuroEvolutionary Framework for Evolving Dynamic Neural Field Architectures

<img src="https://raw.githubusercontent.com/Jgocunha/neat-dnfs/main/neat-dnfs/resources/images/logo.png" alt="logo" width="800" height="800">

---

[![CI](https://img.shields.io/github/actions/workflow/status/Jgocunha/neat-dnfs/ci.yml?branch=main&style=flat-square&logo=githubactions&logoColor=white&label=CI)](https://github.com/Jgocunha/neat-dnfs/actions/workflows/ci.yml)
[![Static Analysis](https://img.shields.io/github/actions/workflow/status/Jgocunha/neat-dnfs/static-analysis.yml?branch=main&style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjZmZmIiBzdHJva2Utd2lkdGg9IjIuNSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIj48Y2lyY2xlIGN4PSIxMCIgY3k9IjEwIiByPSI3Ii8%2BPGxpbmUgeDE9IjE1IiB5MT0iMTUiIHgyPSIyMSIgeTI9IjIxIi8%2BPC9zdmc%2B&logoColor=white&label=static%20analysis)](https://github.com/Jgocunha/neat-dnfs/actions/workflows/static-analysis.yml)
[![Coverage](https://img.shields.io/codecov/c/github/Jgocunha/neat-dnfs?style=flat-square&logo=codecov&logoColor=white)](https://codecov.io/gh/Jgocunha/neat-dnfs)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue?style=flat-square&logo=readthedocs&logoColor=white)](https://jgocunha.github.io/neat-dnfs/)
[![Wiki](https://img.shields.io/badge/wiki-GitHub%20Wiki-blue?style=flat-square&logo=github&logoColor=white)](https://github.com/Jgocunha/neat-dnfs/wiki)

[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?style=flat-square&logo=cplusplus&logoColor=white)](https://en.cppreference.com/w/cpp/20)
[![CMake](https://img.shields.io/badge/CMake-3.31%2B-064F8C?style=flat-square&logo=cmake&logoColor=white)](https://cmake.org)
[![Windows](https://img.shields.io/badge/Windows-0078D6?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0iI2ZmZiI%2BPHBhdGggZD0iTTAgMy40NDlMOS43NSAyLjF2OS40NTFIMG0xMC45NDktOS42MDJMMjQgMHYxMS40SDEwLjk0OU0wIDEyLjZoOS43NXY5LjQ1MUwwIDIwLjY5OU0xMC45NDkgMTIuNkgyNFYyNGwtMTIuOS0xLjgwMSIvPjwvc3ZnPg%3D%3D&logoColor=white)](https://github.com/Jgocunha/neat-dnfs/actions)
[![Linux](https://img.shields.io/badge/Linux-FCC624?style=flat-square&logo=linux&logoColor=black)](https://github.com/Jgocunha/neat-dnfs/actions)
[![macOS](https://img.shields.io/badge/macOS-000000?style=flat-square&logo=apple&logoColor=white)](https://github.com/Jgocunha/neat-dnfs/actions)

**neat-dnfs** is a C++ framework that extends **NeuroEvolution of Augmenting Topologies (NEAT)** to the automated synthesis of **Dynamic Neural Field (DNF)** architectures.
It enables the joint evolution of **continuous-time neural dynamics**, **kernel-based interactions**, and **architectural topology**, supporting the discovery of compact and interpretable Dynamic Field Theory (DFT) models without manual tuning.

---

## Overview

Dynamic Neural Fields (DNFs) provide a biologically grounded and mathematically principled framework for modelling neural population dynamics underlying perception, working memory, selection, and decision-making. Despite their expressive power, DNF architectures are traditionally **hand-designed and manually parameterised**, a process that is time-consuming, difficult to generalise, and highly dependent on expert knowledge.

**neat-dnfs** addresses this limitation by integrating DNFs with **neuroevolution**.
By extending NEAT to operate directly on neural fields and spatial interaction kernels—rather than discrete neurons and scalar weights—the framework enables the **autonomous discovery of DNF architectures** that exhibit desired dynamical behaviours.

The system evolves both:

* **Intrinsic field dynamics** (e.g., time constants, resting levels, kernel profiles)
* **Inter-field structure** (number of fields and their spatial couplings)

Evolution proceeds from minimal architectures and introduces complexity **only when required by task constraints**, in line with the minimal cognitive construction principle.

---

## Key Features

* **Evolution of Dynamic Neural Field Architectures**
  Simultaneous evolution of neural field parameters and architectural topology.

* **Continuous-Time, Kernel-Based Neuroevolution**
  Genomes encode spatial interaction kernels and field dynamics instead of discrete synaptic weights.

* **Incremental Structural Complexification**
  New fields and interactions emerge gradually through NEAT-style structural mutations.

* **Interpretability by Design**
  Evolved solutions consist of explicit neural fields with identifiable functional roles.

* **Task-General Framework**
  Applicable to a hierarchy of DFT-inspired tasks, from basic instabilities to compositional cognitive paradigms.

* **Comprehensive Evolutionary Analysis**
  Built-in logging, statistics, and visualisation of species, genomes, and architectural growth.

---

## Architecture

### Core Components

| Component             | Description                                                                              |
| --------------------- | ---------------------------------------------------------------------------------------- |
| **Genome**            | Encodes a DNF-based architecture as field genes and interaction genes.                   |
| **Population**        | Manages evolution, evaluation, selection, and reproduction.                              |
| **Species**           | Groups similar architectures to protect structural innovation.                           |
| **Solution**          | Defines task-specific fitness evaluation based on field dynamics.                        |
| **Field Genes**       | Represent individual neural fields (input, hidden, output) and their intrinsic dynamics. |
| **Interaction Genes** | Represent spatially structured kernel-based couplings between fields.                    |

<img src="https://raw.githubusercontent.com/Jgocunha/neat-dnfs/main/neat-dnfs/resources/images/phenotype-genotype-mapping-wb.png">

**Genotype-to-phenotype mapping in neat-dnfs.**
*Field genes encode intrinsic neural field dynamics, while interaction genes specify kernel-defined couplings. Together, they map directly to a continuous-time DNF architecture.*

---

## Evolutionary Process

1. **Initialization** – Start from ultra-minimal architectures (input and output fields only).
2. **Simulation** – Evaluate continuous-time DNF dynamics under task-specific stimuli.
3. **Fitness Evaluation** – Assess qualitative dynamical properties (e.g., peak formation, stability, selection).
4. **Speciation** – Protect novel architectural innovations using compatibility distance.
5. **Selection & Reproduction** – Apply NEAT-style crossover and fitness sharing.
6. **Mutation** – Refine parameters or introduce new fields and interactions.

---

## Implemented Tasks

The framework includes a hierarchy of benchmark tasks designed to probe increasingly complex DFT mechanisms:

### Core Dynamic Mechanisms

* **Detection Instability** – Transient input-driven activation and decay
* **Memory Instability** – Self-sustained activation without input
* **Selection Instability** – Winner-take-all competition

### Compositional Tasks Requiring Structural Innovation

* **Delayed Match-to-Sample (DMTS)** – Internal memory biasing later selection
* **Inhibition of Return (IOR)** – Delayed inhibitory bias against previously selected locations

Additional simple logical tasks (e.g., AND, XOR) are included for validation and demonstration.

---

## Building

### Prerequisites

* **CMake 3.31.6+**
* **C++20** compiler
* **VCPKG** package manager

**Dependencies (via VCPKG):**

* `imgui`, `implot`, `imgui-node-editor`, `nlohmann-json`

**Additional dependencies:**

* [`imgui-platform-kit`](https://github.com/Jgocunha/imgui-platform-kit)
* [`dynamic-neural-field-composer`](https://github.com/Jgocunha/dynamic-neural-field-composer)

### Build Instructions

```bash
export VCPKG_ROOT=/path/to/vcpkg
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

---

## Usage

> **New here?** [HOW-TO.md](HOW-TO.md) is the practical guide: running experiments and
> ablations, and which config files to edit to change hyperparameters.

### Basic Example

```cpp
#include "neat/population.h"
#include "solutions/xor.h"

XOR solution{ SolutionTopology{{ 
    { {FieldGeneType::INPUT, {50, 1.0}}, 
      {FieldGeneType::INPUT, {50, 1.0}}, 
      {FieldGeneType::OUTPUT, {50, 1.0}} } 
}}};

PopulationParameters parameters{100, 150, 0.95};
Population population{parameters, std::make_unique<XOR>(solution)};
population.initialize();
population.evolve();
```

### Custom Tasks

To define a new task:

1. Inherit from the `Solution` base class
2. Specify the initial minimal topology
3. Implement `evaluate()` using field-dynamics-based fitness criteria

```cpp
void evaluate() override 
{
    // Define fitness in terms of DNF dynamics:
    // peak existence, position, amplitude, width, or decay to baseline
    parameters.fitness = computedFitness;
}
```

### Ablation Studies

`neat-dnfs-evol` and `neat-dnfs-inc-evol` accept `--task NAME` and `--ablation NAME` at runtime, so
sweeping a task across every mechanism condition is a shell loop, not a rebuild (`--list` prints the
available tasks and ablations):

```bash
neat-dnfs-evol --task and --ablation no-crossover --runs 30 --pop 500 --gens 200 --target 0.9
```

Five conditions are available, each a config-only override applied before `Population::initialize()`
(see `include/neat/ablation_presets.h`):

| Ablation | What it disables |
|---|---|
| `no-growth-io-only` | Structural mutations; genome starts fully connected input/output only |
| `no-growth-one-hidden` | Same, plus one seeded hidden field |
| `no-speciation` | Compatibility-distance species assignment; population starts non-minimal |
| `no-crossover` | Two-parent reproduction; offspring are single-parent clones |
| `random-initial-topology` | Minimal-start bias; genome seeds 1-5 random hidden fields and connections |

Each ablated run writes to its own `data/<Task> <Ablation>/` folder, alongside the unablated
`data/<Task>/` control, so the analysis dashboard (see below) lists every arm as a separate
experiment with no extra setup.

---

## Statistics and Analysis

The framework automatically records:

* **Fitness evolution**
* **Species diversity and lineage**
* **Architectural complexity (fields and interactions)**
* **Mutation and structural growth statistics**

All data are stored in the `data/` directory.

### Analysis Tools (`analysis/`)

A Streamlit dashboard with six pages, split into two scopes:

* **Single run** -- Fitness, Species, Topology, Mutations
* **Across runs** -- Experiment (aggregates every run in one experiment), Compare (several experiments side by side)

Install dependencies once with `pip install -r analysis/requirements.txt`, then run
`launch-visualizer.bat` (Windows) or `launch-visualizer.sh` (macOS/Linux) to open the app, and
select the data root and experiment/run from the sidebar. Each run keeps a persisted cache in
`<run>/.viz_cache/`, safe to delete at any time -- it is rebuilt automatically the next time that
run is opened.

---

## Project Structure

```bash
neat-dnfs/
├── include/
│   ├── neat/          # Core NEAT-DNF implementation
│   ├── solutions/     # Task definitions
│   ├── tools/         # Logging and utilities
│   └── constants.h    # Hyperparameter definition
├── src/
├── apps/
├── tests/
├── data/              # Evolution outputs
├── analysis/          # Post-hoc analysis tools
└── CMakeLists.txt
```

---

## Video explanation

[![Watch the video](https://img.youtube.com/vi/tgNbhQQRmbM/maxresdefault.jpg)](https://youtu.be/tgNbhQQRmbM)

---

## Documentation

For a full exploration of the repository, refer to the [Wiki.](https://github.com/anonymous-author-submissions/neat-dnfs/wiki)

---

## Main inspiration for this work

* Amari, Shun-ichi (1977) - "Dynamics of pattern formation in lateral-inhibition type neural fields"
* Schöner, Gregor and Spencer, John and Research Group, Dft (2015) - "Dynamic Thinking: A Primer on Dynamic Field Theory"
* Nolfi, Stefano and Floreano, Dario (2000) - "Evolutionary robotics: the biology, intelligence, and technology of self-organizing machines"
* Floreano, Dario (2023) - "Bio-Inspired Artificial Intelligence: Theories, Methods, and Technologies"
* Erlhagen, Wolfram and Bicho, Estela (2006) - "The dynamic neural field approach to cognitive robotics"
* Krichmar, Jeffrey L. (2018) - "Neurorobotics — A Thriving Community and a Promising Pathway Toward Intelligent Cognitive Robots"
* Stanley, Kenneth O. and Miikkulainen, Risto (2002) - "Evolving Neural Networks through Augmenting Topologies"
* Erlhagen, Wolfram and Bicho, Estela (2014) - "A Dynamic Neural Field Approach to Natural and Efficient Human-Robot Collaboration"
* Pfeifer, Rolf and Bongard, Josh (2006) - "How the Body Shapes the Way We Think: A New View of Intelligence"
* Coombes, Stephen and Beim Graben, Peter and Potthast, Roland and Wright, James (2014) - "Neural fields: theory and applications"

---

## Citation

If you use this work in your research, please cite:

> J. G. Cunha, W. Erlhagen, R. H. Cuijpers, E. Bicho, "NEAT-DNFs: A NeuroEvolutionary Framework for Evolving Dynamic Neural Field Architectures," in *Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '26)*, Association for Computing Machinery, New York, NY, USA, 2026, pp. 966–974. https://doi.org/10.1145/3795095.3805169





