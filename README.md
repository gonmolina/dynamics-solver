# Dynamics Solver 🚀

[![Python Version](https://img.shields.io/badge/python->=%203.10-blue.svg)](https://www.python.org/)
[![Development Status](https://img.shields.io/badge/status-work%20in%20progress-orange.svg)]()
[![Built with AI](https://img.shields.io/badge/built%20with-100%25%20AI-purple.svg)]()
[![PyPI](https://img.shields.io/pypi/v/dynamics-solver.svg)](https://pypi.org/project/dynamics-solver/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

🌐 **Read in other languages:** [Español](LEAME.md)

**Dynamics Solver** is a Python library designed for modeling, interconnecting, and simulating complex non-linear dynamic systems and control block diagrams.

> ⚠️ **IMPORTANT NOTICE / DISCLAIMER**  
> This project has been **developed 100% using Artificial Intelligence tools** (AI-driven development) and is currently in an **active phase of development (Work in Progress / Alpha)**. APIs and features may evolve or change as the project matures.

---

## ✨ Key Features

* **Decoupled & Interconnected Architecture**: 
  * Subsystems (`SubsystemBase`) are pure, independent mathematical models that store no references to other blocks or the container network.
  * The entire network topology is managed centrally via `InterconnectedSystem.connect(...)`.

* **Automated Topological Execution Order**:
  * Resolves block execution order in each simulation step using Kahn's Algorithm.
  * Automatically detects instantaneous algebraic loops and identifies inconsistencies in blocks with direct feedthrough (`direct_feedthrough`).

* **Steady-State Analysis & Linearization**:
  * `find_steady_state(...)`: Calculates the equilibrium state vector $x_{ss}$ given constant or operational inputs.
  * `linearize(...)`: Numerically computes the Jacobian matrices of the linearized state-space system ($A, B, C, D$) between configured external inputs and outputs.

* **Library of Basic & Dynamic Blocks**:
  * **Dynamic**: Continuous Integrator, Transfer Functions $G(s)$, State-Space Systems ($A, B, C, D$).
  * **Operators**: Multi-input Adder/Subtractor (with implicit adders on compound inputs), Multiplier/Divider.
  * **Input Sources**: Constant Sources, Step Sources, and generic Time Functions $u(t)$.

* **Flexible Signal Naming (States, Inputs, and Outputs)**:
  * Automatic label generation based on hierarchy: $x_{\text{system}\_1}$, $u_{\text{system}\_1}$, $y_{\text{system}\_1}$.
  * Allows assigning custom names individually or in batch (`set_state_name`, `set_input_names`, etc.), which are automatically preserved even if the container subsystem is renamed.

* **Integrated Result Visualization**:
  * Built-in `plot_results(...)` tool based on `matplotlib` for generating clear multi-subplot figures for inputs and outputs.

---

## 📦 Installation

You can install the library directly from **PyPI**:

```bash
pip install dynamics-solver
```

---

## ⚡ Quick Start Example

The following example demonstrates how to connect an integrator with a gain in open-loop:

```python
import numpy as np
from dynamics_solver import InterconnectedSystem, SubsystemBase

# 1. Define custom or standard blocks
class Integrator(SubsystemBase):
    def __init__(self, name="integrator", x0=None):
        super().__init__(state_dim=1, input_dim=1, output_dim=1, name=name, direct_feedthrough=False, x0=x0)

    def update_outputs(self, x, u):
        return x

    def compute_derivatives(self, x, u):
        return u

class Gain(SubsystemBase):
    def __init__(self, k=2.0, name="gain"):
        self.k = k
        super().__init__(state_dim=0, input_dim=1, output_dim=1, name=name, direct_feedthrough=True)

    def update_outputs(self, x, u):
        return self.k * u

    def compute_derivatives(self, x, u):
        return np.array([])

# 2. Create the interconnected system and add blocks
sys_net = InterconnectedSystem()
int1 = Integrator("int1", x0=[1.0])
gain1 = Gain(k=3.0, name="gain1")

sys_net.add_subsystem(int1)
sys_net.add_subsystem(gain1)

# 3. Connect the output of the integrator to the input of the gain
sys_net.connect(int1, 0, gain1, 0)

# 4. Simulate the network
t, x_global, states, outputs = sys_net.solve(
    t_span=(0.0, 5.0),
    u_ext_func=lambda t: np.array([2.0])
)

print("Gain output at t=5s:", outputs["gain1"][-1])
```

---

## 📚 Advanced Examples & Documentation

For more advanced application cases, check out the notebooks and examples included in the repository:

* **Nuclear Reactor Control**: [`ejemplos/control_reactores.py`](ejemplos/control_reactores.py) and [`ejemplos/control_reactores.md`](ejemplos/control_reactores.md)
  * Models point neutron kinetics with precursors, thermal transfer in the coolant, and thermal reactivity feedback.
  * Includes transient simulation and reactor linearization around steady-state.
* **Complete Simulation Documentation**: [`docs/simulation_docs.md`](docs/simulation_docs.md)

---

## 🛠️ Project Status (AI-Generated & WIP)

This module was conceived and generated entirely using Language Models and Artificial Intelligence as an advanced exploration of software engineering for dynamic simulation.

If you find any issues or have suggestions for the architecture, contributions and issues are more than welcome!

---

## 📄 License

This project is distributed under the [MIT](LICENSE) license.
