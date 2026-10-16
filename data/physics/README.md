# Physics Benchmark Dataset (`data/physics/`)

This directory contains formal theoretical and mathematical physics proof/derivation tasks formulated in the standard TCS-Bench schema.

## Tasks Overview

| Task ID | arXiv Reference | Problem Type | Target Theorem | Description |
| :--- | :---: | :---: | :--- | :--- |
| `task_gr-qc_9504004_statement_01_01.json` | `gr-qc/9504004` | **Closed (with GT)** | **Einstein Equation of State** *(Jacobson, 1995)* | Derivation of Einstein field equations $R_{ab} - \frac{1}{2}Rg_{ab} + \Lambda g_{ab} = \frac{2\pi}{\hbar\eta} T_{ab}$ from horizon thermodynamics $\delta Q = T dS$. |
| `task_2510.24491_statement_01_01.json` | `2510.24491` | **Closed (with GT)** | **Semiclassical Einstein Equations from Relative Entropy** *(Dorau & Much)* | Derivation of semiclassical gravitational field equations from Araki-Uhlmann relative entropy on horizon von Neumann algebra for coherent states. |
| `task_2510.24491_open_01_01.json` | `2510.24491` | **Open Problem (no GT)** | **Semiclassical Einstein Equations for General Hadamard States** | Extension of the Dorau-Much relative entropy flux formula to arbitrary normal Hadamard states on the horizon algebra (currently an unsolved open problem in mathematical QFT / semiclassical gravity). |

## Task JSON Schema
Each task contains:
- `arxiv_id`: Primary arXiv category & identifier (e.g. `gr-qc/9504004`, `2510.24491`).
- `statement_id`: Identifier of the theorem/derivation (`statement_01`, `open_01`).
- `prompt`: Full problem formulation including physical system model, background lemmas, evaluation criteria (including physical & dimensional integrity), response format, and the target theorem.
- `ground_truth_solution`: Peer-reviewed reference proof (or `null` for frontier open problems).
