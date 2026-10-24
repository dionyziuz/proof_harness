# From Spacetime Thermodynamics to Open Quantum Horizons: How an Autonomous Prover Learns on the Fly

### *How we solved Ted Jacobson's 1995 Einstein Equation of State, reproduced Dorau & Much's 2026 semiclassical gravity proof, and engineered a 100-turn self-evolving harness to attack open mathematical physics.*

---

## The Frontier Beyond Toy Benchmarks

In 1995, mathematical physicist Ted Jacobson published a four-page paper in *Physical Review Letters* that fundamentally rewired theoretical physics: *Thermodynamics of Spacetime: The Einstein Equation of State*. 

Jacobson showed that Albert Einstein's field equations of general relativity:
$$R_{ab} - \tfrac{1}{2} R g_{ab} + \Lambda g_{ab} = 8\pi G \, T_{ab}$$
are not fundamental dynamical equations that need to be quantized. Instead, they are thermodynamic equations of state—emerging naturally from the equilibrium Clausius relation $\delta Q = T \, dS$ applied to local Rindler horizons across every spacetime point, with the Unruh temperature $T = \frac{\hbar \kappa}{2\pi}$ and the Bekenstein-Hawking entropy $S = \eta \, \mathcal{A}$.

Thirty years later, in a March 2026 paper (*Physical Review Letters* 136, 091602), Philipp Dorau and Albert Much took Jacobson’s thermodynamic vision into the rigorous foundations of Algebraic Quantum Field Theory (AQFT). Using Tomita–Takesaki modular theory and the Araki–Uhlmann relative entropy, they proved that for coherent states on a bifurcate Killing horizon, quantum relative entropy equals the classical energy flux, yielding the semiclassical Einstein equations directly from quantum information.

Yet at the end of their paper lies a formidable frontier: **The Open Problem**. 

What happens when matter is *not* a tidy coherent state, but an arbitrary, highly entangled **Hadamard state**? For arbitrary states, the modular operator does not act as a pure geometric boost. An entanglement obstruction emerges. No proof has ever been published for general states.

Could an autonomous, neuro-symbolic agent harness—powered by an open-weight 27B model running on local hardware—tackle both foundational physics proofs, extract verified heuristics from their derivations on the fly, enrich itself with frontier literature, and mount an orchestrated, 100-turn deep-horizon assault on the unsolved problem?

Here is how we built that pipeline, how the engine performed on the solved physics benchmarks, and how we engineered an autonomous research loop to attack open theoretical physics.

---

## 1. The Physics Benchmark Suite

To test whether our reasoning architecture ([Proof Harness](https://github.com/)) could operate at the level of graduate mathematical physics, we configured three tasks from recent spacetime thermodynamics and AQFT literature:

| Problem ID | Theorem / Domain | Status in Literature | Target Equation |
| :--- | :--- | :--- | :--- |
| **`gr-qc/9504004`** | **Jacobson (1995)**: Einstein Eq. of State | Solved (PRL 1995) | $\delta Q = T \, dS \implies G_{ab} + \Lambda g_{ab} = \frac{2\pi}{\hbar\eta} T_{ab}$ |
| **`2510.24491` (Statement 1)** | **Dorau & Much (2025/2026)**: Coherent State Horizon Flux | Solved (PRL 2026) | $S^{\mathrm{rel}}(\omega_0 \| \omega_\phi) \propto \delta\mathcal{A} \implies G_{ab} + \Lambda g_{ab} = 8\pi \langle :T_{ab}: \rangle$ |
| **`2510.24491` (Open 1)** | **General Hadamard State Horizon Flux** | **Unsolved (Open Problem)** | Prove $S^{\mathrm{rel}} = \text{Flux}$, or characterize non-local entanglement obstruction |

Unlike multiple-choice or short-answer math benchmarks, each task requires generating a complete, self-contained, publication-grade LaTeX proof. The proof is evaluated by an independent, multi-replicate referee judge across four strict axes: **mathematical rigor, completeness, boundary/regime discharge, and absence of hallucinations**.

```
                        ┌──────────────────────────────────────────────────────────┐
                        │             Autonomous Physics Proof Engine              │
                        └────────────────────────────┬─────────────────────────────┘
                                                     │
                   ┌─────────────────────────────────┴─────────────────────────────────┐
                   ▼                                                                   ▼
       ┌───────────────────────┐                                           ┌───────────────────────┐
       │   Jacobson (1995)     │                                           │  Dorau & Much (2026)  │
       │   (Classical Thermo)  │                                           │   (AQFT Coherent)     │
       └───────────┬───────────┘                                           └───────────┬───────────┘
                   │                                                                   │
                   │ Solved: Turn 3 (0.86 Pass)                                        │ Solved: Turn 5 (0.87 Pass)
                   ▼                                                                   ▼
       ┌───────────────────────┐                                           ┌───────────────────────┐
       │ Invariant Extraction  │                                           │ Invariant Extraction  │
       │ (Shrinking Horizons)  │                                           │ (Coefficient Matching)│
       └───────────┬───────────┘                                           └───────────┬───────────┘
                   │                                                                   │
                   └─────────────────────────────────┬─────────────────────────────────┘
                                                     ▼
                                      ┌─────────────────────────────┐
                                      │  TurboVec Procedural Memory │
                                      │   (prime_math_procedural)   │
                                      └──────────────┬──────────────┘
                                                     │
                                                     │ + Web Research Ingestion
                                                     │   (Casini, Wall, Bousso QNEC)
                                                     ▼
                                      ┌─────────────────────────────┐
                                      │     The 100-Turn Prover     │
                                      │  Target: The Open Problem   │
                                      │  (General Hadamard States)  │
                                      └─────────────────────────────┘
```

---

## 2. Phase 1: Solving the Known Theorems

We deployed Proof Harness using `Qwen 3.8 27B` with thinking tokens enabled, wrapped by our specialized zero-trust execution environment, context compactor, and subagent swarm (Context Architect, Computational Falsifier, Adversarial Skeptic, and Publication Scribe).

### Task 1: Jacobson (1995) — The Einstein Equation of State
In `task_gr-qc_9504004_statement_01_01`, the solver was given only the local Rindler horizon setup, Unruh temperature, Raychaudhuri expansion near a stationary bifurcation surface, and energy-momentum conservation.

The prover decomposed the challenge in **3 turns (20.0 minutes total wall-clock time)**:
1. **Turn 1 (Blueprint & Invariant Formulation)**: The Context Architect identified that the area variation $\delta\mathcal{A} = \int \theta \, d\lambda \, d\mathcal{A}$ combined with the Raychaudhuri expansion $\theta = -\lambda R_{ab} k^a k^b + \mathcal{O}(\lambda^2)$ yields:
   $$T \, dS = -\frac{\hbar \kappa \eta}{2\pi} \int_{\mathcal{H}} \lambda \, R_{ab} k^a k^b \, d\lambda \, d\mathcal{A}$$
2. **Turn 2 (Algebraic & Null-Vector Verification)**: The model tested the null tensor identity in the local Python kernel. Equating integrands for all null generators $k^a$ yields:
   $$T_{ab} k^a k^b = \frac{\hbar\eta}{2\pi} R_{ab} k^a k^b \implies \frac{2\pi}{\hbar\eta} T_{ab} = R_{ab} + f \, g_{ab}$$
3. **Turn 3 (Divergence & Bianchi Identity Synthesis)**: Taking the covariant divergence $\nabla^a$ of both sides and invoking $\nabla^a T_{ab} = 0$ alongside the contracted Bianchi identity $\nabla^a (R_{ab} - \frac{1}{2} R g_{ab}) = 0$ forced $\nabla_b (f + \frac{1}{2} R) = 0$, proving $f = -\frac{1}{2} R + \Lambda$ for a universal constant $\Lambda$.

**Result**: Scored **0.86 / 1.00 (Strong Pass)** by the external TCS Judge (`reps=[0.86], P(pass)=1.0`).

### Task 2: Dorau & Much (2025/2026) — Quantum Relative Entropy on Coherent States
In `task_2510.24491_statement_01_01`, the mathematical physics becomes substantially more delicate:
* Matter is quantized as a Klein-Gordon field $\Phi$ on a bifurcate Killing horizon.
* The observable algebra $\mathscr{N}_{\mathcal{R}}$ is a von Neumann factor.
* The state $\omega_\phi$ is a coherent excitation generated by a Weyl operator.
* The target requires proving that $S^{\mathrm{rel}}(\omega_0 \| \omega_\phi)$ matches the energy flux, and deriving the exact coupling constant $8\pi$ under the Bekenstein-Hawking normalization $S^{\mathrm{rel}} = \delta\mathcal{A}/4$.

The prover faced a subtle trap that trips up many human graduate students: **the degenerate zero-flux problem**. If an author derives the constant $C$ in $R_{\xi\xi} = C T_{\xi\xi}$ by naively dividing the integrated area variation by the flux integral $\delta Q$, the proof is invalid whenever the state has vanishing net flux across the horizon portion.

On Turn 5, our solver formulated an elegant, coordinate-free coefficient comparison:
> *"The equality $\delta Q_{\mathcal{R}} = \delta\mathcal{A}_{\mathcal{R}}/4$ holds as a local coefficient relation across all local horizon portions and all admissible profiles $\phi$. Comparing the integrand coefficients yields $-2\pi = -C/4 \implies C = 8\pi$ identically, without dividing by the integral of any specific excitation."*

The internal referee auditor evaluated the 6,022-character draft:
```json
{
  "statement_fidelity_exact": true,
  "inequality_directions_verified": true,
  "theorem_hypotheses_verified": true,
  "rigor": 0.86,
  "completeness": 0.88,
  "boundary_cases": 0.85,
  "validity": 0.87,
  "verdict": "VALID",
  "issues": "none"
}
```

---

## 3. Phase 2: Learning on the Fly via Closed-Loop Memory

When an LLM solves a problem in a standard chat session or static script, that mathematical reasoning is wiped out the moment the process terminates. Every new problem starts with a blank slate.

In Proof Harness, every problem run feeds directly into an autonomous cognitive cycle based on the **CoALA (Cognitive Architectures for Language Agents)** framework:

```
        ┌────────────────────────────────────────────────────────┐
        │                 Closed-Loop Memory Cycle               │
        └───────────────────────────┬────────────────────────────┘
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │    Problem Solved &       │
                      │    Audited (VALID)        │
                      └─────────────┬─────────────┘
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │   Universal Heuristic     │
                      │   Validation Filter       │
                      └─────────────┬─────────────┘
                                    │
                       Passed Gate  │  Rejected (Overfitted/Trivial)
                      ┌─────────────┴─────────────┐
                      ▼                           ▼
        ┌───────────────────────────┐   ┌───────────────────┐
        │ TurboVec Vector Database  │   │  Discarded from   │
        │  (prime_math_procedural)  │   │  Global Memory    │
        └─────────────┬─────────────┘   └───────────────────┘
                      │
                      ▼
        ┌───────────────────────────┐
        │  Next Prover Session:     │
        │  Semantic Retrieval (k=5) │
        │  Injected into Prompt     │
        └───────────────────────────┘
```

The moment Task 1 passed the referee, the system's memory hygiene pipeline triggered:
1. **Rule Extraction**: The model extracted the core strategic invariant:
   > *"When deriving local field equations from global thermodynamic identities, employ a shrinking-horizon limit to isolate pointwise tensor components, rigorously bounding higher-order curvature and expansion terms to ensure they vanish relative to the leading-order geometric integrals."*
2. **Universality Filter (`heuristic_validation.py`)**: The proposed heuristic was checked against leak-safety and universality gates:
   - Must be $\ge 25$ characters with genuine mathematical substance.
   - Zero problem-specific constants or hardcoded IDs.
   - Zero brute-force or numeric extrapolation hacks.
3. **TurboVec Indexing**: Embedded via fast ONNX vectors and stored into `prime_math_procedural`.

When subsequent physics tasks launch, their prompts are automatically pre-seeded with these learned heuristics. The system grows smarter with every problem it solves.

---

## 4. Phase 3: Domain Enrichment from Frontier Literature

With both baseline problems solved and their structural invariants captured, we turned to the ultimate objective: **The Open Problem** (`task_2510.24491_open_01_01`).

The open problem asks:
> *Can the derivation of semiclassical Einstein equations from Araki-Uhlmann relative entropy be extended to **arbitrary (non-coherent) Hadamard states** $\omega$ on the horizon algebra $\mathscr{N}_{\mathcal{R}}$?*

The problem statement included a crucial remark:
> *"Remark (open problem): no proof of this statement is known. A submission may prove it in full, prove it under additional explicitly stated hypotheses, or precisely characterize the obstruction. Any added hypothesis must be stated explicitly as part of the theorem and its physical plausibility discussed; smuggling an assumption in as an 'obvious' step fails criterion 1."*

Because this is a genuine unsolved research question, there was no "ground truth" to cheat on. To give our solver the best possible conceptual foundation, we conducted targeted online research into recent advances in Algebraic Quantum Field Theory, Tomita–Takesaki modular theory, and null-surface thermodynamics.

Our research synthesized three major theoretical pillars:

### Finding 1: The Relative Entropy Decomposition (Casini 2008, Faulkner et al. 2014)
For any quantum state $\omega$ relative to a reference state $\omega_0$ with modular Hamiltonian $K_0 = -\ln \Delta_{\omega_0}$:
$$S^{\mathrm{rel}}(\omega_0 \| \omega) = \Delta \langle K_0 \rangle - \Delta S_{\mathrm{ent}}$$
* In a **coherent state**, the excitation is created by a unitary displacement $W(\phi)$, which preserves the vacuum entanglement spectrum. Hence $\Delta S_{\mathrm{ent}} = 0$, and relative entropy equals the classical boost energy flux $\Delta \langle K_0 \rangle$.
* In a **general Hadamard state**, $\Delta S_{\mathrm{ent}} \neq 0$. The state contains non-local quantum correlations across the horizon cut. Setting $S^{\mathrm{rel}}$ equal *only* to the energy flux is mathematically false unless the entanglement entropy difference is explicitly accounted for.

### Finding 2: Modular Flow on Null Horizons (Ciolli, Longo, & Ruzzi 2019, 2021/2022)
In *The Information in a Wave* and *Relative Entropy and Curved Spacetimes*, Ciolli, Longo, and Ruzzi proved that translating Rindler spacetime along its upper null horizon using half-sided modular inclusions produces a non-linear convex modular curve for non-coherent states. This directly confirmed the non-local nature of the modular Hamiltonian away from the coherent sector.

### Finding 3: The Quantum Null Energy Condition (Bousso, Fisher, Leichenauer, & Wall 2014–2016)
On null surfaces, the second functional variation of the generalized entropy isolates the local stress tensor:
$$\langle T_{kk} \rangle \ge \frac{1}{2\pi} \frac{\partial^2 S_{\mathrm{ent}}}{\partial U^2}$$
If one wishes to derive the semiclassical Einstein equations for arbitrary Hadamard states, the equations must include an effective quantum entanglement stress tensor $\tau_{ab}^{[\mathrm{ent}]} \propto \frac{\partial^2 S_{\mathrm{ent}}}{\partial U^2}$.

---

### The Five Injected Frontier Heuristics

We codified these mathematical insights into five leak-safe, universal procedural heuristics and committed them into TurboVec memory:

```python
heuristics = [
    # Heuristic 1: The Relative Entropy Decomposition Invariant
    "When evaluating Araki-Uhlmann relative entropy across a null horizon in algebraic quantum field theory, "
    "decompose the relative modular generator into the geometric vacuum boost modular energy flux and the non-local entanglement entropy variation. "
    "Explicitly isolate the non-local entanglement defect Delta S_ent rather than assuming the relative modular operator acts as a pure geometric dilation for general non-coherent states.",

    # Heuristic 2: The Iso-Entropic Subspace Characterization
    "To establish equality between horizon relative entropy and energy flux for general Hadamard states, identify the iso-entropic subspace "
    "where the horizon-cut entanglement entropy matches the local vacuum. Show that unitary horizon displacement operations "
    "preserve the local vacuum entanglement spectrum, extending the coherent-state flux identity to all entanglement-preserving Hadamard states.",

    # Heuristic 3: The First Law of Modular Perturbations
    "In perturbative semiclassical gravity, apply the First Law of Entanglement to first-order state variations around the local vacuum. "
    "Demonstrate that linear variations in non-local entanglement entropy cancel identically against the modular Hamiltonian variation, "
    "guaranteeing that the horizon energy flux governs the leading-order variation of the semiclassical Einstein tensor.",

    # Heuristic 4: The Near-Bifurcation Scaling Limit
    "Near an instantaneously stationary bifurcate horizon where expansion and shear vanish, apply the near-horizon scaling limit along the null affine parameter. "
    "Leverage the universal short-distance Hadamard singularity to prove that non-local multi-point entanglement correlations scale at order O(U^2), "
    "vanishing relative to the linear boost generator and restoring the local stress-energy tensor proportionality.",

    # Heuristic 5: Effective Entanglement Stress Tensor (Obstruction Characterization)
    "When characterizing the non-local obstruction to the semiclassical Einstein equation for arbitrary Hadamard states, "
    "formulate the generalized gravitational field equations with an effective quantum entanglement stress tensor: "
    "G_ab + Lambda g_ab = 8 pi (<:T_ab:> + tau_ab^[ent]), where tau_ab^[ent] represents the second null functional derivative of the horizon entanglement entropy matching the Quantum Null Energy Condition."
]
```

When the prover queries its memory bank during the open problem solve, these five exact rules are retrieved at the very top of its attention context.

---

## 5. Phase 4: Engineering the 100-Iteration Deep Prover

A standard 8-turn budget is sufficient for textbook theorems where the reduction path is known. But for an open research problem requiring hypothesis generation, counterexample falsification, and multi-branch exploration, 8 turns is a straightjacket.

To attack the open problem, we overhauled Proof Harness to support **deep-horizon research runs of up to 100 turns**.

```
                        ┌────────────────────────────────────────────────────────┐
                        │             The 100-Turn Research Engine               │
                        └───────────────────────────┬────────────────────────────┘
                                                    │
                                                    ▼
                               ┌──────────────────────────────────────────┐
                               │   Configurable Turn & Timeout Scaling    │
                               │   (--max-turns 100, dynamic budget)      │
                               └────────────────────┬─────────────────────┘
                                                    │
                                                    ▼
                               ┌──────────────────────────────────────────┐
                               │     Context Compactor & Sliding Anchor   │
                               │   (Protects task, compacts scratchpad)   │
                               └────────────────────┬─────────────────────┘
                                                    │
                                                    ▼
                               ┌──────────────────────────────────────────┐
                               │     Deep Stagnation Pivot & Archive      │
                               │   (Archives best draft, clears kernel)   │
                               └────────────────────┬─────────────────────┘
                                                    │
                                                    ▼
                               ┌──────────────────────────────────────────┐
                               │     End-of-Run Cross-Pivot Recovery      │
                               │  (Restores highest-scoring draft scored) │
                               └──────────────────────────────────────────┘
```

Here are the key engineering pillars that make 100-turn runs possible without system freeze or attention collapse:

### 1. Dynamic Timeout Auto-Scaling
Previously, timeouts were hardcoded to a 1-hour wall-clock limit. In a 100-turn run with deep thinking tokens, a static 1-hour timeout guarantees premature termination.

We updated [`core_engine/prover.py`](https://github.com/) to auto-scale both the per-task budget and the per-call stream timeout:
```python
# Auto-scale total solve time for deep-horizon runs
base_timeout = float(tcs_cfg.solver_timeout_seconds)
if current_max_turns > 8:
    per_turn_s = 450.0  # 7.5 minutes per turn budget
    max_time_seconds = max(base_timeout, current_max_turns * per_turn_s)
```
For a 100-turn run, this allocates an expansive **12.5-hour total solve window**, ensuring the model is never cut short mid-derivation.

### 2. Context Compaction & The Sliding Anchor
Autoregressive LLMs degrade when conversation histories balloon beyond 30,000 tokens ("lost-in-the-middle" syndrome). 

Our context compactor intercepts the message history every 2–3 turns:
* **Protected Anchors**: The system prompt, original target statement, and Context Architect blueprint are permanently pinned at indices 0 and 1.
* **Middle Compaction**: Intermediate REPL outputs, verbose code traces, and rejected scratchpad prose are summarized into structured deductive lemmas.
* **Recent Window**: Only the two most recent turns are preserved in full fidelity.

This keeps active context tokens under 24,000 tokens indefinitely—even across 100 turns.

### 3. Deep Stagnation Multi-Pivot & Cross-Pivot Archive
If an LLM fails an audit 3 times in a row, standard provers enter an "apology loop," repeatedly rephrasing the same defective equation.

Our prover implements **Deep Stagnation Multi-Pivoting**:
1. When consecutive rejections reach 6 turns, the solver acknowledges a dead reduction path.
2. It calls `pool.archive_for_pivot()`, safely storing its highest-scoring candidate draft in a cross-pivot memory vault.
3. It completely purges the Python kernel namespace and state machine (`kernel.reset()`).
4. It issues a **Clean-Slate Hard Pivot**, strictly banning the refuted premises and instructing the model to formulate a fundamentally fresh proof architecture.
5. If the new branch fails to discover a superior proof, `pool.restore_best_from_archive()` automatically resurrects the highest-validity candidate from across all explored branches at session end.

### 4. Hardware-Aware Throttling on Unified APUs
Our testing environment is powered by an AMD Ryzen AI Max+ 395 (Strix Halo APU) with 64GB Unified Memory (`uma: 1`), where system RAM is dynamically shared between CPU page cache and GPU VRAM.

Uncontrolled disk sweeps or unbuffered logs can trigger extreme memory pressure on the unified bus. We implemented:
* Streamlined, non-blocking chunked writes to task logs.
* Direct in-memory buffer recycling.
* Socket-level TCP keepalives (`PRIME_MATH_TCP_KEEPALIVE=true`) with exponential jitter backoff (5 retries), preventing silent stream drops during 10-minute thinking pauses.

---

## 6. Launching the Open Problem Assault

With the baseline theorems solved, memory heuristics indexed, and the 100-turn engine primed, launching an autonomous deep-horizon research run into the open problem is executed with a single command:

```bash
cd /home/sysadmin/Documents/github/proof_harness
.venv/bin/python scripts/run_open_problems_prove.py \
  --env-file .env.physics_qwen \
  --tasks-dir data/physics/tasks \
  --task-id task_2510.24491_open_01_01 \
  --max-turns 100 \
  --prove-dir data/eval_results/physics_open_100turns_v1 \
  --telegram
```

As the solver iterates through its turns, live progress updates are dispatched directly to Telegram:
* Real-time turn numbers, REPL execution outputs, and proof candidate lengths.
* Auditor critique notes flagging whether the model is attempting an iso-entropic hypothesis, a near-horizon scaling limit, or an effective entanglement stress tensor derivation.
* Automatic candidate recovery and final referee validity scoring.

---

## The Takeaway

Large language models are often treated as static answer engines—useful for autocomplete, summarization, or coding assistance, but fundamentally incapable of real mathematical creativity.

What our experiments demonstrate is that **the limiting factor is rarely the raw model weights; it is the scaffolding around them**:
1. When guided by a **zero-trust deterministic kernel**, false claims are killed before they poison the proof.
2. When equipped with **closed-loop procedural memory**, breakthroughs on simple baselines immediately inform subsequent explorations.
3. When enriched with **frontier literature heuristics**, the model avoids known dead ends and attacks the problem using modern mathematical frameworks.
4. When backed by **multi-pivot recovery and auto-scaled 100-turn horizons**, the agent has the time and space to explore, fail, pivot, and synthesize.

We don't yet know if a 27B model will fully solve the general Hadamard horizon problem. But for the first time, we have an autonomous reasoning engine that can read the literature, solve the prerequisites, learn the invariants on the fly, and spend 12 hours exploring the frontier of quantum gravity—completely unattended.

---

*The full Proof Harness codebase, evaluation data, and benchmark logs are available on GitHub.*
