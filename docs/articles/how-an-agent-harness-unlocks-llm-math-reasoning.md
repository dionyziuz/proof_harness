# How an Agent Harness Unlocks Mathematical Superpowers in a 4-Bit 28B Model

### *The story of how architectural scaffolding, symbolic execution, and cognitive partitioning turned a lightweight 4-bit open-weight LLM into a graduate-level theorem prover.*

---

## 1. Beyond Generic Agent Scaffolding: The Myth of the Monolithic Mathematician

In the modern AI landscape, conventional wisdom dictates that solving frontier, graduate-level mathematics requires massive frontier models: multi-hundred-billion-parameter clusters consuming kilowatts of data center power. While multi-turn agent harnesses and execution loops have become standard practice across software engineering, most mathematical reasoning harnesses remain generic—relying on unguided iterative prompting, conversational multi-agent debates, or raw Python code interpreters that lack formal mathematical semantics and cross-problem learning. When researchers benchmark open-weight models on hard research domains—such as research-level Theoretical Computer Science (TCS)—unassisted first-pass attempts and unguided scaffolds yield predictably bleak results (under 4% passing grades).

Even capable open-weight models routinely hallucinate algebraic leaps, assert false lemmas with supreme confidence, or fail completely when tracking complex invariant inheritance.

We set out to demonstrate that the true performance bottleneck is not raw parameter scale, but harness specialization:

> **Core Thesis:** *Mathematical reasoning is not purely a function of raw parameter scale. It is unlocked by a **specialized neuro-symbolic proving harness** equipped with **continuous domain learning via CoALA vector heuristics** and **zero-trust deterministic testing inside an airgapped mathematical kernel**.*

To test this thesis, we evaluated **unsloth/Qwen3.8-27B-NVFP4**—a 27B/28B parameter model running in **4-bit quantization** on consumer hardware (a single **NVIDIA GeForce RTX 5090** with 32GB VRAM)—across **300 graduate-level research problems from TCS-Bench**, drawn directly from recent theoretical computer science literature on arXiv.

The results demonstrated the transformative power of an agent harness:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│        TCS-Bench 300 Performance: Unassisted Initial Draft vs. Full Proof Harness      │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  Evaluation Paradigm          │ Grade C & Up (≥0.40)│ Grade B & Up (≥0.50)│ Grade A & Up (≥0.70)│
├───────────────────────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
│  1. Unassisted Initial Draft  │    12 / 300 (4.0%)  │    10 / 300 (3.3%)  │     7 / 300 (2.3%)  │
│  2. Full Proof Harness        │    71 / 300 (23.7%) │    63 / 300 (21.0%) │    54 / 300 (18.0%) │
├───────────────────────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
│  Multiplier vs. Initial Draft │      5.92× Gain!    │      6.30× Gain!    │      7.71× Gain!    │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

By wrapping a 4-bit 28B model in **Proof Harness** (`proof_harness`)—a dedicated, closed-loop agentic theorem proving harness—we shattered the original goal of merely doubling or tripling unassisted first-draft reasoning. Across **300 graduate- and PhD-level research problems from TCS-Bench** (drawn directly from recent arXiv theoretical computer science preprints in spectral graph theory, quantum computing, parameterized complexity, and combinatorial algorithms), Proof Harness achieves:

- **All Passing Grades (≥ 0.40, Grade C and Above — Sound & Directional):** **23.67% ($71/300$)** — nearly **sextupling ($5.92\times$)** the unassisted initial draft baseline of $4.00\%$ ($12/300$).
- **Certified Sound Passes (≥ 0.50, Grade B and Above):** **21.00% ($63/300$)** — a **$6.30\times$ multiplier** over the unassisted initial draft baseline of $3.33\%$ ($10/300$).
- **Publication-Grade Proofs (≥ 0.70, Grade A and Above):** **18.00% ($54/300$)** — a **$7.71\times$ multiplier** over unassisted initial generation ($2.33\%$, $7/300$), with 31 proofs reaching near-flawless ($\ge 0.90$) journal quality.
- **Exceptional Proof Validity:** On certified passing proofs, Proof Harness lifts the mean referee validity from flat zero/low-rigor initial guesses to an extraordinary **$0.853 \pm 0.129$** (median **$0.880$**), with structural completeness reaching **$0.907$** and logical rigor reaching **$0.857$**. Exactly **100% of all passing proofs** are discovered through governed multi-turn exploration in Turns 2–12.

At the heart of this leap are **two foundational differentiators** that set Proof Harness apart from standard LLM wrappers:
1. **The Zero-Trust Deterministic Mathematical Kernel**: Providing **immediate, day-one accuracy**, this dual-runtime symbolic, SMT, and arbitrary-precision execution engine refuses to trust the LLM's probabilistic intuition—testing every bound, graph invariant, covariance assumption, and asymptotic claim deterministically in code from the very first solve.
2. **The CoALA Heuristics & Continuous Domain Learning Engine**: Providing **cumulative intelligence over time**, this self-improving procedural and episodic vector memory system (backed by dense vector embeddings) allows the machine to *learn by solving*—enabling practitioners and researchers to train the harness in specialized mathematical or engineering domains purely through embeddings as passed proofs accumulate, without ever touching model weights.

Here is the story of the twin blind spots that break standalone LLMs, and how these two engines—combined with multi-agent cognitive partitioning—turn a compact quantized model into a self-improving mathematician.

---

## 2. The Problem Statement: Why Standalone LLMs Fail on Frontier & Domain-Specific Math

When you deploy an off-the-shelf LLM on graduate-level mathematics or specialized industrial domains (such as quantitative finance, cryptographic protocol verification, or algorithmic complexity), the system hits **two fundamental structural walls**, followed by four tactical failure modes:

### Structural Wall #1: Probabilistic Token Prediction $\neq$ Mathematical Truth (The Trust Fallacy)
Autoregressive transformers generate tokens by maximizing conditional probability—not by verifying mathematical truth. When an LLM relies solely on its internal weights to do math:
- It cannot reliably distinguish a true algebraic identity from a fluent hallucination.
- It cannot deterministically check whether an inequality $f(n) \le g(n)$ flips direction at boundary values $n=1, 2$, whether conditioning on a sum event $E$ destroys pairwise independence (`Cov(A, B) != 0`), or whether a graph lemma fails on $K_{3,3}$.
- **Trusting an LLM to self-verify mathematics purely through prose is a category error.** Without a deterministic computational kernel that forces conjectures through exact symbolic algebra, SMT solvers, and exhaustive small-instance counterexample sweeps, a single unverified leap cascades until the entire proof collapses.

### Structural Wall #2: Frozen Weights & Episodic Amnesia (No Continuous Domain Learning)
Standard LLMs are **stateless at inference time**. An agent might spend 45 minutes and 50,000 tokens discovering a brilliant domain-specific reduction—such as how to bound hypergraph conductance across vertex separators or how to apply Hölder's inequality over shifted dyadic boxes—only to suffer **total amnesia** the moment the session ends.
- On the very next problem in the same domain, the model starts from scratch and repeats the exact same mistakes.
- For engineering teams working in specialized verticals, fine-tuning or running RLHF on a 28B–70B model after every batch of problems is prohibitively expensive, slow, and prone to catastrophic forgetting. **Without a mechanism to continuously extract, validate, and embed reusable mathematical heuristics from its own solving trajectories, an agent cannot compound its domain expertise over time.**

### Four Tactical Failure Modes of Ungoverned Inference
Compounding those two structural walls are four tactical breakdowns in naive prompting:
1. **Premature Convergence (The Turn-1 Trap)**: In single-shot prompting, the LLM commits to an initial algebraic path within its first few hundred tokens. When that path breaks, it papers over the flaw with hand-waving prose rather than backtracking.
2. **Context Poisoning via Scratchpad Noise**: In naive multi-turn loops, exploratory dead ends, stack traces, and formal drafting share a single growing chat transcript, polluting self-attention by Turn 5.
3. **Silent Invariant Violation**: Reductions routinely transform objects (conditioning random variables, taking induced subgraphs, projecting matrices). LLMs blindly assume fragile parent properties (like independence, connectivity, or invertibility) survive transformations that actually destroy them.
4. **Blueprint Dogmatism**: When an initial planning step proposes a loose intermediate bound that fails on a sub-regime, naive agents doubt the *target theorem* and inject unauthorized domain restrictions instead of replacing the intermediate bound.

---

## 3. Core Pillar I: The Zero-Trust Deterministic Mathematical Kernel (`kernel.py`, `prover_kernel.py` & `math_tools.py`)

If the CoALA Heuristics Engine is the harness's **long-term memory**, the **Deterministic Mathematical Kernel** is its **scientific laboratory**.

Instead of trusting the 28B LLM to evaluate an integral, check an asymptotic limit, or verify whether two events are independent in its head, Proof Harness forces mathematical claims through a stateful, airgapped Python execution engine (`PersistentMathKernel` and ephemeral `AuditorKernel`).

```python
class PersistentMathKernel:
    """Zero-trust symbolic/numerical execution engine with deterministic math verification tools.
    When evaluated on isolated benchmarks (such as TCS-Bench), _NetworkIsolationMixin is added for airgapping."""

    # 1. Preloaded Symbolic & SMT Arsenal:
    #    - SymPy (exact rational algebra, symbolic limits, series, closed-form sums)
    #    - Z3 SMT Solver (formal constraint satisfiability & counterexample search)
    #    - mpmath (mp.dps = 50 arbitrary-precision decimal evaluation)
    #    - NetworkX, NumPy, SciPy (graph invariants, spectral eigenvalues)
    #
    # 2. Preloaded Domain Verification Toolkit (`core_engine/math_tools.py`):
    #    - check_asymptotic_dominance(f, g, var='n') -> exact o(), O(), Theta(), omega()
    #    - check_event_covariance(sampler_fn, n_trials=2000) -> detects conditioning dependence
    #    - find_graph_counterexample(predicate_fn, max_n=7) -> exhaustive family sweep
    #    - planar_dual_and_faces(edges), check_cycle_chords_and_holes(adj, cycle) -> topology
    #    - smt_disjoint_paths, smt_check_parameter_hierarchy -> Z3 formal routing & scaling
    #    - check_psd_inequality(A, B), check_quantum_channel_claim -> CPTP & Loewner PSD
    #    - holds_on_samples(rel, max_points=2000) -> honest empirical falsification
    #    - state (MathStateMachine) -> tracks hereditary vs. fragile properties

    def execute(self, code: str, timeout_seconds: Optional[float] = None) -> ExecutionResult:
        # Layer 1: AST Network & Escape-Hatch Denylist (blocks http, socket, urllib,
        #          subprocess, importlib, ctypes, os.system/fork, sys.modules, __import__)
        # Layer 2: Combinatorial Ceiling (blocks permutations(range(n)) for n >= 7,
        #          and caps generator sweeps at MAX_COMBINATORIAL_STEPS = 200,000)
        blocked = self._forbidden_import(code.strip()) or self._combinatorial_ceiling(code.strip())
        if blocked:
            return ExecutionResult(success=False, error=blocked, ...)

        # Layer 3: Thread-local runtime socket guard + _detect_trace_tampering
        return super().execute(code, timeout_seconds or self.default_timeout)
```

### Seven Structural Advantages of Deterministic Kernel Grounding

#### 1. Instantaneous Counterexample Falsification (`Z3`, `SymPy`, `NetworkX`)
To prove a universal theorem, finding even **one** concrete counterexample where the premise holds but the conclusion fails is 100% definitive. When the LLM proposes an intermediate inequality or graph reduction, the kernel tests it in milliseconds—killing false conjectures on Turn 1 or Turn 2 before the model wastes 6 turns writing a 15,000-character proof around a lie.

#### 2. Built-In Domain Diagnostic Primitives (`core_engine/math_tools.py`)
Rather than hoping the LLM writes bug-free test scripts from scratch every time, `PersistentMathKernel` preloads domain-agnostic verification primitives directly into the global namespace:
- **`check_asymptotic_dominance(f, g, var="n")`**: Uses `SymPy` to symbolically compute $\lim_{n \to \infty} f(n)/g(n)$ and deterministically classify whether $f = o(g)$, $f = \mathcal{O}(g)$, $f = \Theta(g)$, or $f = \omega(g)$—eliminating sign errors and log-factor hallucinations in complexity proofs.
- **`check_event_covariance(sampler_fn, n_trials=2000)`**: In probabilistic proofs, LLMs frequently apply $\mathbb{P}(A \cap B) = \mathbb{P}(A)\mathbb{P}(B)$ to events that are subtly correlated after conditioning. This tool runs Monte Carlo trials, computes $\text{Cov}(A, B)$ and Pearson correlation $\rho$, and raises an explicit `DEPENDENCE DETECTED` alert whenever $|\rho| > 0.05$.
- **`find_graph_counterexample(predicate_fn, max_n=7)`**: Exhaustively sweeps a proposed graph invariant across Paths $P_n$, Cycles $C_n$, Complete graphs $K_n$, Stars $S_n$, Complete Bipartite graphs $K_{n_1, n_2}$, and seeded Erdős–Rényi random graphs $G(n, p)$. Crucially, it is **fail-closed**: if `predicate_fn` raises an exception on any graph family, the tool returns `inconclusive: True, verified: False` rather than silently counting a crashed test as passing.

#### 3. Specialized Domain Verification: Topology, SMT Hierarchies, and Quantum Operators
When proving advanced theorems in graph theory, topological connectivity, or quantum information, standard Python REPLs lack primitives for reasoning about planar dual faces, cycle chords, or quantum operator inequalities. `core_engine/math_tools.py` preloads specialized verification tools:
- **Topological Planar Duals & Cycle Chords (`planar_dual_and_faces`, `check_cycle_chords_and_holes`)**: Extracts combinatorial planar faces, tests Jordan curve enclosure, and deterministically identifies chords or chordless holes for planar cuts and graph minor proofs.
- **SMT Parameter Hierarchies & Terminal Routings (`smt_disjoint_paths`, `smt_slice_terminal_pairing`, `smt_check_parameter_hierarchy`)**: Formally encodes disjoint path routings and verifies multi-variable asymptotic scaling orders ($1 \ll k \ll \ell \ll n$) under Z3.
- **Quantum Channels & Loewner Order PSD Matrix Inequalities (`check_psd_inequality`, `check_quantum_channel_claim`, `random_density_matrix`, `uhlmann_fidelity`)**: Proves positive semi-definiteness ($A \succeq B$) via Cholesky decomposition, verifies trace-preservation and complete positivity for CPTP quantum channels, and computes exact Uhlmann fidelity on Haar-random states.
- **NetworkX 3.x Compatibility Shims & Combinatorial Safety Ceilings**: Provides transparent shims for renamed graph APIs (`nx.random_tree`, `nx.vertex_connectivity`) alongside a hard ceiling (`MAX_COMBINATORIAL_STEPS = 200,000`) that prevents exponential combinatorial runaway (such as iterating over $6.87 \times 10^9$ subsets) from exhausting system RAM.

#### 4. Honest Empirical Semantics (`holds_on_samples` vs. Unsound Symbolic Truthiness)
In standard Python/SymPy, calling `bool(x**2 + 1 > 0)` on symbolic expressions raises a `TypeError` for undecidable relations. Some agent frameworks monkeypatch `Relational.__bool__` to return `True` if a few sample points pass—a catastrophic soundness hole that tricks the LLM into believing a finite spot-check is a formal proof.
Proof Harness refuses to patch `Relational.__bool__`. Instead, it provides **`holds_on_samples(rel, max_points=2000)`**, which respects declared SymPy symbol assumptions (`positive=True`, `integer=True`) and enforces strict three-valued epistemic logic:
- **`False`**: A concrete counterexample was found (definitive refutation).
- **`True`**: No counterexample found among the sample grid—reported to the LLM strictly as `holds_on_samples: True` (*empirical survival evidence only, NOT a formal proof*).
- **`None`**: Inconclusive (division by zero, non-real values, or incompatible assumptions).

#### 5. Anti-Brute-Force Analytical Ceilings ("Rule 11 Enforcement")
When left unchecked in a REPL, LLMs often try to approximate asymptotic number-theory or combinatorics problems by running massive brute-force loops (`for n in range(10**8)`), which either times out or produces low-precision statistical noise ($\mathcal{O}(1/\sqrt{N})$).
`PersistentMathKernel` installs **`guarded_range`** and diagnostic ceilings (`MAX_DIAGNOSTIC_SIEVE_LIMIT = 200,000` on `fast_sieve`, `fast_totients`, `fast_spf`, and $n < 7$ on `permutations`): any attempt to allocate a range exceeding $200,000$ elements immediately raises `ExecutionBlockedByRule11`, instructing the model to stop brute-force simulation and derive the exact analytical formula, generating function, or Euler product.

#### 6. Dual-Kernel Isolation (`PersistentMathKernel` vs. Ephemeral `AuditorKernel`)
If the Proof Auditor ran its verification scripts inside the solver's own Python kernel, buggy helper functions or overwritten variables defined by the solver could corrupt the audit. Proof Harness therefore pairs the persistent **`PersistentMathKernel`** (used by the solver across turns) with a completely separate, ephemeral **`AuditorKernel`** (`prover_kernel.py`) that boots a fresh namespace for every audit pass with a hard 15-second timeout (`TimeoutException` subclassing `BaseException` so LLM `try ... except Exception:` blocks cannot swallow timeouts) and thread-local `ThreadAwareStream` proxies that prevent stdout cross-contamination across concurrent workers.

#### 7. Epistemic State Tracking of Hereditary vs. Fragile Properties (`MathStateMachine`)
Embedded directly inside the kernel as `state` (`core_engine/state_machine.py`), the **Epistemic State Machine** tracks the lineage DAG of mathematical objects (`state.register`, `state.derive`, `state.checkpoint`, `state.rollback`):
- **`HEREDITARY_PROPERTIES`** (`bipartite`, `planar`, `forest`, `acyclic`, `measurable`, `positive_semidefinite`, `abelian`, `hausdorff`) are automatically inherited when deriving sub-structures.
- **`FRAGILE_PROPERTIES`** (`independent`, `martingale`, `stationary`, `markov`, `connected`, `regular`, `hamiltonian`, `invertible`, `positive_definite`, `compact`) are **automatically revoked** whenever an object undergoes `conditioning`, `subgraph`, `projection`, `quotient`, or `limit` derivations.
- **The Epistemic Scribe Gate**: If the solver tries to hand off to the Publication Scribe while any derived object has unresolved `revoked_properties`, the kernel blocks the transition and demands an explicit invariance witness (`state.assert_property(name, prop, witness=...)`).
- **The Epistemic Scribe Gate**: If the solver tries to hand off to the Publication Scribe while any derived object has unresolved `revoked_properties`, the kernel blocks the transition and demands an explicit invariance witness (`state.assert_property(name, prop, witness=...)`).

---

## 4. Core Pillar II: The CoALA Continuous Domain Learning Engine (`memory.py` & `remediation.py`)

How do you make a frozen 4-bit 28B model become progressively smarter in a specialized mathematical domain—without gradient updates or weight fine-tuning?

Proof Harness solves this by implementing a **Tripartite CoALA (Cognitive Architectures for Language Agents) Memory & Remediation Engine** backed by a high-speed local vector database.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│        Closed-Loop CoALA Domain Learning Lifecycle (Vector Memory)           │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────┐          ┌─────────────────────────────────┐   │
│   │ 1. Solve Episode        │          │ 2. Post-Solve Remediation Engine│   │
│   │ (Pass or Fail Trace)    │─────────►│ (3-Tier Root-Cause Diagnosis)   │   │
│   └─────────────────────────┘          └────────────────┬────────────────┘   │
│                                                         │                    │
│                                        Extracts Candidate Domain Heuristic   │
│                                                         │                    │
│                                                         ▼                    │
│   ┌──────────────────────────────────────────────────────────────────────┐   │
│   │ 3. Six-Gate Universality & Zero-Leakage Validator                    │   │
│   │    (heuristic_validation.py — PROVENANCE_VERSION = 2)                │   │
│   │  ✓ Blocks empirical cheats (Richardson/Aitken/brute-force sieves)    │   │
│   │  ✓ Scrubs & rejects ground-truth answers & ≥4-digit constants        │   │
│   │  ✓ Strips problem IDs, titles & instance-specific variable bindings  │   │
│   │  ✓ Deduplicates against existing rules (cosine similarity > 0.90)    │   │
│   └─────────────────────────────────────────────────────┬────────────────┘   │
│                                                         │ Validated Rule     │
│                                                         ▼                    │
│   ┌──────────────────────────────────────────────────────────────────────┐   │
│   │ 4. Role-Isolated Vector Embedding Collections                        │   │
│   │  • prime_math_procedural / episodic    (Core Deductive Logician)     │   │
│   │  • tcs_architect_procedural / episodic (Reduction Blueprints)        │   │
│   │  • tcs_verifier_procedural             (Pre-Flight Falsification)    │   │
│   │  • tcs_skeptic_procedural              (Counterexample Attack Rules) │   │
│   │  • tcs_scribe_procedural               (Academic LaTeX Patterns)     │   │
│   └─────────────────────────────────────────────────────┬────────────────┘   │
│                                                         │                    │
│                                 Just-in-Time Cosine Retrieval (score ≥ 0.68) │
│                                                         ▼                    │
│   ┌──────────────────────────────────────────────────────────────────────┐   │
│   │ 5. Next Problem in Domain: Subagents Equipped with Learned Wisdom    │   │
│   └──────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 1. Learning by Solving: How Experience Becomes Embeddings
Every time Proof Harness finishes a problem—whether it achieves a `0.96` victory or suffers a `0.15` rejection—the trajectory does not disappear:
- **Episodic Memory Consolidation (`record_episode`)**: `CoALAMemoryManager` records the problem structure, executed verification code blocks, verification status, and post-mortem critique into `<agent_id>_episodic`, while successful architectural blueprints are persisted directly to `tcs_architect_episodic`.
- **3-Tier Post-Evaluation Remediation (`RemediationEngine` in `remediation.py`)**: Our automated diagnostic engine inspects failed and borderline runs across three tiers:
  - **Tier 1 (Harness & Kernel Code)**: Fixes parser, timeout, or sandbox mechanics.
  - **Tier 2 (Universal Orchestration Invariants)**: Governed by a strict **4-Gate Universality Rubric (`Tier2Rubric`)**—requiring a rule to be 100% domain-agnostic, behavioral, zero-degradation, and impossible to handle via retrieval.
  - **Tier 3 (Domain-Specific Procedural Heuristics $\to$ Vector Memory)**: Any rule that is specific to a mathematical domain (e.g., spectral graph theory, reversible resolution, martingale concentration, or sublinear edit distance) is automatically routed to **Tier 3 Procedural Memory** (`add_procedural_heuristic`), embedded as a dense vector, and indexed for future retrieval.

### 2. Role-Specific Memory Partitioning
Unlike naive RAG systems that dump retrieved text into a single prompt, Proof Harness maintains **separate procedural vector collections for each cognitive role**:
- When the **Context Architect** faces a new tree-decomposition theorem, it queries `tcs_architect_procedural` (`min_score = 0.68`) for structural reduction patterns.
- When the **Adversarial Skeptic** attacks a graph claim, it queries `tcs_skeptic_procedural` for known boundary counterexample templates (e.g., star graphs, disconnected components, or parity traps).
- When the **Publication Scribe** typesets the proof, it queries `tcs_scribe_procedural` for domain-appropriate formal exposition patterns.

### 3. The 6-Gate Anti-Cheating & Zero-Leakage Shield (`heuristic_validation.py`)
A well-known failure mode of self-improving agents is **memory poisoning**: an agent accidentally memorizes a benchmark's expected answer, stores a problem-specific constant, or learns an unsound numerical shortcut (like guessing an infinite limit via finite extrapolation).

Before *any* heuristic can be committed to vector memory, `is_valid_universal_heuristic()` and `find_ground_truth_leak()` enforce six deterministic gates (`PROVENANCE_VERSION = 2`):
1. **Substantive Strategy Gate**: Rejects trivial platitudes (`"review constraints"`, `"check algebra"`) or strings under 25 characters.
2. **Anti-Extrapolation Hack Gate**: Explicitly blocks empirical curve-fitting or sieving shortcuts (`Richardson extrapolation`, `Aitken's delta`, `Shanks transformation`, `brute-force sieve`, `fast_spf`) that masquerade as proofs.
3. **Zero Benchmark Binding**: Rejects any mention of problem numbers (`Problem #850`, `task_2202`), benchmark names, or problem titles.
4. **Zero Instance-Variable Binding**: Blocks hard-coded instance equations (`2^k == 3^m`, `"in this specific problem"`).
5. **Ground-Truth & Constant Scrubbing**: Automatically scans against the problem statement and reference answer, rejecting any heuristic that contains the expected answer or any distinctive $\ge 4$-digit constant from the problem statement.
6. **Near-Duplicate Suppression & Retrieval Self-Exclusion**: Blocks rules with $>0.90$ cosine similarity to existing heuristics (preventing vector clutter) and enforces `exclude_problem_id` / `exclude_task_id` at query time so a task can never retrieve a heuristic derived from itself.

### 4. Why This Makes Proof Harness Ideal for Domain-Specific Industrial & Scientific Solutions
Because domain intelligence lives in **curated, role-partitioned vector embeddings** rather than fine-tuned weights:
- **Zero-Weight-Update Domain Training**: A quantitative trading desk, formal verification team, or cryptography lab can point Proof Harness at 200 domain-specific problems. Over the course of solving and remediating those problems, the harness populates its procedural vector memory with validated, domain-specific mathematical lemmas, counterexample templates, and reduction strategies.
- **Inspectable, Auditable, and Transferable Expertise**: Every learned heuristic in vector memory is human-readable, provenance-tagged, and can be audited, shared across teams, or transferred to a newer base LLM overnight without retraining.

---

## 5. The 7-Role Multi-Agent Orchestration & Closed-Loop Search Governance

Around the **Deterministic Math Kernel** and the **CoALA Continuous Learning Engine**, `MathProver` (`core_engine/prover.py` & `prover_stages.py`) orchestrates **seven specialized cognitive roles**:

```
                               ┌───────────────────────────────────────────┐
                               │      Target Research Problem & Context    │
                               └─────────────────────┬─────────────────────┘
                                                     │
                                                     ▼
                                     ┌───────────────────────────────┐
                                     │   1. Context Architect        │
                                     │ (Lemma Inventory & Blueprint) │
                                     │   [Memory: tcs_architect]     │
                                     └───────────────┬───────────────┘
                                                     │ Seeds state.register()
                                                     ▼
                                     ┌───────────────────────────────┐
                                     │ 2. Pre-Flight Falsifier       │
                                     │ (SymPy / Monte Carlo Sanity)  │
                                     │   [Memory: tcs_verifier]      │
                                     └───────────────┬───────────────┘
                                                     │ Verified Facts & Warnings
                                                     ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ 3. Deductive Logician (_run_exploratory_turns)   [Memory: prime_math]    │
  │    • Turn-1 Exploration Gate (Formal \begin{proof} strictly banned)      │
  │    • Emits structured [CLAIM] / [REDUCTION] & REPL verification scripts  │
  └──────────────────┬───────────────────────────────────────┬───────────────┘
                     │ Proposes [CLAIM]                      │ Ready for Synthesis
                     ▼                                       │ (Epistemic State Gate)
     ┌───────────────────────────────┐                       ▼
     │   4. Adversarial Skeptic      │       ┌───────────────────────────────┐
     │ (Isolated Z3/NetworkX Attack) │       │    5. Publication Scribe      │
     │     [Memory: tcs_skeptic]     │       │ (Critical-Path LaTeX Synth)   │
     └───┬───────────────────────┬───┘       │     [Memory: tcs_scribe]      │
         │ VERIFIED              │ REFUTED   └───────────────┬───────────────┘
         ▼                       ▼                           │
  state.record_lemma()   state.rollback()                    ▼
                                             ┌───────────────────────────────┐
                                             │  6. Internal Proof Auditor    │
                                             │  • Stage 1: AuditorKernel     │
                                             │  • Stage 1.5: Domain Pre-Check│
                                             │  • Stage 2: 6-Gate Referee    │
                                             │  • Stage 2.5: 2nd-Opinion Pass│
                                             └───────┬───────────────┬───────┘
                                                     │               │
                                    Pass (≥ 0.90)    │               │ Reject (< 0.90)
                                    or Max Turns     │               ▼
                                                     │   ┌───────────────────────────────┐
                                                     │   │ 7. Dynamic Critique Falsifier │
                                                     │   │  + Stateless Blackboard       │
                                                     │   │  + CandidatePool / Hard Pivot │
                                                     │   └───────────┬───────────────────┘
                                                     ▼               │ Loops to Turn t+1
                                             ┌───────────────────────┴───────┐
                                             │  External 3-Replicate Judge   │
                                             │   + CoALA Memory Consolidation│
                                             └───────────────────────────────┘
```

### How the 7 Roles Interact with CoALA Memory and the Kernel
1. **Context Architect (`_run_context_architect`)**: Queries `tcs_architect_procedural` memory, catalogs context lemmas, seeds `state.register()` in the kernel, and emits a **Quarantined Reduction Blueprint** (`UNVERIFIED CONJECTURAL SKETCH`).
2. **Pre-Flight Computational Falsifier (`_run_computational_falsifier`)**: Queries `tcs_verifier_procedural` memory and tests the Architect's conjectured bounds in `PersistentMathKernel` before Turn 1 starts.
3. **Deductive Logician (`_run_exploratory_turns`)**: Governed by the **Turn-1 Exploration Gate** (banning `\begin{proof}` on Turn 1), it uses `prime_math_procedural` heuristics and the persistent REPL to test invariants and propose `[CLAIM]` / `[REDUCTION]` steps.
4. **Adversarial Skeptic (`_run_adversarial_skeptic`)**: Runs in a zero-leakage context with `tcs_skeptic_procedural` memory, attacking each `[CLAIM]` in the kernel via `Z3`/`NetworkX`/`SymPy`. On `REFUTED`, the kernel executes `state.rollback('pre_claim')`; on `VERIFIED`, it records the claim to `state.record_lemma`.
5. **Publication Scribe (`_run_publication_scribe`)**: Runs in an isolated 65k-token context with `tcs_scribe_procedural` memory, executing **Backward Critical-Path Pruning** and **Full Domain Coverage** to typeset clean declarative LaTeX.
6. **Multi-Stage Internal Proof Auditor (`_audit_proof_draft`)**: Combines Stage 1 empirical witness testing in the ephemeral `AuditorKernel`, Stage 1.5 regex domain-restriction pre-checks (`_detect_unauthorized_restrictions`), Stage 2 6-gate boolean refereeing (`statement_fidelity_exact`, `inequality_directions_verified`, `theorem_hypotheses_verified`), and Stage 2.5 adversarial confirmation on borderline `[0.75, 0.88]` drafts.
7. **Dynamic Critique Falsifier (`_run_critique_falsifier`) & Search Governance**:
   - **Empirical Critique Stress-Testing**: Tests the Auditor's rejection critique itself in the Python kernel before the next turn.
   - **Stateless 4-Message Blackboard Revision**: Resets revision turns to `[System, Clean Task Prompt, Pinned Proof Draft, Surgical Repair Directive + Modular Lemma Immutability]`.
   - **`CandidatePool` with Noise-Band Occam's Razor ($\epsilon = 0.03$)**: Prefers $>25\%$ more concise proofs within the $\pm 0.03$ auditor noise band, blocks revisions that bloat length by $>50\%$ for a $\le 0.03$ bump, and automatically restores earlier higher-scoring drafts when a later turn degrades.
   - **Golden Balance Hard Pivot & Overtime Gate (Turns 9–12)**: Purges the kernel and logs `BANNED PREMISES` to the Dead Route Ledger when a reduction stagnates below `0.25`, while granting a 4-turn overtime extension when a Turn-8 draft demonstrates directional validity ($\ge 0.30$).
   - **Hybrid Context Compactor (`MathContextCompactor`)**: Combines verbatim deterministic regex extraction (`_extract_deterministic_ledger`) with a 5-part neural summary while pinning anchor messages by identity.

---

## 6. Cognitive Pathologies of Autonomous LLM Provers (And the Harness Scaffolding Remediations)

Evaluating autonomous language models on research-level mathematics reveals six fundamental failure modes that break unassisted LLM provers. Table 5 details these cognitive pathologies alongside real-world benchmark tasks and the corresponding architectural solutions implemented in `core_engine`:

| Failure Mode | Representative Benchmark Case | What Happened in the Trace | Architectural Fix in `core_engine` |
| :--- | :--- | :--- | :--- |
| **1. The Blueprint Authority Trap ("Task 42 Syndrome")** | `task_2108.01161_statement_14_242` | The Context Architect proposed a loose intermediate bound $2\sum_{m\ge 1} e^{-(2m-1)^2 a} \le e^{-a}$. In Turn 1, the REPL showed this intermediate bound fails for small $a$ (large $\beta$). Instead of discarding the Architect's loose inequality (the target theorem is trivially true for $\beta \ge 1$), the Logician trusted the Architect and injected *"under the regime $\beta \le 1$"* into the proof—earning a severe referee penalty for an unauthorized domain restriction. | **Blueprint Quarantine** (`UNVERIFIED CONJECTURAL SKETCH`) + **Deterministic Domain-Restriction Pre-Check** (`_detect_unauthorized_restrictions`) + `statement_fidelity_exact` auto-cap ($\le 0.25$). (Resolved: **0.68 PASS**). |
| **2. Multi-Turn Proof Bloat** | `task_2104.10620_statement_48_191` | Unchecked 10-turn revisions accumulated exploratory helper lemmas and redundant case splits, ballooning proofs from ~4,500 chars to 18,000+ chars and exposing $4\times$ more auxiliary surface area to the referee. | **Noise-Band Occam's Razor** ($\epsilon = 0.03$) in `CandidatePool` + **Critical-Path Pruning** in the Publication Scribe. (Resolved: **0.96 PASS**). |
| **3. Single-Pass Auditor Optimism** | `task_2107.01307_statement_271_450` | A polished LaTeX draft applied Markov's inequality in the wrong direction for non-integer variables. A single-call internal auditor scored the fluent prose `0.78 (VALID)`, exiting early before the external referee caught the reversal. | **6-Gate Boolean Rubric** (`inequality_directions_verified`) + **Stage 2.5 Second-Opinion Confirmation** on borderline `[0.75, 0.88]` drafts. |
| **4. Combinatorial Runaway & Memory Thrashing** | `task_2410.18878_statement_11_63` (Task 234) | An unconstrained graph search attempted to exhaustively iterate over $6.87 \times 10^9$ cycle combinations, locking execution threads and threatening system RAM. | **Combinatorial Safety Ceilings**: Enforced `MAX_COMBINATORIAL_STEPS = 200,000` generator ceiling and `guarded_range < 2,000,000` analytical bounds, forcing mathematical deduction over brute-force loops. (Resolved: Clean completion, **0.25**). |
| **5. Destructive Candidate Eviction (The Drift Trap)** | Multi-Task Cluster (Tasks 236–241) | When an auditor flagged an isolated defect in Turn 6, unconstrained LLM backtracking would wipe viable structural foundations ($v \ge 0.35$), restarting from scratch and oscillating into flat zeros. | **Candidate Protection Gate**: `[Hard Pivot Prevented — Substantive/Best Candidate Protected]`. If structural validity is $\ge 0.25$, destructive candidate wiping is blocked, forcing surgical blackboard revision and preserving progress. |
| **6. Topological & Algebraic Kernel Blindspots** | Graph Cuts & Quantum Tasks (Tasks 71, 236, 293) | Standard Python REPL environments lack first-class primitives for planar dual faces, cycle chords, SMT terminal routing hierarchies, or quantum CPTP channels. | **Domain-Specific Verification Library**: Added `planar_dual_and_faces`, `check_cycle_chords_and_holes`, `smt_disjoint_paths`, `smt_slice_terminal_pairing`, `check_psd_inequality`, `uhlmann_fidelity`, and NetworkX 3.x aliases (`nx.random_tree`, `nx.vertex_connectivity`). |

---

## 7. Empirical Results: The 300-Problem PhD-Level TCS-Bench Evaluation

To evaluate the full system, we ran the continuous autonomous pipeline (`run_tcs_continuous_pipeline.py`) across all **300 research problems from TCS-Bench**, evaluating whether an agent harness can multiply the performance of passing grades counting C and up versus single-shot inference.

Each completed proof is immediately graded by the external **Calibrated TCS Judge** (`core_engine/judging/tcs_judge.py`), which runs independent referee replicates (with adaptive early stopping outside the `[0.30, 0.70]` borderline band and multi-replicate median aggregation inside the borderline band) across six strict criteria: *Statement Fidelity*, *Quantifier & Edge-Case Completeness*, *Step-by-Step Logical Soundness*, *Invocation of External Theorems*, *Absence of Hand-Waving*, and *Structural Completeness*.

### The Final Scorecard: Cumulative Passing Thresholds

```
================================================================================
     TCS-Bench 300 Empirical Scorecard: Cumulative Passing Thresholds
================================================================================
  Total Problems Evaluated:                     300
  Total Scored (0 Parse / 0 Judge Errors):      300 (100.0%)

  CUMULATIVE PASSING THRESHOLDS & MULTIPLIERS OVER FIRST-SHOT:
  • All Passing Grades (≥ 0.40, Grade C & Above): 71 / 300  (23.67%)  [5.92× vs. Pass@1]
  • Certified Sound Passes (≥ 0.50, Grade B & Above): 63 / 300 (21.00%) [6.30× vs. Pass@1]
  • Publication-Grade Proofs (≥ 0.70, Grade A & Above): 54 / 300 (18.00%) [7.71× vs. Pass@1]
    - High-Rigor Certified Sound (≥ 0.80):      50 / 300  (16.67%)  [8.33× vs. Pass@1]
    - Journal-Ready Flawless Proofs (≥ 0.90):   31 / 300  (10.33%)  [7.75× vs. Pass@1]
  • Incomplete / Refuted Proofs (< 0.40):       229 / 300 (76.33%)
--------------------------------------------------------------------------------
  HEADLINE VALIDATION ON PASSING PROOFS:
  • Mean Referee Validity:                      0.853 ± 0.129  (Median: 0.880)
  • Structural Completeness:                    0.907 ± 0.089  (Median: 0.940)
  • Logical Deductive Rigor:                    0.857 ± 0.121  (Median: 0.900)
  • Boundary & Edge-Case Completeness:          0.878 ± 0.127  (Median: 0.900)
================================================================================
```

When we dissect *which* iteration produced each of the **63 certified passing proofs**, the vital role of multi-turn symbolic verification becomes undeniable:

| Turn Iteration | Operational Harness Phase | Passes ($\ge 0.50$) | % of Total Passes | Cumulative Pass Rate |
| :---: | :--- | :---: | :---: | :---: |
| **Turn 1 (Pass@1 Baseline Contrast)** | Exploration Gate (Drafts Blocked) | **0** | **0.0%** | **0.00%** |
| **Turn 2** | Post-Exploration First Synthesis | 5 | 7.9% | 1.67% |
| **Turn 3** | First Blackboard Surgical Repair | 5 | 7.9% | 3.33% |
| **Turns 4–5** | Mid-Horizon Invariant & SMT Pruning | 9 | 14.3% | 6.33% |
| **Turns 6–7** | Deep Counterexample Elimination | 9 | 14.3% | 9.33% |
| **Turn 8** | **Primary Publication Scribe Horizon** | **22** | **34.9%** | **16.67%** |
| **Turns 9–12 (Overtime)** | **Overtime Surgical Repair Synthesis** | **13** | **20.6%** | **21.00%** |
| **Total** | **Full Governed Trajectory ($T \le 12$)** | **63** | **100.0%** | **21.00%** |

### The Insights Behind the Numbers

1. **Shattering the 2×–3× Bar (Nearly 6×–8× Gains)**: Our target was to evaluate whether an agent harness could double or triple performance over initial unassisted generation. In reality, Proof Harness achieved a **$5.92\times$ multiplier on Grade C and above passing grades ($23.67\%$ vs. $4.00\%$)**, a **$6.30\times$ multiplier on Grade B and above certified sound passes ($21.00\%$ vs. $3.33\%$)**, and a staggering **$7.71\times$ multiplier on Grade A and above publication-grade proofs ($18.00\%$ vs. $2.33\%$)**.
2. **Initial Draft Guesses vs. 0.853 Passing Validity**: Unassisted initial drafting yields superficial proofs that fail boundary cases and invert inequalities (mean validity 0.142 across all 300 individual first-pass judgments). Proof Harness drives passing proofs to a stellar **0.853 mean validity (0.880 median)**, with **0.907 structural completeness** and **0.857 logical rigor**.
3. **The Turn-1 Exploration Gate Pays Off**: Unlike naive prompting where models immediately rush into flawed Turn-1 drafts, Proof-Harness enforces invariant discovery and counterexample sweeps before drafting. Zero passes occur on Turn 1 by design—yet by Turn 3, **10 sound proofs (3.33%)** are already secured.
4. **Deep Multi-Turn Iteration Drives 100% of Wins**: Every single passing proof in Proof Harness required multi-turn refinement, with **$84.1\%$ of passing proofs (53 of 63)** won in Turn 4 or later.
5. **Overtime Extensions Deliver 20.6% of Passes**: When Turn-8 drafts exhibited strong structural viability, granting 4 overtime turns (Turns 9–12) captured **13 additional certified passes**, lifting the benchmark from $16.67\%$ to $21.00\%$.

---

## 8. Case Studies from the Benchmark

### Case Study A: An Autonomous 0.94 Victory on Output-Sensitive High-Dimensional Sumsets
Consider **Task 242** (`task_2410.21942_statement_30_71`), a research theorem on computing prefix-restricted sumsets for high-dimensional Las Vegas algorithms:

> **The Target**: Compute $C = (A + B) \cap [0 \dots t]^d$ for sets $A, B \subseteq [0 \dots t]^d$ in output-sensitive time:
> $$\mathcal{O}\left(\left(n + m + (nm)^{\frac{d}{d+1}} c^{\frac{1}{d+1}}\right) (\log(nm))^{\mathcal{O}(d)}\right)$$

1. **Turn 1 (Exploration Gate)**: Blocked from rushing a proof, the Logician explored coordinate-wise divide-and-conquer and identified linear-term blowup across recursive subproblems.
2. **Turns 2–4 (REPL & Skeptic Intervention)**: In `PersistentMathKernel`, small coordinate partitions ($d=2, 3$) revealed that coordinate-wise bounding fails unless dyadic scaling is applied to the bounding box $T$.
3. **Turns 5–7 (Structural Realization)**: The model pivoted to a $d$-dimensional dyadic box decomposition on $T$, defining shifted boxes $A_\alpha^*$ and bounding the non-vacuous subproblems using **Hölder's inequality** with conjugate exponents $p = \frac{d}{d+1}$ and $q = \frac{1}{d+1}$.
4. **Turn 8 (Publication Scribe Synthesis)**: The Scribe assembled a crisp, 7,275-character inductive proof over $T$, bounding the sum of recursive terms via Hölder's inequality over the $2^d$ sub-boxes.
5. **External Referee Verdict**: **Validity: 0.94** (*Rigor: 0.92, Completeness: 0.95, Confidence: 0.93*).

### Case Study B: Flipping a Borderline Failure into a 0.82 Pass (`task_2202.08066_statement_67_239`)
On **Task 69** (hierarchical tree edit-distance alignment reductions), an earlier harness iteration without invariant-first gating and domain-coverage checks stalled at a borderline **0.50 (`verdict: "0"`)** because it left tree-node substring length preconditions ($|Y_v| = |X_v| + 2K$) implicit and hand-waved capped-vs-uncapped edit-distance steps.
With `core_engine`'s **Turn-1 Exploration Gate**, **Critical-Path Scribe Pruning**, and **Stateless Blackboard Revision**, the prover explicitly discharged the inductive alignment invariants across all internal and leaf nodes—jumping **Rigor from 0.45 to 0.86**, **Completeness from 0.60 to 0.90**, **Boundary Cases from 0.40 to 0.82**, and overall **Validity from 0.50 to 0.82 (`PASS`)**.

---

## 9. Takeaways for AI Engineers Building Domain-Specific Reasoning Harnesses

If you are designing agent harnesses for reasoning-heavy vertical domains—whether graduate-level mathematics, formal verification, quantitative finance, compiler synthesis, or physics—eight core principles emerge from `core_engine`:

1. **Train the Harness via Embeddings, Not Weight Fine-Tuning (CoALA Continual Learning)**: Don't accept episodic amnesia from frozen LLM weights, and don't pay the catastrophic-forgetting cost of continual LoRA fine-tuning. Use a **CoALA Heuristics Engine** backed by role-isolated vector collections and a strict universality/anti-leakage gate (`heuristic_validation.py`) so your harness *learns by solving*—distilling every solved task and every falsified failure into reusable domain knowledge.
2. **Never Trust Probabilistic Tokens Over Deterministic Execution**: Treat every mathematical assertion from an LLM as an *untrusted conjecture*. Pair the model with a **Zero-Trust Deterministic Kernel** (`SymPy`, `Z3`, 50-digit `mpmath`, `NetworkX`), preloaded diagnostic operators (`check_asymptotic_dominance`, `check_event_covariance`, `find_graph_counterexample`), and honest three-valued empirical semantics (`holds_on_samples`).
3. **Block Brute-Force Hacks to Force Structural Deduction**: Unconstrained code interpreters let LLMs cheat via massive sieves or numerical sequence extrapolation. Enforce hard analytical ceilings (`guarded_range` $\le 200,000$, `permutations` $n < 7$) and block extrapolation hacks in both the kernel and the memory validator.
4. **Track Epistemic Fragility Explicitly in the Kernel**: When mathematical objects undergo conditioning, subgraph restriction, or projection, use a formal state machine (`MathStateMachine`) to automatically revoke fragile properties (`independent`, `connected`, `invertible`) until re-verified by a witness.
5. **Enforce a Turn-1 Exploration Gate**: Never let a reasoning agent write its final deliverable on Turn 1. Force it to query its CoALA procedural memory, formulate testable `[CLAIM]` invariants, and write verification code in the kernel first.
6. **Quarantine Architectural Blueprints**: Planning subagents are great at high-level decomposition and terrible at tight algebraic constants. Treat planner blueprints as *unverified sketches*, and forbid the solver from restricting a problem's domain when a planner's loose intermediate bound fails.
7. **Separate Discovery from Typesetting (And Prune the Critical Path)**: Use an isolated **Publication Scribe** (with its own `tcs_scribe` memory) that traces backwards from the conclusion and omits dead exploratory side-lemmas.
8. **Repair on a Stateless Blackboard with Noise-Band Occam's Razor**: Collapse revisions to a 4-message blackboard, audit drafts in an ephemeral **AuditorKernel**, and penalize revisions that bloat length by $>50\%$ for a marginal score bump within auditor noise ($\pm 0.03$).

---

## 10. Academic References

- **[Stellar Colosseum (`arXiv:2609.15983`)](https://arxiv.org/abs/2609.15983)** *(Lin, Woodruff, Deng, Mao, Zuo, & Mirrokni, Google Research & CMU, 2026)*: A many-agent inference harness for long-horizon research in TCS and mathematics, investigating multi-agent inference-time scaling on TCS-Bench with Gemini frontier models.
- **[Cognitive Architectures for Language Agents (CoALA)](https://arxiv.org/abs/2309.02427)** *(Sumers et al., Princeton NLP, TMLR 2023)*: The tripartite cognitive memory taxonomy (Working, Procedural, and Episodic Memory) that Proof Harness extends with 10 role-isolated vector embedding collections, a 3-Tier `RemediationEngine`, and a 6-gate Zero-Leakage Universality Validator (`PROVENANCE_VERSION = 2`).
- **[TCS-Bench (`arXiv:2608.09538`)](https://arxiv.org/abs/2608.09538)**: The 300-problem graduate-level Theoretical Computer Science benchmark suite used to evaluate single-shot versus multi-turn neuro-symbolic proof synthesis.

*All benchmarks, CoALA memory modules, deterministic kernel engines, and evaluation scripts are open-sourced in the [Proof Harness repository](https://github.com/dionyziuz/proof_harness).*

