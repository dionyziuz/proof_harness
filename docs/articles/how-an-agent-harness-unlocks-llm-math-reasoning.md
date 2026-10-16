# How an Agent Harness Unlocks Mathematical Superpowers in a 4-Bit 28B Model

### *The story of how architectural scaffolding, symbolic execution, and cognitive partitioning turned a lightweight 4-bit open-weight LLM into a graduate-level theorem prover.*

---

## 1. The Myth of the Monolithic Mathematician

In the modern AI landscape, conventional wisdom dictates that solving frontier, graduate-level mathematics requires massive frontier models: multi-hundred-billion-parameter clusters consuming kilowatts of data center power. When researchers benchmark smaller open-weight models on hard mathematical domains—such as research-level Theoretical Computer Science (TCS)—the results from standard **Pass@1** evaluations are predictably bleak.

Prompted in a single shot, even capable open-weight models often hallucinate algebraic leaps, assert false lemmas with supreme confidence, or fail completely when tracking complex invariant inheritance.

We set out to challenge this assumption with an alternative paradigm:

> **Core Thesis:** *Mathematical reasoning is not purely a function of raw parameter scale. It is a function of cognitive architecture, **continuous domain learning via CoALA vector heuristics**, and **zero-trust deterministic testing inside a symbolic execution kernel**.*

To test this thesis, we evaluated **unsloth/Qwen3.8-27B-NVFP4**—a 27B/28B parameter model running in **4-bit quantization** on consumer hardware (a single **NVIDIA GeForce RTX 5090** with 32GB VRAM)—across **300 graduate-level research problems from TCS-Bench**, drawn directly from recent theoretical computer science literature on arXiv.

The results demonstrated the transformative power of an agent harness:

```
┌────────────────────────────────────────────────────────────────────────┐
│               TCS-Bench 300 Empirical Progression                      │
├────────────────────────────────────────────────────────────────────────┤
│  Evaluation Paradigm          │ Passes (≥ 0.50) │ Pass Rate            │
├───────────────────────────────┼─────────────────┼──────────────────────┤
│  1. Single-Shot (Pass@1)      │  10 / 300       │   3.33%              │
│  2. Lightweight (Pass@3)      │  21 / 300       │   7.00% (2.1x)       │
│  3. Full Multi-Turn Harness   │  63 / 300       │  21.00% (6.3x Gain!) │
└───────────────────────────────┴─────────────────┴──────────────────────┘
```

By wrapping a 4-bit 28B model in **Prime Math**—a dedicated, closed-loop agentic theorem proving harness—we unlocked a **6.3× performance multiplier**, lifting certified sound proofs from **3.3% to 21.0%** (with **44 proofs scoring ≥ 0.70 publication-grade**).

At the heart of this leap are **two foundational differentiators** that set Prime Math apart from standard LLM wrappers:
1. **The CoALA Heuristics & Continuous Domain Learning Engine**: A self-improving procedural and episodic vector memory system (backed by **TurboVec**) where the machine *learns by solving*—allowing practitioners to train the harness in specialized mathematical or engineering domains purely through embeddings, without ever touching model weights.
2. **The Zero-Trust Deterministic Mathematical Kernel**: A dual-runtime symbolic, SMT, and arbitrary-precision execution engine that refuses to trust the LLM's probabilistic intuition—testing every bound, graph invariant, covariance assumption, and asymptotic claim deterministically in code.

Here is the story of the twin blind spots that break standalone LLMs, and how these two engines—combined with multi-agent cognitive partitioning—turn a compact quantized model into a self-improving mathematician.

---

## 2. The Problem Statement: Why Standalone LLMs Fail on Frontier & Domain-Specific Math

When you deploy an off-the-shelf LLM on graduate-level mathematics or specialized industrial domains (such as quantitative finance, cryptographic protocol verification, or algorithmic complexity), the system hits **two fundamental structural walls**, followed by four tactical failure modes:

### Structural Wall #1: Frozen Weights & Episodic Amnesia (No Continuous Domain Learning)
Standard LLMs are **stateless at inference time**. An agent might spend 45 minutes and 50,000 tokens discovering a brilliant domain-specific reduction—such as how to bound hypergraph conductance across vertex separators or how to apply Hölder's inequality over shifted dyadic boxes—only to suffer **total amnesia** the moment the session ends.
- On the very next problem in the same domain, the model starts from scratch and repeats the exact same mistakes.
- For engineering teams working in specialized verticals, fine-tuning or running RLHF on a 28B–70B model after every batch of problems is prohibitively expensive, slow, and prone to catastrophic forgetting. **Without a mechanism to continuously extract, validate, and embed reusable mathematical heuristics from its own solving trajectories, an agent cannot compound its domain expertise over time.**

### Structural Wall #2: Probabilistic Token Prediction $\neq$ Mathematical Truth (The Trust Fallacy)
Autoregressive transformers generate tokens by maximizing conditional probability—not by verifying mathematical truth. When an LLM relies solely on its internal weights to do math:
- It cannot reliably distinguish a true algebraic identity from a fluent hallucination.
- It cannot deterministically check whether an inequality $f(n) \le g(n)$ flips direction at boundary values $n=1, 2$, whether conditioning on a sum event $E$ destroys pairwise independence (`Cov(A, B) != 0`), or whether a graph lemma fails on $K_{3,3}$.
- **Trusting an LLM to self-verify mathematics purely through prose is a category error.** Without a deterministic computational kernel that forces conjectures through exact symbolic algebra, SMT solvers, and exhaustive small-instance counterexample sweeps, a single unverified leap cascades until the entire proof collapses.

### Four Tactical Failure Modes of Ungoverned Inference
Compounding those two structural walls are four tactical breakdowns in naive prompting:
1. **Premature Convergence (The Turn-1 Trap)**: In single-shot prompting, the LLM commits to an initial algebraic path within its first few hundred tokens. When that path breaks, it papers over the flaw with hand-waving prose rather than backtracking.
2. **Context Poisoning via Scratchpad Noise**: In naive multi-turn loops, exploratory dead ends, stack traces, and formal drafting share a single growing chat transcript, polluting self-attention by Turn 5.
3. **Silent Invariant Violation**: Reductions routinely transform objects (conditioning random variables, taking induced subgraphs, projecting matrices). LLMs blindly assume fragile parent properties (like independence, connectivity, or invertibility) survive transformations that actually destroy them.
4. **Blueprint Dogmatism**: When an initial planning step proposes a loose intermediate bound that fails on a sub-regime, naive agents doubt the *target theorem* and inject unauthorized domain restrictions instead of replacing the intermediate bound.

---

## 3. Core Pillar I: The CoALA Continuous Domain Learning Engine (`memory.py` & `remediation.py`)

How do you make a frozen 4-bit 28B model become progressively smarter in a specialized mathematical domain—without gradient updates or weight fine-tuning?

Prime Math solves this by implementing a **Tripartite CoALA (Cognitive Architectures for Language Agents) Memory & Remediation Engine** backed by **TurboVec**, a high-speed local vector database (`127.0.0.1:18792`).

```
┌──────────────────────────────────────────────────────────────────────────────┐
│          Closed-Loop CoALA Domain Learning Lifecycle (TurboVec)              │
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
│   │ 4. Role-Isolated TurboVec Embedding Collections                      │   │
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
Every time Prime Math finishes a problem—whether it achieves a `0.96` victory or suffers a `0.15` rejection—the trajectory does not disappear:
- **Episodic Memory Consolidation (`record_episode`)**: `CoALAMemoryManager` records the problem structure, executed verification code blocks, verification status, and post-mortem critique into `<agent_id>_episodic`, while successful architectural blueprints are persisted directly to `tcs_architect_episodic`.
- **3-Tier Post-Evaluation Remediation (`RemediationEngine` in `remediation.py`)**: Our automated diagnostic engine inspects failed and borderline runs across three tiers:
  - **Tier 1 (Harness & Kernel Code)**: Fixes parser, timeout, or sandbox mechanics.
  - **Tier 2 (Universal Orchestration Invariants)**: Governed by a strict **4-Gate Universality Rubric (`Tier2Rubric`)**—requiring a rule to be 100% domain-agnostic, behavioral, zero-degradation, and impossible to handle via retrieval.
  - **Tier 3 (Domain-Specific Procedural Heuristics $\to$ TurboVec)**: Any rule that is specific to a mathematical domain (e.g., spectral graph theory, reversible resolution, martingale concentration, or sublinear edit distance) is automatically routed to **Tier 3 Procedural Memory** (`add_procedural_heuristic`), embedded as a dense vector in TurboVec, and indexed for future retrieval.

### 2. Role-Specific Memory Partitioning
Unlike naive RAG systems that dump retrieved text into a single prompt, Prime Math maintains **separate procedural vector collections for each cognitive role**:
- When the **Context Architect** faces a new tree-decomposition theorem, it queries `tcs_architect_procedural` (`min_score = 0.68`) for structural reduction patterns.
- When the **Adversarial Skeptic** attacks a graph claim, it queries `tcs_skeptic_procedural` for known boundary counterexample templates (e.g., star graphs, disconnected components, or parity traps).
- When the **Publication Scribe** typesets the proof, it queries `tcs_scribe_procedural` for domain-appropriate formal exposition patterns.

### 3. The 6-Gate Anti-Cheating & Zero-Leakage Shield (`heuristic_validation.py`)
A well-known failure mode of self-improving agents is **memory poisoning**: an agent accidentally memorizes a benchmark's expected answer, stores a problem-specific constant, or learns an unsound numerical shortcut (like guessing an infinite limit via finite extrapolation).

Before *any* heuristic can be committed to TurboVec, `is_valid_universal_heuristic()` and `find_ground_truth_leak()` enforce six deterministic gates (`PROVENANCE_VERSION = 2`):
1. **Substantive Strategy Gate**: Rejects trivial platitudes (`"review constraints"`, `"check algebra"`) or strings under 25 characters.
2. **Anti-Extrapolation Hack Gate**: Explicitly blocks empirical curve-fitting or sieving shortcuts (`Richardson extrapolation`, `Aitken's delta`, `Shanks transformation`, `brute-force sieve`, `fast_spf`) that masquerade as proofs.
3. **Zero Benchmark Binding**: Rejects any mention of problem numbers (`Problem #850`, `task_2202`), benchmark names, or problem titles.
4. **Zero Instance-Variable Binding**: Blocks hard-coded instance equations (`2^k == 3^m`, `"in this specific problem"`).
5. **Ground-Truth & Constant Scrubbing**: Automatically scans against the problem statement and reference answer, rejecting any heuristic that contains the expected answer or any distinctive $\ge 4$-digit constant from the problem statement.
6. **Near-Duplicate Suppression & Retrieval Self-Exclusion**: Blocks rules with $>0.90$ cosine similarity to existing heuristics (preventing vector clutter) and enforces `exclude_problem_id` / `exclude_task_id` at query time so a task can never retrieve a heuristic derived from itself.

### 4. Why This Makes Prime Math Ideal for Domain-Specific Industrial & Scientific Solutions
Because domain intelligence lives in **curated, role-partitioned vector embeddings** rather than fine-tuned weights:
- **Zero-Weight-Update Domain Training**: A quantitative trading desk, formal verification team, or cryptography lab can point Prime Math at 200 domain-specific problems. Over the course of solving and remediating those problems, the harness populates its TurboVec procedural indices with validated, domain-specific mathematical lemmas, counterexample templates, and reduction strategies.
- **Inspectable, Auditable, and Transferable Expertise**: Every learned heuristic in TurboVec is human-readable, provenance-tagged, and can be audited, shared across teams, or transferred to a newer base LLM overnight without retraining.

---

## 4. Core Pillar II: The Zero-Trust Deterministic Mathematical Kernel (`kernel.py`, `prover_kernel.py` & `math_tools.py`)

If the CoALA Heuristics Engine is the harness's **long-term memory**, the **Deterministic Mathematical Kernel** is its **scientific laboratory**.

Instead of trusting the 28B LLM to evaluate an integral, check an asymptotic limit, or verify whether two events are independent in its head, Prime Math forces mathematical claims through a stateful, airgapped Python execution engine (`TCSCompliantKernel` and `AuditorKernel`).

```python
class TCSCompliantKernel(_NetworkIsolationMixin, PersistentMathKernel):
    """3-layer airgapped symbolic/numerical REPL with deterministic math verification tools."""

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
    #    - holds_on_samples(rel, max_points=2000) -> honest empirical falsification
    #    - state (MathStateMachine) -> tracks hereditary vs. fragile properties

    def execute(self, code: str, timeout_seconds: Optional[float] = None) -> ExecutionResult:
        # Layer 1: AST Network & Escape-Hatch Denylist (blocks http, socket, urllib,
        #          subprocess, importlib, ctypes, os.system/fork, sys.modules, __import__)
        # Layer 2: Combinatorial Ceiling (blocks permutations(range(n)) for n >= 7)
        blocked = self._forbidden_import(code.strip()) or self._combinatorial_ceiling(code.strip())
        if blocked:
            return ExecutionResult(success=False, error=blocked, ...)

        # Layer 3: Thread-local runtime socket guard + _detect_trace_tampering
        return super().execute(code, timeout_seconds or self.default_timeout)
```

### Six Structural Advantages of Deterministic Kernel Grounding

#### 1. Instantaneous Counterexample Falsification (`Z3`, `SymPy`, `NetworkX`)
To prove a universal theorem $\forall x \in D, \Phi(x) \implies \Psi(x)$, finding **one** concrete counterexample $x_0 \in D$ where $\Phi(x_0) \land \neg \Psi(x_0)$ holds is 100% definitive. When the LLM proposes an intermediate inequality or graph reduction, the kernel tests it in milliseconds—killing false conjectures on Turn 1 or Turn 2 before the model wastes 6 turns writing a 15,000-character proof around a lie.

#### 2. Built-In Domain Diagnostic Primitives (`core_engine/math_tools.py`)
Rather than hoping the LLM writes bug-free test scripts from scratch every time, `TCSCompliantKernel` preloads three domain-agnostic verification primitives directly into the global namespace:
- **`check_asymptotic_dominance(f, g, var="n")`**: Uses `SymPy` to symbolically compute $\lim_{n \to \infty} f(n)/g(n)$ and deterministically classify whether $f = o(g)$, $f = \mathcal{O}(g)$, $f = \Theta(g)$, or $f = \omega(g)$—eliminating sign errors and log-factor hallucinations in complexity proofs.
- **`check_event_covariance(sampler_fn, n_trials=2000)`**: In probabilistic proofs, LLMs frequently apply $\mathbb{P}(A \cap B) = \mathbb{P}(A)\mathbb{P}(B)$ to events that are subtly correlated after conditioning. This tool runs Monte Carlo trials, computes $\text{Cov}(A, B)$ and Pearson correlation $\rho$, and raises an explicit `DEPENDENCE DETECTED` alert whenever $|\rho| > 0.05$.
- **`find_graph_counterexample(predicate_fn, max_n=7)`**: Exhaustively sweeps a proposed graph invariant across Paths $P_n$, Cycles $C_n$, Complete graphs $K_n$, Stars $S_n$, Complete Bipartite graphs $K_{n_1, n_2}$, and seeded Erdős–Rényi random graphs $G(n, p)$. Crucially, it is **fail-closed**: if `predicate_fn` raises an exception on any graph family, the tool returns `inconclusive: True, verified: False` rather than silently counting a crashed test as passing.

#### 3. Honest Empirical Semantics (`holds_on_samples` vs. Unsound Symbolic Truthiness)
In standard Python/SymPy, calling `bool(x**2 + 1 > 0)` on symbolic expressions raises a `TypeError` for undecidable relations. Some agent frameworks monkeypatch `Relational.__bool__` to return `True` if a few sample points pass—a catastrophic soundness hole that tricks the LLM into believing a finite spot-check is a formal proof.
Prime Math refuses to patch `Relational.__bool__`. Instead, it provides **`holds_on_samples(rel, max_points=2000)`**, which respects declared SymPy symbol assumptions (`positive=True`, `integer=True`) and enforces strict three-valued epistemic logic:
- **`False`**: A concrete counterexample was found (definitive refutation).
- **`True`**: No counterexample found among the sample grid—reported to the LLM strictly as `holds_on_samples: True` (*empirical survival evidence only, NOT a formal proof*).
- **`None`**: Inconclusive (division by zero, non-real values, or incompatible assumptions).

#### 4. Anti-Brute-Force Analytical Ceilings ("Rule 11 Enforcement")
When left unchecked in a REPL, LLMs often try to approximate asymptotic number-theory or combinatorics problems by running massive brute-force loops (`for n in range(10**8)`), which Either times out or produces low-precision statistical noise ($\mathcal{O}(1/\sqrt{N})$).
`PersistentMathKernel` installs **`guarded_range`** and diagnostic ceilings (`MAX_DIAGNOSTIC_SIEVE_LIMIT = 200,000` on `fast_sieve`, `fast_totients`, `fast_spf`, and $n < 7$ on `permutations`): any attempt to allocate a range exceeding $200,000$ elements immediately raises `ExecutionBlockedByRule11`, instructing the model to stop brute-force simulation and derive the exact analytical formula, generating function, or Euler product.

#### 5. Dual-Kernel Isolation (`TCSCompliantKernel` vs. Ephemeral `AuditorKernel`)
If the Proof Auditor ran its verification scripts inside the solver's own Python kernel, buggy helper functions or overwritten variables defined by the solver could corrupt the audit. Prime Math therefore pairs the persistent **`TCSCompliantKernel`** (used by the solver across turns) with a completely separate, ephemeral **`AuditorKernel`** (`prover_kernel.py`) that boots a fresh namespace for every audit pass with a hard 15-second timeout (`TimeoutException` subclassing `BaseException` so LLM `try ... except Exception:` blocks cannot swallow timeouts) and thread-local `ThreadAwareStream` proxies that prevent stdout cross-contamination across concurrent workers.

#### 6. Epistemic State Tracking of Hereditary vs. Fragile Properties (`MathStateMachine`)
Embedded directly inside the kernel as `state` (`core_engine/state_machine.py`), the **Epistemic State Machine** tracks the lineage DAG of mathematical objects (`state.register`, `state.derive`, `state.checkpoint`, `state.rollback`):
- **`HEREDITARY_PROPERTIES`** (`bipartite`, `planar`, `forest`, `acyclic`, `measurable`, `positive_semidefinite`, `abelian`, `hausdorff`) are automatically inherited when deriving sub-structures.
- **`FRAGILE_PROPERTIES`** (`independent`, `martingale`, `stationary`, `markov`, `connected`, `regular`, `hamiltonian`, `invertible`, `positive_definite`, `compact`) are **automatically revoked** whenever an object undergoes `conditioning`, `subgraph`, `projection`, `quotient`, or `limit` derivations.
- **The Epistemic Scribe Gate**: If the solver tries to hand off to the Publication Scribe while any derived object has unresolved `revoked_properties`, the kernel blocks the transition and demands an explicit invariance witness (`state.assert_property(name, prop, witness=...)`).

---

## 5. The 7-Role Multi-Agent Orchestration & Closed-Loop Search Governance

Around the **CoALA Continuous Learning Engine** and the **Deterministic Math Kernel**, `MathProver` (`core_engine/prover.py` & `prover_stages.py`) orchestrates **seven specialized cognitive roles**:

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
2. **Pre-Flight Computational Falsifier (`_run_computational_falsifier`)**: Queries `tcs_verifier_procedural` memory and tests the Architect's conjectured bounds in `TCSCompliantKernel` before Turn 1 starts.
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

## 6. Hard-Won Harness Engineering Lessons: What Broke Between V3 and V4

Building an autonomous theorem-proving harness is full of counterintuitive traps. During our engineering progression from V3 to V4, forensic trace analysis uncovered three classic failure modes—and fixing them taught us as much as our wins:

| Failure Mode | Representative Benchmark Case | What Happened in the Trace | Architectural Fix in `core_engine` |
| :--- | :--- | :--- | :--- |
| **1. The Blueprint Authority Trap ("Task 42 Syndrome")** | `task_2108.01161_statement_14_242` | The Context Architect proposed a loose intermediate bound $2\sum_{m\ge 1} e^{-(2m-1)^2 a} \le e^{-a}$. In Turn 1, the REPL showed this intermediate bound fails for small $a$ (large $\beta$). Instead of discarding the Architect's loose inequality (the target theorem is trivially true for $\beta \ge 1$), the Logician trusted the Architect and injected *"under the regime $\beta \le 1$"* into the proof—earning a severe referee penalty for an unauthorized domain restriction. | **Blueprint Quarantine** (`UNVERIFIED CONJECTURAL SKETCH`) + **Deterministic Domain-Restriction Pre-Check** (`_detect_unauthorized_restrictions`) + `statement_fidelity_exact` auto-cap ($\le 0.25$). |
| **2. Multi-Turn Proof Bloat** | `task_2104.10620_statement_48_191` | Unchecked 10-turn revisions accumulated exploratory helper lemmas and redundant case splits, ballooning proofs from ~4,500 chars to 18,000+ chars and exposing $4\times$ more auxiliary surface area to the referee. | **Noise-Band Occam's Razor** ($\epsilon = 0.03$) in `CandidatePool` + **Critical-Path Pruning** in the Publication Scribe. |
| **3. Single-Pass Auditor Optimism** | `task_2107.01307_statement_271_450` | A polished LaTeX draft applied Markov's inequality in the wrong direction for non-integer variables. A single-call internal auditor scored the fluent prose `0.78 (VALID)`, exiting early before the external referee caught the reversal. | **6-Gate Boolean Rubric** (`inequality_directions_verified`) + **Stage 2.5 Second-Opinion Confirmation** on borderline `[0.75, 0.88]` drafts. |

---

## 7. Empirical Results: The 300-Problem TCS-Bench Evaluation

To evaluate the full system, we ran the continuous autonomous pipeline (`run_tcs_continuous_pipeline.py`) across **300 research problems from TCS-Bench**.

Each completed proof is immediately graded by the external **Calibrated TCS Judge** (`core_engine/judging/tcs_judge.py`), which runs independent referee replicates (with adaptive early stopping outside the `[0.30, 0.70]` borderline band and multi-replicate median aggregation inside the borderline band) across six strict criteria: *Statement Fidelity*, *Quantifier & Edge-Case Completeness*, *Step-by-Step Logical Soundness*, *Invocation of External Theorems*, *Absence of Hand-Waving*, and *Structural Completeness*.

### The Scorecard: Where the Proofs Were Won

```
================================================================================
                    TCS-Bench 300 Final Empirical Scorecard
================================================================================
  Total Problems Evaluated:               300
  Certified Sound Passes (≥ 0.50):        63 / 300 (21.00%)
  Publication-Grade Proofs (≥ 0.70):      44 / 300 (14.67%)
  Directionally Sound Reductions (≥ 0.40): 77 / 300 (25.67%)
================================================================================
```

When we dissect *which* iteration produced each passing proof, the vital role of multi-turn symbolic verification becomes undeniable:

| Turn Iteration | Passes ($\ge 0.50$) | % of Total Passes | Cumulative Pass Rate |
| :---: | :---: | :---: | :---: |
| **Turn 1 (Pass@1)** | **10** | **15.9%** | **3.33%** |
| **Turn 2** | 7 | 11.1% | 5.67% |
| **Turn 3** | 4 | 6.3% | **7.00%** |
| **Turns 4–7** | 17 | 27.0% | 12.67% |
| **Turn 8+** | **25** | **39.7%** | **21.00%** |

### The Insights Behind the Numbers

1. **The Pass@1 Illusion**: Evaluating the 28B model with standard single-turn prompting yielded only **10 passes (3.33%)**. In a standard single-shot leaderboard, this model would be written off as incapable of graduate-level theoretical computer science.
2. **Deep Iteration Drives 66.7% of Wins**: Two-thirds of all successful proofs (42 out of 63) required 4 or more turns of hypothesis testing, counterexample elimination in Python, and surgical blackboard revision.
3. **Late-Turn Synthesis Spike**: **Turn 8+ produced 25 passing proofs (39.7% of all wins)**—driven by the Publication Scribe synthesizing the accumulated lemma ledger after the Deductive Logician and Adversarial Skeptic eliminated dead routes in Turns 1–7.

---

## 8. Case Studies from the Benchmark

### Case Study A: An Autonomous 0.94 Victory on Output-Sensitive High-Dimensional Sumsets
Consider **Task 242** (`task_2410.21942_statement_30_71`), a research theorem on computing prefix-restricted sumsets for high-dimensional Las Vegas algorithms:

> **The Target**: Compute $C = (A + B) \cap [0 \dots t]^d$ for sets $A, B \subseteq [0 \dots t]^d$ in output-sensitive time:
> $$\mathcal{O}\left(\left(n + m + (nm)^{\frac{d}{d+1}} c^{\frac{1}{d+1}}\right) (\log(nm))^{\mathcal{O}(d)}\right)$$

1. **Turn 1 (Exploration Gate)**: Blocked from rushing a proof, the Logician explored coordinate-wise divide-and-conquer and identified linear-term blowup across recursive subproblems.
2. **Turns 2–4 (REPL & Skeptic Intervention)**: In the `TCSCompliantKernel`, small coordinate partitions ($d=2, 3$) revealed that coordinate-wise bounding fails unless dyadic scaling is applied to the bounding box $T$.
3. **Turns 5–7 (Structural Realization)**: The model pivoted to a $d$-dimensional dyadic box decomposition on $T$, defining shifted boxes $A_\alpha^*$ and bounding the non-vacuous subproblems using **Hölder's inequality** with conjugate exponents $p = \frac{d}{d+1}$ and $q = \frac{1}{d+1}$.
4. **Turn 8 (Publication Scribe Synthesis)**: The Scribe assembled a crisp, 7,275-character inductive proof over $T$, bounding the sum of recursive terms via Hölder's inequality over the $2^d$ sub-boxes.
5. **External Referee Verdict**: **Validity: 0.94** (*Rigor: 0.92, Completeness: 0.95, Confidence: 0.93*).

### Case Study B: Flipping a Borderline Failure into a 0.82 Pass (`task_2202.08066_statement_67_239`)
On **Task 69** (hierarchical tree edit-distance alignment reductions), an earlier harness iteration without invariant-first gating and domain-coverage checks stalled at a borderline **0.50 (`verdict: "0"`)** because it left tree-node substring length preconditions ($|Y_v| = |X_v| + 2K$) implicit and hand-waved capped-vs-uncapped edit-distance steps.
With `core_engine`'s **Turn-1 Exploration Gate**, **Critical-Path Scribe Pruning**, and **Stateless Blackboard Revision**, the prover explicitly discharged the inductive alignment invariants across all internal and leaf nodes—jumping **Rigor from 0.45 to 0.86**, **Completeness from 0.60 to 0.90**, **Boundary Cases from 0.40 to 0.82**, and overall **Validity from 0.50 to 0.82 (`PASS`)**.

---

## 9. Takeaways for AI Engineers Building Domain-Specific Reasoning Harnesses

If you are designing agent harnesses for reasoning-heavy vertical domains—whether graduate-level mathematics, formal verification, quantitative finance, compiler synthesis, or physics—eight core principles emerge from `core_engine`:

1. **Train the Harness via Embeddings, Not Weight Fine-Tuning (CoALA Continual Learning)**: Don't accept episodic amnesia from frozen LLM weights, and don't pay the catastrophic-forgetting cost of continual LoRA fine-tuning. Use a **CoALA Heuristics Engine** backed by role-isolated vector collections (`TurboVec`) and a strict universality/anti-leakage gate (`heuristic_validation.py`) so your harness *learns by solving*—distilling every solved task and every falsified failure into reusable domain knowledge.
2. **Never Trust Probabilistic Tokens Over Deterministic Execution**: Treat every mathematical assertion from an LLM as an *untrusted conjecture*. Pair the model with a **Zero-Trust Deterministic Kernel** (`SymPy`, `Z3`, 50-digit `mpmath`, `NetworkX`), preloaded diagnostic operators (`check_asymptotic_dominance`, `check_event_covariance`, `find_graph_counterexample`), and honest three-valued empirical semantics (`holds_on_samples`).
3. **Block Brute-Force Hacks to Force Structural Deduction**: Unconstrained code interpreters let LLMs cheat via massive sieves or numerical sequence extrapolation. Enforce hard analytical ceilings (`guarded_range` $\le 200,000$, `permutations` $n < 7$) and block extrapolation hacks in both the kernel and the memory validator.
4. **Track Epistemic Fragility Explicitly in the Kernel**: When mathematical objects undergo conditioning, subgraph restriction, or projection, use a formal state machine (`MathStateMachine`) to automatically revoke fragile properties (`independent`, `connected`, `invertible`) until re-verified by a witness.
5. **Enforce a Turn-1 Exploration Gate**: Never let a reasoning agent write its final deliverable on Turn 1. Force it to query its CoALA procedural memory, formulate testable `[CLAIM]` invariants, and write verification code in the kernel first.
6. **Quarantine Architectural Blueprints**: Planning subagents are great at high-level decomposition and terrible at tight algebraic constants. Treat planner blueprints as *unverified sketches*, and forbid the solver from restricting a problem's domain when a planner's loose intermediate bound fails.
7. **Separate Discovery from Typesetting (And Prune the Critical Path)**: Use an isolated **Publication Scribe** (with its own `tcs_scribe` memory) that traces backwards from the conclusion and omits dead exploratory side-lemmas.
8. **Repair on a Stateless Blackboard with Noise-Band Occam's Razor**: Collapse revisions to a 4-message blackboard, audit drafts in an ephemeral **AuditorKernel**, and penalize revisions that bloat length by $>50\%$ for a marginal score bump within auditor noise ($\pm 0.03$).

---

## 10. Academic References

- **[Cognitive Architectures for Language Agents (CoALA)](https://arxiv.org/abs/2309.02427)** *(Sumers et al., Princeton NLP, TMLR 2023)*: The tripartite cognitive memory taxonomy (Working, Procedural, and Episodic Memory) that Prime Math extends with 10 role-isolated TurboVec embedding collections, a 3-Tier `RemediationEngine`, and a 6-gate Zero-Leakage Universality Validator (`PROVENANCE_VERSION = 2`).
- **[TCS-Bench (`arXiv:2608.09538`)](https://arxiv.org/abs/2608.09538)**: The 300-problem graduate-level Theoretical Computer Science benchmark suite used to evaluate single-shot versus multi-turn neuro-symbolic proof synthesis.

*All benchmarks, CoALA memory modules, deterministic kernel engines, and evaluation scripts are open-sourced in the [Prime Math repository](https://github.com/dionyziuz/prime-math).*

