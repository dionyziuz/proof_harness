# How an Agent Harness Unlocks Mathematical Reasoning in a 27B Model

### *Why raw LLMs struggle with research-level math, and how a zero-trust code kernel, vector memory, and adaptive horizons boosted pass rates nearly 6x on consumer hardware.*

---

## The Illusion of the Fluent Mathematician

If you ask an off-the-shelf Large Language Model to solve a research-level math problem, you will usually get back an answer that *looks* brilliant. It uses the right terminology, cites standard theorems, and writes elegant LaTeX. 

Then you check line 4.

Somewhere in the middle of the derivation, an inequality quietly flipped direction. Or a random variable was assumed to stay independent after conditioning. Or a graph property was claimed to hold for all $n$, but fails completely on a 4-vertex cycle. 

Because autoregressive language models predict the next most likely token rather than verifying truth, they excel at sounding like mathematicians while routinely hallucinating algebraic leaps.

Parameter scale certainly matters: massive frontier models with hundreds of billions of parameters perform significantly better out of the box. But model scale is only part of the story—even the largest models struggle with subtle invariant shifts and false algebraic confidence when left unguided, while requiring expensive cloud compute clusters.

This raised a practical question: **How much reasoning performance can a well-engineered agent harness unlock from an accessible, local model?**

To test this, we took an open-weight 27B model (`unsloth/Qwen3.8-27B-NVFP4`) running locally on a single consumer GPU (an NVIDIA RTX 5090 with 32GB VRAM), and evaluated it on **300 research-level problems from TCS-Bench**—spanning spectral graph theory, quantum computing, parameterized complexity, and combinatorial algorithms.

When asked to draft proofs unassisted in a single pass, the model struggled, passing only **4.0%** of problems. 

Then we wrapped that exact same model inside **Proof Harness**—a specialized neuro-symbolic agent architecture. 

The results were striking:

| Evaluation Tier | Unassisted First Draft | Full Proof Harness | Multiplier |
| :--- | :---: | :---: | :---: |
| **All Passing Proofs (Grade C & Up, $\ge 0.40$)** | 4.0% (12 / 300) | **23.7% (71 / 300)** | **5.92× Gain** |
| **Certified Sound Proofs (Grade B & Up, $\ge 0.50$)** | 3.3% (10 / 300) | **21.0% (63 / 300)** | **6.30× Gain** |
| **Publication-Grade Proofs (Grade A & Up, $\ge 0.70$)** | 2.3% (7 / 300) | **18.0% (54 / 300)** | **7.71× Gain** |

### What Do the Grades Actually Mean?

Because mathematical deduction is not a multiple-choice benchmark, every proof is evaluated by an external calibrated referee across logical rigor, edge-case discharge, and structural completeness:

- **Grade A ($\ge 0.70$ — Publication Grade):** An airtight, journal-quality proof. Every step is rigorously justified, all quantifier regimes and boundary cases ($n=1, 2$, empty graphs, degenerate matrices) are explicitly proved, and preconditions for cited theorems (like Chernoff or Hölder bounds) are verified in full. Over $85\%$ of the harness's certified passes reached this publication-ready tier.
- **Grade B ($\ge 0.50$ — Certified Sound Pass):** The benchmark's headline pass threshold. The central mathematical reduction and derivations are completely sound, with no logical leaps, false claims, or hallucinated lemmas. Minor routine edge cases may be handled concisely, but the mathematical argument is fundamentally correct.
- **Grade C ($\ge 0.40$ — Directionally Sound Reduction):** The core proof architecture is correct (e.g., the right inductive invariant, reduction, or probabilistic technique was found), but contains minor technical gaps, an unverified constant, or a loose intermediate inequality that needs patching. This tier captures all proofs with substantive, viable mathematical foundations.
- **Below Grade C ($< 0.40$ — Refuted / Incomplete):** Fatal logical flaws, inverted inequalities, circular reasoning, illegal property inheritance (e.g., assuming independence after conditioning), or hand-waving prose like *"it obviously follows"*.

Nearly **6x as many passing proofs**, with 18% reaching publication-grade quality—all running on consumer hardware without fine-tuning a single model weight. (While our empirical results were benchmarked on consumer hardware, the harness itself is model-agnostic and can wrap any LLM).

Here is how the harness works, the engineering under the hood, and why it succeeded where one-shot generation failed.

---

## 1. Pillar I: The Zero-Trust Deterministic Kernel

The first rule of mathematical agents: **never trust an LLM's prose.**

If a language model claims that an inequality $f(n) \le g(n)$ holds for all positive integers, asking the model *"Are you sure?"* is useless. It will simply hallucinate a polite explanation of why it is sure.

Instead, Proof Harness pairs the LLM with a stateful, airgapped Python execution environment equipped with symbolic algebra (`SymPy`), SMT constraint solvers (`Z3`), arbitrary-precision arithmetic (`mpmath`), and graph analysis tools (`NetworkX`).

Whenever the model formulates an intermediate claim, it is forced to verify it in code:

- **Instant Counterexample Search**: Before spending thousands of tokens drafting a 5-page proof around an unverified lemma, the kernel sweeps small canonical instances (cycles, stars, bipartite graphs, boundary values). If a counterexample exists, the conjecture is killed in milliseconds on Turn 1.
- **50-Digit Arbitrary Precision (`mpmath`)**: Standard 64-bit floating-point arithmetic suffers from catastrophic cancellation errors when evaluating delicate asymptotic constants or spectral gaps. Pre-configuring 50 decimal places of precision ensures the agent isn't misled by IEEE-754 rounding artifacts.
- **Dual-Kernel Isolation**: If an internal referee validated a proof inside the same Python session where the solver wrote its messy scratchpad code, bugs in helper variables could quietly pollute the audit. Proof Harness isolates them: the solver gets a persistent scratchpad, but the Proof Auditor runs in a fresh, ephemeral kernel with a hard 15-second timeout.
- **The Epistemic State Machine (Fragile vs. Hereditary Properties)**: One of the most common failure modes in AI reasoning is illegal property inheritance. If a graph is planar or bipartite, its subgraphs remain planar or bipartite (*hereditary properties*). But if two random variables are independent, conditioning on their sum completely destroys that independence (*fragile properties*). Our state machine tracks the mathematical lineage of every object and automatically revokes fragile properties under conditioning, projection, or restriction until the model provides an explicit witness.
- **Anti-Brute-Force Guardrails**: When language models don't know how to prove something analytically, they often attempt brute-force loops (`for n in range(10**8):`). The kernel enforces a strict ceiling (`guarded_range` $\le 200,000$), forcing the model to stop looping and derive the exact analytical formula or generating function.

By treating every mathematical statement as guilty until proven innocent, the agent catches flawed logic early rather than compounding errors.

---

## 2. Pillar II: Continuous Domain Learning (A System that Gets Better Over Time)

The primary vision behind Proof Harness is to create a **continuous learning system that gets progressively smarter over time for practitioners using it—especially when focused on a specific domain** (such as spectral graph theory, quantum computing, algorithmic game theory, or cryptographic verification).

Standard LLMs suffer from **episodic amnesia**. An agent might spend 15 minutes and 10 iterations figuring out a clever reduction or uncovering a tricky boundary condition, only to forget everything the second the session ends. On the very next problem in that same field, it starts from scratch and repeats the exact same mistakes.

Fine-tuning model weights after every problem isn't practical—it is slow, computationally expensive, and frequently causes catastrophic forgetting.

Proof Harness solves this by implementing **continuous domain learning through curated vector memory**:

1. **Strategy Extraction**: When a problem is solved (or when a failure is analyzed), a low-temperature reflection pass distills verified mathematical strategies into concise heuristics (e.g., *"When bounding high-dimensional sumsets, dyadic decomposition prevents recursive dimension blowup"*).
2. **Generalization & Sanitization**: Candidate heuristics are sanitized to remove instance-specific numbers or equations, ensuring only reusable, generalizable mathematical strategies are committed to memory.
3. **Role-Partitioned Retrieval**: Learned heuristics are stored in dedicated vector collections for each cognitive role (e.g., the Context Architect retrieves structural blueprints, while the Adversarial Skeptic retrieves known counterexample templates).

**The Result: Compounding Domain Expertise Without Fine-Tuning**  
As a research team or practitioner uses the harness on a specific domain of problems, the system steadily builds up a curated repository of proven reduction patterns, boundary warnings, and verification templates. The base model's weights remain completely untouched, but the harness compounds domain expertise over time—becoming progressively faster, sharper, and more capable with every theorem it proves.

---

## 3. Pillar III: Cognitive Roles, Turn-1 Exploration, and Adaptive Overtime

Modern reasoning models spend significant internal chain-of-thought thinking before generating an answer. But purely internal thinking has a major blind spot: it relies entirely on the model's own latent representations without external ground truth. An LLM can "think" through thousands of internal tokens while reinforcing a subtly false algebraic step or an invalid property inheritance.

Furthermore, in unguided agent loops, models typically attempt to draft the complete final proof in their very first response turn. Once an initial proof draft is on the page, subsequent revision turns tend to anchor on that existing text—defending the initial path and patching superficial wording rather than rigorously checking whether the core mathematical invariants hold.

To prevent premature commitment to flawed algebraic paths, Proof Harness orchestrates the search through specialized cognitive roles and an explicit **Turn-1 Exploration Gate**:

- **The Turn-1 Exploration Gate**: The model is strictly banned from generating a formal `\begin{proof}` on Turn 1. Instead, the entire first interaction turn must be dedicated to computational exploration: inventorying context lemmas, querying vector memory for proven strategies, formulating testable `[CLAIM]` invariants, and executing verification scripts in the deterministic kernel. Formal proof synthesis only begins on Turn 2 after invariants have been tested in code.
- **Role Specialization**: 
  - The **Context Architect** sketches high-level reduction blueprints.
  - The **Logician** tests steps iteratively in the Python REPL.
  - The **Adversarial Skeptic** actively searches for edge-case counterexamples.
  - The **Publication Scribe** writes the formal, rigorous LaTeX only after the mathematical foundation is verified.
- **Stateless Blackboard Revisions**: If an internal referee finds a flaw in a draft, the harness doesn't keep a massive, noisy chat history that confuses the model. It collapses the context into a clean, 4-message blackboard: the problem statement, the best draft so far, and the specific surgical repair directive.
- **The High-Potential Overtime Gate**: Fixed turn limits either waste tokens on lost causes or cut off promising proofs just before completion. At Turn 8 (the nominal horizon), the harness checks if the candidate proof achieved directional validity ($\text{score} \ge 0.35$). If so, it dynamically unlocks an **extra 4 iterations (Turns 9–12)** and $+1,800\text{s}$ wall-clock budget for surgical polishing.

In our 300-problem benchmark, **100% of all passing proofs were discovered through this governed multi-turn process**. Over 84% of wins were achieved in Turn 4 or later, and the **Overtime Gate alone delivered 13 certified passing proofs (over 20% of all wins)** that would have been lost under a rigid 8-turn cutoff.

---

## Inside the Benchmark: Why the Harness Won Where One-Shot Failed

To see the difference in practice, consider three real problems from TCS-Bench where unassisted generation failed, but the harness found a sound proof:

### 1. The Blueprint Trap: Theta-Series Tail Bound
- **The Problem**: Proving a universal tail bound on a theta-function series across all positive parameter values $\beta > 0$.
- **Why One-Shot Failed**: In an unguided pass, the model used a standard loose intermediate bound. For small values ($\beta \le 1$), the intermediate bound failed. Instead of replacing the loose bound, the unguided model arbitrarily restricted the theorem statement (*"assuming $\beta \le 1$"*), immediately failing the referee's validity check for illegal domain narrowing.
- **How the Harness Solved It**: When the Logician tested the planner's blueprint in the kernel, the code immediately flagged that the intermediate bound broke. The harness quarantined the blueprint, enforced full domain coverage, and compelled the model to find a tighter bound valid for all $\beta > 0$, scoring **0.68 (Certified Pass)**.

### 2. The False Independence Trap: Graph Sparsification
- **The Problem**: Proving cut-preservation under edge sampling in graph sparsification.
- **Why One-Shot Failed**: The unassisted model casually applied Chernoff bounds by asserting that edge-inclusion events were mutually independent, completely ignoring that edges sharing vertices have subtle covariance under conditioned cuts ($v = 0.20$).
- **How the Harness Solved It**: The Adversarial Skeptic ran the kernel's Monte Carlo covariance auditor on small seeded graphs, which triggered an immediate `DEPENDENCE DETECTED` alert. The model pivoted to a union bound over spanning forests that did not rely on false independence, scoring **0.96 (Journal-Grade Pass)**.

### 3. The Exponential Blowup: High-Dimensional Sumsets
- **The Problem**: Computing prefix-restricted sumsets $C = (A + B) \cap [0 \dots t]^d$ in output-sensitive time $\mathcal{O}\left((n + m + (nm)^{\frac{d}{d+1}} c^{\frac{1}{d+1}}) \log(nm)^{\mathcal{O}(d)}\right)$.
- **Why One-Shot Failed**: The single-shot model attempted coordinate-wise divide-and-conquer, failing because unscaled recursive subproblem costs grow exponentially in dimension $d$.
- **How the Harness Solved It**: Over 8 turns, the model used `SymPy` in the kernel to test coordinate partitions. It discovered that bounding the non-vacuous subproblems required a simultaneous $2^d$-ary dyadic box decomposition, and verified that **Hölder's inequality** with conjugate exponents $p = \frac{d}{d+1}$ and $q = \frac{1}{d+1}$ collapsed the recursive sum cleanly into the target bound, earning **0.94 (Publication-Grade Pass)**.

---

## Key Takeaways for AI Engineers

Whether you are building agents for mathematics, software engineering, quantitative finance, or scientific research, the architectural lessons from Proof Harness apply broadly:

1. **Don't trust language models to verify their own prose.** Connect your agent to deterministic ground truth—code interpreters, compilers, linters, or SMT solvers.
2. **Enforce an exploration phase.** Never let an agent jump straight to the final output on Turn 1. Force it to inspect inputs, test assumptions, and explore the search space first.
3. **Use vector memory for fast, weight-free learning.** Distilling lessons into curated, validated vector embeddings lets your agents get smarter over time without the cost and instability of continuous model fine-tuning.
4. **Give promising runs adaptive overtime.** Rigid turn budgets cause near-misses. Triggering dynamic extra iterations on promising candidates can convert 20%+ of borderline attempts into clean successes.
5. **Isolate generation from critique.** Have separate execution environments for drafting solutions and finding flaws. An agent asked to grade its own work will almost always be overly optimistic.
6. **Keep multi-turn context clean.** Long conversation histories accumulate noise. Summarize or reset to a clean "blackboard" containing only the current best candidate and the exact failure to address.

---

## Looking Forward

Larger foundation models will continue to push the frontier of raw capability. But as our benchmark shows, how you scaffold and govern a model's reasoning process—coupling it with deterministic code verification, structured vector memory, and adaptive search horizons—has a sizeable, transformative impact on what language models can actually solve.

By pairing an open-weight 27B model with a zero-trust execution kernel, continuous heuristic memory, and adaptive search horizons, Proof Harness demonstrates that consumer hardware can tackle problems that previously seemed out of reach.

*For implementation details, evaluation scripts, and the full benchmark harness, check out the [Proof Harness repository on GitHub](https://github.com/dionyziuz/proof_harness). For theoretical analysis and comprehensive ablation studies, our full technical paper is available in the repository documentation.*
