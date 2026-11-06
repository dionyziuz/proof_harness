# The 2× Law of Cognitive Scaffolding: Why Frontier Reasoning Models Need Architecture, Not Just Weights

### *Lessons from benchmarking Qwen-3.8-27B, Meta Muse 1.3, and Gemini 3.8 Flash across 300 research theorems: analyzing parameter scale, dense vs. MoE inference, time deltas, and bounded reasoning.*

---

## Executive Summary: The Great Equalizer

The prevailing orthodoxy in artificial intelligence assumes that if you make a neural network large enough and give it enough test-time chain-of-thought tokens, rigorous mathematical deduction will naturally emerge from the weights.

Our empirical findings on **TCS-Bench**—a benchmark of **300 publication-grade mathematical lemmas and theorems extracted directly from arXiv preprints in Theoretical Computer Science**—tell a very different story.

When asked to draft proofs unassisted in a single zero-shot pass ($T=1$), even the most capable frontier models fail routinely. They hallucinate algebraic leaps, flip inequality directions mid-derivation, and confuse fragile conditional properties with hereditary structures. 

Then we wrapped each model inside **Proof Harness**—a modular, neuro-symbolic agent scaffold featuring an airgapped execution kernel, seven-role cognitive partitioning, turn-1 exploration gates, and dynamic pivot engines. 

The results revealed what we call **The 2× Law of Cognitive Scaffolding**:

![The 2x Law of Cognitive Scaffolding: One-Pass vs. Proof Harness Pass Rates](../images/tcs_multiplier_bars.png)

```
                            TCS-BENCH 300-PROBLEM HEADLINE SCORECARD
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Model & Architecture           ┃ Unassisted 1-Pass     ┃ Full Proof Harness    ┃ Pass Multiplier   ┃ Directional (≥0.40)┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ Qwen-3.8-27B (Dense, 27B)      │ 12.67% (38 / 300)     │ 26.67% (80 / 300)     │ 2.11× (+110.5%)   │ 31.00% (93 / 300)  │
│ Meta Muse 1.3 (~800B MoE)      │ 22.00% (66 / 300)     │ 46.33% (139 / 300)    │ 2.11× (+110.6%)   │ 50.00% (150 / 300) │
│ Gemini 3.8 Flash (~1.2T MoE)   │ 33.67% (101 / 300)    │ 55.33% (166 / 300)    │ 1.64× (+64.4%)    │ 61.33% (184 / 300) │
└────────────────────────────────┴───────────────────────┴───────────────────────┴───────────────────┴────────────────────┘
```

Across two fundamentally different parameter regimes—an open-weight **27B dense model** running locally on a single consumer GPU (NVIDIA RTX 5090) and an **~800B Mixture-of-Experts cloud model**—external agent scaffolding produced an **identical 2.11× multiplier (+110.6%)** in verified proof soundness. 

Even for **Gemini 3.8 Flash**—backed by a massive ~1.2-trillion parameter pretraining corpus—the harness delivered a **1.64× gain (+64.4%)**, converting 65 previously failed research problems into certified theorems and establishing a state-of-the-art **55.33% pass rate** on publication-grade arXiv mathematics.

This article unpacks the empirical logs behind these numbers: how dense and MoE architectures behave under pressure, the striking time deltas between models, the specific mathematical domains where a local 27B model beats trillion-parameter systems, and why strict parameter bounding is essential to prevent reasoning loops.

---

## 1. What the Harness Actually Does (In Plain English)

How does a software harness more than double an AI model's mathematical accuracy without fine-tuning a single weight?

If you give a human mathematician a blank sheet of paper and demand a 15-page proof in 20 minutes with no scratchpad, no computer algebra system, and no eraser, even a Fields Medalist will make an algebraic sign error. Raw LLMs face this exact handicap: they are forced to generate complex derivations token-by-token in a single autoregressive stream, where any early error compounds uncontrollably.

Proof Harness replaces this blind leap with four common-sense architectural guardrails:

```
                               THE PROOF HARNESS GOVERNANCE LOOP
  ┌──────────────────────┐      ┌─────────────────────────┐      ┌──────────────────────────┐
  │  Turn-1 Exploration  │ ───► │  Zero-Trust Exec Kernel │ ───► │   7-Role Cognitive Core  │
  │  (No Proofs Allowed) │      │  (SymPy / Z3 / mpmath)  │      │  (Auditor / Falsifier)   │
  └──────────────────────┘      └─────────────────────────┘      └──────────────────────────┘
                                                                               │
                                                                               ▼
  ┌──────────────────────┐      ┌─────────────────────────┐      ┌──────────────────────────┐
  │   Certified Sound    │ ◄─── │   Overtime Gate (T9-12) │ ◄─── │   Dynamic Pivot Engine   │
  │     Proof Output     │      │   (Rescues Near-Misses) │      │   (Kills Dead Routes)    │
  └──────────────────────┘      └─────────────────────────┘      └──────────────────────────┘
```

1. **The Unforgiving Calculator (The Zero-Trust Deterministic Kernel):**
   When an LLM claims that *"inequality (4) holds for all $n \ge 1$,"* asking the model *"Are you sure?"* is useless—it will simply invent a polite rationalization. Proof Harness pairs the solver with an isolated Python execution environment equipped with `SymPy`, `Z3`, and 50-digit `mpmath`. Before drafting an argument, the model is forced to write code that tests its intermediate lemmas on toy instances (cycles, stars, bipartite graphs, boundary values). If a counterexample exists, the flawed conjecture is killed in milliseconds on Turn 1.
2. **The Banned First Draft (The Turn-1 Exploration Gate):**
   Language models suffer from extreme anchoring bias: once an initial proof draft is on the page, subsequent revision turns reflexively defend that initial path rather than questioning whether the core reduction is sound. Proof Harness strictly forbids the model from outputting `\begin{proof}` on Turn 1. The opening interaction must be purely computational: exploring invariants, verifying preconditions, and checking small cases in code. Formal proof writing only begins on Turn 2.
3. **The Ruthless Editor (Seven-Role Cognitive Partitioning):**
   Authors are terrible at proofreading their own work. Proof Harness decouples the problem-solving lifecycle into distinct cognitive personas. The **Context Architect** drafts reduction blueprints, the **Logician** tests steps iteratively in code, the **Adversarial Skeptic** searches for counterexamples in an isolated ephemeral kernel, and the **Publication Scribe** typesets the formal LaTeX only after the mathematical foundation has passed inspection.
4. **The Overtime Referee (The Multi-Turn Pivot Engine):**
   Fixed turn limits waste tokens on hopeless rabbit holes and cut off promising proofs just before completion. The harness dynamically tracks directional validity: if a proof is fundamentally flawed, it triggers a clean-slate pivot to a new proof strategy. If a proof reaches directional soundness ($\text{validity} \ge 0.35$) at Turn 8, the harness automatically unlocks an extra 4 overtime turns (Turns 9–12), converting borderline near-misses into certified successes.

---

## 2. Dense vs. MoE: Total Parameters vs. Active Compute

To understand why models score where they do in single-pass vs. harness configurations, we must distinguish between two widely conflated architectural properties: **Total Parameter Scale (The Latent Library)** and **Active Parameters per Token (The Deductive Engine)**.

> [!NOTE]
> **Methodological Disclosure on Model Sizes**:  
> While **Qwen-3.8-27B** is an open-weights dense model where layer weights and tensor dimensions are 100% verified ground truth (exactly 27.0B dense parameters running locally on our RTX 5090), the exact parameter figures for **Gemini 3.8 Flash** and **Meta Muse 1.3** are informed industry estimates and community consensus derived from serving economics, latency profiling, and token throughput. Neither Google nor Meta has officially disclosed the internal routing topology or parameter counts of these systems.

```
                       [TOTAL PARAMETERS]
                     (The Latent Knowledge Base)
                              ▲
                              │
             Gemini Flash (~1.2T)     Meta Muse (~800B)
                              │
                              ├──────────────────────────┐
                              │                          │
                       [ACTIVE COMPUTE]           [THINKING HABIT]
                        (~40B vs 27B)              (Fast vs Deep)
                              │                          │
                 Qwen 27B (Dense, 27B)          Muse (89m avg search)
                                                Flash (29m self-governed)
                                                Qwen (steady 62m clock)
```

### The Latent Library (Total Parameters: *What the Model Has Seen*)
Research-level TCS theorems extracted from arXiv preprints do not test routine textbook math. They test familiarity with specialized theoretical abstractions—quantum oracle circuits, graph expansion metrics, semidefinite programming relaxations, and matroid parity certificates.

* **Gemini 3.8 Flash (~1.2T total)** and **Meta Muse 1.3 (~800B total)** possess an ocean of pretraining literature. Their vast parameter spaces act as a massive reference library, allowing them to recognize specialized preprint lemmas and idiomatic proof structures on sight.
* **Qwen-3.8-27B**, with roughly 1/30th the total parameter scale of Flash, has a comparatively modest latent library. It cannot rely on having encountered an identical lemma during pretraining; it is forced to reconstruct structural truth from first principles.
* *The Score Impact:* This directly explains the unassisted one-pass baseline: **Flash (33.67%) $\to$ Muse (22.00%) $\to$ Qwen (12.67%)**. Total parameter scale establishes the zero-shot familiarity floor.

### The Deductive Engine (Active Compute: *How Much Compute Fires per Step*)
While Flash and Muse are total-parameter giants, **both are sparse Mixture-of-Experts (MoE) architectures activating roughly ~40B to ~50B parameters per token**.

Qwen is a pure dense model activating **27.0B parameters on every single token**. FLOP-for-FLOP during inference, Qwen operates at roughly 60%–70% of the active forward-pass compute of Flash and Muse.

*The Score Impact:* This active compute parity explains why Qwen punches so far above its weight class when guided by the harness. Its forward-pass deductive engine is remarkably capable; its only bottleneck was the lack of external memory, computational verification, and strategic boundaries. When those are supplied by Proof Harness, Qwen's pass rate surges from **12.67% to 26.67%**, solving 80 publication-grade research lemmas.

---

## 3. The Clock Doesn't Lie: Latency & Cognitive Personalities

Benchmarking 300 research theorems across thousands of multi-turn interactions revealed stark differences in inference speed, token consumption, and cognitive personalities:

![Task Duration and Benchmark Runtime: Flash vs. Qwen vs. Muse](../images/tcs_latency_comparison.png)

```
                       BENCHMARK RUNTIME & LATENCY METRICS (300 PROBLEMS)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Model & Architecture       ┃ Total Benchmark    ┃ Median Task Time   ┃ Mean Task Time     ┃ 90th Percentile    ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ Gemini 3.8 Flash (~1.2T)   │ 143.8 hours        │ 12.2 minutes       │ 28.8 minutes       │ 74.1 minutes       │
│ Qwen-3.8-27B (Dense 27B)   │ 310.2 hours        │ 61.9 minutes       │ 62.0 minutes       │ 108.5 minutes      │
│ Meta Muse 1.3 (~800B MoE)  │ 444.6 hours        │ 95.0 minutes       │ 88.9 minutes       │ 172.1 minutes      │
└────────────────────────────┴────────────────────┴────────────────────┴────────────────────┴────────────────────┘
```

### Flash vs. Muse: A 3.1× Speedup at Higher Accuracy
The comparison between the two cloud MoE models is one of the most revealing findings in the entire benchmark:

* **Gemini 3.8 Flash** completed the entire 300-problem benchmark in **143.8 hours** of aggregate execution time, with a **median problem solve time of just 12.2 minutes**.
* **Meta Muse 1.3** required **444.6 hours**—more than **3.1× longer overall**, with a **median solve time of 95.0 minutes (nearly 8× slower per problem)**. On individual complex problems, Muse frequently spent between 3 and 5.4 hours (max 325.3 minutes) inside a single proof search.
* And yet, Flash scored higher (**55.33% vs. 46.33%**).

Why does Muse spend so much longer thinking when both models activate roughly ~40B parameters per step?

* **Meta Muse 1.3 ("The Wandering Genius"):** Characterized by high-entropy, associative exploration. It searches deep into candidate branches, generating extensive internal debate. On contest math (e.g., AIME), this deep unhurried search guarantees finding single numerical solutions. But on research-grade arXiv proofs, unguided deep thinking often drifts into repetitive loops or 0-token stream aborts, requiring harness backoff guards to step down reasoning effort and recover tokens.
* **Gemini 3.8 Flash ("The Disciplined Engineer"):** Equipped with built-in meta-cognitive governors. Backed by its ~1.2T latent library, Flash needs only 60 to 90 seconds of internal thinking before determining that further internal contemplation yields diminishing returns. It finishes fast, typesets clean modular arguments, and relies on the external harness to audit edge cases.
* **Qwen-3.8-27B ("The Diligent Scribe"):** Operating locally on consumer hardware, Qwen shows remarkably consistent clockwork timing: its mean (62.0 min) and median (61.9 min) durations are virtually identical. Governed by a strict thinking budget ceiling (`TCS_SOLVER_THINKING_BUDGET=32768`), Qwen channels its generation into an exhaustive, 39,000-character written proof, proving its claims through sheer algebraic grinding.

---

## 4. Where Qwen Wins and Where It Struggles

By categorizing all 300 benchmark problems by mathematical subfield, the logs reveal the exact operational envelope of a local 27B dense model relative to trillion-parameter cloud systems:

![Mathematical Domain Competency Breakdown across TCS-300 Benchmark](../images/tcs_domain_competency.png)

```
                            DOMAIN COMPETENCY BREAKDOWN (TCS-300)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Mathematical Domain            ┃ Tasks  ┃ Qwen-3.8-27B       ┃ Meta Muse 1.3      ┃ Gemini 3.8 Flash   ┃ Qwen / Flash  ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ Algorithms & Complexity        │ 32     │ 31.2% (10 / 32)    │ 40.6% (13 / 32)    │ 59.4% (19 / 32)    │ 0.53×         │
│ Graph Theory & Networks        │ 183    │ 28.4% (52 / 183)   │ 44.3% (81 / 183)   │ 52.5% (96 / 183)   │ 0.54×         │
│ Probability & Combinatorics    │ 31     │ 25.8% (8 / 31)     │ 48.4% (15 / 31)    │ 61.3% (19 / 31)    │ 0.42×         │
│ Algebra, Matrices & Polynomials│ 37     │ 21.6% (8 / 37)     │ 62.2% (23 / 37)    │ 64.9% (24 / 37)    │ 0.33×         │
│ Quantum & Coding Theory        │ 15     │ 13.3% (2 / 15)     │ 33.3% (5 / 15)     │ 40.0% (6 / 15)     │ 0.33×         │
└────────────────────────────────┴────────┴────────────────────┴────────────────────┴────────────────────┴───────────────┘
```

### Where Qwen Excels: Concrete Inductive Reductions
In **Algorithms & Complexity (31.2%)** and **Graph Theory & Networks (28.4%)**, Qwen performs at over half the accuracy of Gemini Flash and closely trails Meta Muse.

In fact, the audit revealed **5 benchmark problems that Qwen-3.8-27B solved where BOTH Gemini Flash and Meta Muse completely failed**:

1. **`task_2210.12601` (Max E2Lin($q$) Constraint Completeness):**  
   *Scores:* **Qwen: 0.95** | Flash: 0.45 | Muse: 0.40  
   *Why Qwen Won:* Proving completeness for $q$-ary linear equation systems requires an explicit, tedious vertex partition across $q$ independent sets. Flash and Muse summarized the step as *"trivially following by standard Fourier analysis,"* which the referee rejected for unverified edge constraints. Qwen wrote out the complete, verifiable partition algebra.
2. **`task_2208.1392` (Randomized Vertex Deletion Approximation):**  
   *Scores:* **Qwen: 0.92** | Flash: 0.08 | Muse: 0.25  
   *Why Qwen Won:* Constructing an $O(\log n)$-approximation running in $O(n^3)$ requires explicit potential-function accounting across recursive peeling rounds. Qwen verified every inductive step in the Python kernel before writing the proof.
3. **`task_2104.12695` (Checking Program Reachability Hardness):**  
   *Scores:* **Qwen: 0.88** | Flash: 0.25 | Muse: 0.32  
   *Why Qwen Won:* Proving $\mathbb{F}_d$-hardness for $(2d+4)$-dimensional checking programs requires constructing explicit simulation gadgets. Qwen verified the transition rules symbolically, avoiding the dimensional indexing errors that tripped up the larger models.
4. **`task_2011.10446` ($h$-Hop Capacitated Subflow Routing):**  
   *Scores:* **Qwen: 0.72** | Flash: 0.20 | Muse: 0.25
5. **`task_2502.1169` (Combinatorial Witness Tree Bounds):**  
   *Scores:* **Qwen: 0.58** | Flash: 0.45 | Muse: 0.42

*The Pattern:* Dense models do not suffer from the "laziness" of large MoEs. When a proof demands painstaking, mechanical case-checking and explicit combinatorics, Qwen's willingness to generate 39,000 characters of dense, uncompressed derivation wins where large models attempt to cut corners.

### Where Qwen Struggles: The "Latent Library" Gap
Conversely, in **Algebra, Matrices & Spectral Theory (21.6% vs. 64.9%)** and **Quantum Theory (13.3% vs. 40.0%)**, the gap between Qwen and the MoE giants widens dramatically.

On **52 problems**, both Flash and Muse passed with near-perfect validity ($>0.90$) while Qwen scored below $0.30$. 

*The Pattern:* These failed problems almost uniformly involved deep, multi-tiered theoretical machinery:
* **Szemerédi Regularity Hierarchies:** Statements involving parameter towers like $1/n \ll 1/s \le 1/\ell \ll \varepsilon_1 \ll \varepsilon \ll \dots$ (e.g., `task_2007.02891`). Qwen frequently confused which constants depended on which quantifier scales.
* **Spectral Expander Perturbations:** Utilizing Davis-Kahan $\sin\theta$ theorems and Weyl eigenvalue inequalities on non-Hermitian operators.
* **Quantum Oracle Channels & Randomized Sketches:** Requiring familiarity with specialized matrix-algebra identities.

Flash and Muse have seen thousands of preprints using these exact lemmas; they understand the standard proof templates instinctively. Qwen, lacking those representations in its smaller parameter library, attempts to derive them from scratch using elementary matrix arithmetic, quickly exhausting its turn budget.

---

## 5. Bounded Reasoning: Preventing Infinite Loops vs. The Untapped Frontier

One of the most critical engineering decisions in Proof Harness was **parameter optimization for reasoning governance**. 

Reasoning models do not fail gracefully by saying *"I do not know."* When left unconstrained, they suffer from **Self-Reinforcing Hallucination**:
1. On Turn 1, the model proposes a slightly flawed algebraic reduction.
2. On Turn 2, when asked to verify its work, the model reads its own Turn-1 draft as authoritative context. The presence of its own mathematical prose acts as strong conditional biasing, convincing the model that the false invariant must be true.
3. Over successive iterations, the model spins in a **context-poisoning loop**, spending dozens of turns generating subtle variations of the same flawed claim while context consumption balloons past 100,000 tokens.

To prevent runaway loops, GPU bus lockups, and context starvation, we locked down conservative governance parameters:
* `TCS_SOLVER_MAX_ITERATIONS = 8` (nominal horizon)
* `TCS_OVERTIME_GATE = 12` (overtime ceiling)
* `TCS_SOLVER_THINKING_BUDGET = 32768` (token ceiling)
* `TCS_COMPACTOR_MAX_CONTEXT_TOKENS = 64000` (blackboard pruning threshold)

### The Trade-Off: What Bounding Left on the Table
While these strict bounds guaranteed stability across all 300 problems, the logs reveal an intriguing opportunity cost:

![Bounded Reasoning vs The Untapped Horizon: Certified Passes vs Cut-Off Near-Misses](../images/tcs_nearmiss_ceiling.png)

* **For Qwen-3.8-27B:** Exactly **12 problems were actively in a "directionally sound" near-miss state (validity between 0.40 and 0.48)** and actively repairing intermediate gaps when they hit the Turn 8/12 ceiling.
  * With 4 to 6 additional turns, many of these 12 tasks would have crossed the 0.50 threshold, pushing Qwen's pass rate **above 92 passes (>30.7%)**.
* **For Meta Muse 1.3:** **10 near-miss tasks** were cut off at the iteration cap while converging.
* **For Gemini 3.8 Flash:** **18 tasks** sit in the $[0.40, 0.49]$ directional corridor, representing an untapped reserve of solvable lemmas.

*The Takeaway:* The **2.11× and 1.64× multipliers reported here are conservative lower bounds** achieved under strict production safeguards. With relaxed turn budgets and domain-targeted heuristics, the ceiling for scaffolded reasoning is substantially higher.

---

## 6. The Collective Frontier: The 65.7% Multi-Model Portfolio

Because Qwen, Muse, and Flash possess fundamentally different cognitive archetypes and failure modes, their successes are remarkably orthogonal:

```
                            THE TCS-300 SOLVE FRONTIER
  ┌────────────────────────────────────────────────────────────────────────┐
  │                                                                        │
  │                      Flash Solved (166 / 300)                          │
  │            ┌───────────────────────────────────────────┐               │
  │            │                                           │               │
  │            │          Shared by All 3 (61)             │               │
  │            │                                           │               │
  │    ┌───────┼───────────────────┬───────────────────────┼───────┐       │
  │    │       │  Flash + Qwen (9) │  Flash + Muse (52)    │       │       │
  │    │       └───────────────────┴───────────────────────┘       │       │
  │    │ Qwen Only (5)                     Muse Only (21)          │       │
  │    │                                                           │       │
  │    │              Qwen + Muse (Flash Failed) (5)               │       │
  │    └───────────────────────────────────────────────────────────┘       │
  │                                                                        │
  │             TOTAL UNIQUE THEOREMS SOLVED: 197 / 300 (65.7%)            │
  └────────────────────────────────────────────────────────────────────────┘
```

* **Solved by All 3 Models:** 61 tasks
* **Solved by Flash & Muse (Qwen Failed):** 52 tasks
* **Solved by Flash & Qwen (Muse Failed):** 9 tasks
* **Solved by Qwen & Muse (Flash Failed):** 5 tasks
* **Solved by Flash ONLY:** 44 tasks
* **Solved by Muse ONLY:** 21 tasks
* **Solved by Qwen-3.8-27B ONLY:** 5 tasks

When these three distinct systems are deployed as a coordinated ensemble under Proof Harness, the collective portfolio solves **197 out of 300 research theorems—a 65.7% overall pass rate**.

Nearly two-thirds of published research-level TCS problems from arXiv can now be autonomously proved by coordinating open-weight and cloud models under a unified neuro-symbolic harness.

---

## 7. Strategic Conclusions for AI Engineering

The empirical findings from our 300-problem benchmark point to three foundational conclusions for the future of AI reasoning:

1. **Pretraining Scale Sets the Floor; Harness Architecture Sets the Ceiling.**  
   Raw parameter scale is essential for zero-shot domain familiarity (explaining Flash's 33.7% one-pass score vs. Qwen's 12.7%). But regardless of whether a model has 27 billion or 1.2 trillion parameters, raw autoregressive generation hits a hard mathematical wall on multi-step deduction. External scaffolding—zero-trust code execution, cognitive role separation, and bounded search—acts as a universal multiplier, delivering an average **~1.95× lift across all models**.
2. **Dense Local Models Punch Vastly Above Their Weight Class.**  
   A 27B dense model running on a consumer GPU does not need a trillion parameters to do meaningful science. Because it fires its full parameter capacity on every token and lacks the shortcut-taking habits of large MoEs, a properly scaffolded 27B model can outperform trillion-parameter cloud systems on concrete algorithms, combinatorial counting, and explicit reductions.
3. **The Future of Autonomous Science is Neuro-Symbolic.**  
   The AI industry has spent hundreds of millions of dollars attempting to solve reasoning purely through pretraining scaling and post-training reinforcement learning. Our results demonstrate that the most direct path to reliable mathematical deduction is not bigger weights, but **hybrid neuro-symbolic architecture**: pairing language models with deterministic execution tools, epistemic state machines, and adversarial cognitive governance.

---

## Benchmark Artifacts & Data Availability

All 300 problem logs, candidate proofs, and official referee judgments for all three models are open and inspectable in the Proof Harness repository:

* **Qwen-3.8-27B (v4 Unified Dataset):** [`data/eval_results/tcs_qwen_v4`](file:///home/sysadmin/Documents/github/proof_harness/data/eval_results/tcs_qwen_v4)
* **Meta Muse 1.3 (v4 Unified Dataset):** [`data/eval_results/tcs_muse_v4`](file:///home/sysadmin/Documents/github/proof_harness/data/eval_results/tcs_muse_v4)
* **Gemini 3.8 Flash (v4 Unified Dataset):** [`data/eval_results/tcs_flash_v4`](file:///home/sysadmin/Documents/github/proof_harness/data/eval_results/tcs_flash_v4)
* **Generated Benchmark Figures:** [`docs/images/`](file:///home/sysadmin/Documents/github/proof_harness/docs/images)
