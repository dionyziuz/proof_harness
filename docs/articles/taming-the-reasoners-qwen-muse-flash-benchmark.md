# Taming the Reasoners: Lessons from Benchmarking Qwen-3.8-27B, Meta Muse 1.3, and Gemini 3.8 Flash on Research-Level Mathematics

### *An empirical deep-dive into reasoning dynamics, token expenditure, and harness governance across 300 Theoretical Computer Science frontier proofs.*

---

## Executive Summary & Narrative Hook
* **The Premise**: As frontier reasoning models ("thinking LLMs") become standard in AI research, raw benchmark claims obscure real-world execution realities. Putting three fundamentally different reasoning engines—**Qwen-3.8-27B** (local weights on RTX 5090), **Meta Muse 1.3** (`muse-spark-1.3-contributor`), and **Google Gemini 3.8 Flash** (`gemini-3.8-flash-high`)—through the grueling **TCS-300** benchmark reveals radically distinct cognitive personalities.
* **The MoE Parameter Paradox**: Both Meta Muse and Gemini are massive, state-of-the-art Mixture-of-Experts (MoE) architectures commanding hundreds of billions of parameters. On public competition math benchmarks like **AIME 2024/2025**, Meta Muse Spark posts astronomical numbers (**96.88%**, rivaling GPT-5.2 and Gemini Pro). Yet, when unleashed unassisted on research-level arXiv proofs in TCS-300, **Gemini Flash nearly doubles Muse's certified pass rate (33.67% vs. 22.00%) at 2.5× the speed**.
* **The Discovery**: Model intelligence is not just a function of parameters or leaderboard accuracy; it is governed by **how the model thinks, when it stops thinking, and how the harness manages runaway reasoning loops**.
* **The Central Thesis**:
  * **Qwen-3.8-27B** is prone to runaway self-debate and requires hard external throttling (`TCS_SOLVER_THINKING_BUDGET`).
  * **Meta Muse 1.3** is a deep, high-entropy reasoner that excels on contest puzzles but easily goes off the rails on dense research definitions without an adaptive step-down backoff ladder.
  * **Gemini 3.8 Flash** exhibits extraordinary internal self-governance and velocity, proving that a self-throttling reasoner makes the ultimate autonomous research primitive.

---

## Section 1: The Arena — TCS-300 Benchmark & The 2-Step Funnel
1. **The Testbed**: 300 publication-grade mathematical lemmas and theorems extracted directly from arXiv preprints in Theoretical Computer Science (TCS), spanning graph theory, quantum complexity, circuit lower bounds, and randomized algorithms.
2. **The 2-Step Funnel Architecture**:
   * *Step 1: Fast Unassisted One-Pass Triage ($T=1$)*: Filter out the "low-hanging" frontier proofs with pure, zero-scaffolding generation.
   * *Step 2: Multi-Turn Cognitive Harness Escalation ($T \le 8$ or $12$)*: Engage persistent Python REPL kernels, multi-agent debate (Architect, Falsifier, Skeptic, Scribe), and the Academic Proof Auditor only on what Step 1 failed to solve.
3. **Evaluation Rigor**: The Official TCS Judge (referee LLM operating calibrated rubrics on $[0.0, 1.0]$, median of 3 replicates for borderline scores, certified pass at $\ge 0.50$, directional soundness at $\ge 0.40$).

---

## Section 2: Head-to-Head Performance & Empirical Scorecard

### 1. The TCS-300 Benchmark Results
* **Scorecard Comparison Table (Final Step 1 One-Pass Results)**:
  * **Gemini-3.8-Flash-High**: **101 / 300 Certified Passes (33.67%)** | **119 / 300 Directional (39.67%)** | Avg Duration: **192.2s (3.2 min)** | 0 Empty Timeouts.
  * **Meta Muse 1.3**: **66 / 300 Certified Passes (22.00%)** | **78 / 300 Directional (26.00%)** | Avg Duration: **488.0s (8.1 min)** | 186 Empty Timeouts.
  * **Qwen-3.8-27B (Throttled Baseline)**: 12 passes banked, averaging **~5.5 min/task** (down from 10+ min unconstrained), truncation rate dropped from 48.7% to 0%.

### 2. The Benchmark Inversion: Published Leaderboards (AIME, GPQA) vs. Real Research Proofs (TCS-300)
* **The Published Baseline Data**:
  * On the **AIME (American Invitational Mathematics Examination)**, Meta Muse Spark achieved a staggering **96.88%**, placing it neck-and-neck with OpenAI's GPT-5.2 (96.88%) and directly challenging Gemini 3.1 Pro Preview (98.13%).
  * On graduate-level reasoning evaluations (**GPQA Diamond**, **MATH-500**), massive frontier MoEs like Muse regularly dominate smaller models like Flash.
* **Why the Results Inverted on TCS-300**:
  * **Contest Math vs. arXiv Research Math**: AIME problems are isolated, self-contained puzzles with a single integer answer ($000$ to $999$). A reasoning model can wander in a 40,000-token symbolic loop and still hit the right number at the end.
  * **Research-Grade Lemma Proving**: TCS-300 theorems require absorbing 5 to 15 pages of novel academic definitions, constructing non-trivial reduction mappings, proving inductive invariant soundness, and formatting a publication-grade LaTeX `\begin{proof}` without unclosed logical gaps.
  * Under research conditions, **Muse's unconstrained reasoning often collapses into token exhaustion or 0-token stream aborts**, while Gemini Flash's disciplined focus yields **+53% more certified mathematical proofs**.

### 3. Wall-Clock Time, Velocity & Token Verbosity
* **Throughput Dynamics**:
  * Flash generated 300 research proofs in **~8.5 hours total wall-clock time** (with 4 workers finishing the last 90 tasks in 97.4 minutes).
  * Muse took **11.74 hours** for its one-shot run (with 186 empty timeouts), and its deep multi-turn rerun averages **~2.89 hours per task**.
  * Qwen on local RTX 5090 hardware saw its task solve times cut from ~10.0m to ~5.5m simply by capping its thinking budget.
* **Token Verbosity vs. Proof Quality**:
  * Why Qwen generating 140,000 characters of rambling scratchwork scored 0.08–0.15, while Flash generating 15,000 characters of clean, disciplined LaTeX scored 0.88–1.00.

---

## Section 3: Problem Topography — Where Each Model Excels
1. **Gemini 3.8 Flash**:
   * *Strong Suits*: Structural reductions, game-theoretic bounds, cleaner inductive definitions, and disciplined boundary-case discharging.
   * *Velocity Advantage*: Rapidly exhausts combinatorial searches without hallucinating recursive cycles.
2. **Meta Muse 1.3**:
   * *Strong Suits*: Intricate asymptotic bounds, complex analysis, and highly non-standard algebraic machinery when guided by multi-turn harness interaction.
   * *The Fragility Factor*: In zero-shot mode, it often blanks out on unfamiliar notation unless saved by backoff.
3. **Qwen-3.8-27B**:
   * *Strong Suits*: Local combinatorial expansions, direct algorithmic verification, matrix algebra, and modular arithmetic.
   * *The Trap*: Gets seduced into manual polynomial or matrix expansions inside thinking tokens rather than delegating to SymPy/Python.

---

## Section 4: Cognitive Dissections — Three Radical Philosophies of "Thinking"
A breakdown of the three distinct thinking architectures encountered in the benchmark:

### 1. Qwen-3.8-27B: The Runaway Philosopher (Why It Must Be Throttled)
* **The Shared Token Dilemma**: In open-weights autoregressive reasoning models, thinking and generation share the single `max_tokens=65536` stream (`<think> ... </think> \begin{proof} ... \end{proof}`).
* **The Overthinking Trap**: Left unconstrained, Qwen spends 45,000+ tokens arguing with itself (*"Let's invent a drift lemma"*, *"No, this is too fabricated"*, *"Alternative final proof..."*).
* **The Truncation Cliff**: Hitting token 65,536 mid-sentence (`Use Lemma \end{proof}`), leaving the judge with an empty carcass.
* **The Solution**: Hardware-level and prompt-level throttling via `TCS_SOLVER_THINKING_BUDGET=32768` (or 40% cap), forcing `</think>` before the cliff and leaving 30k+ tokens to write the actual proof.

### 2. Meta Muse 1.3: The Deep Diver (Why It Demands an Adaptive Backoff Ladder)
* **The 0-Token Drop Syndrome**: Under heavy compute and complex contexts, extreme reasoning effort (`xhigh`) frequently leads to stream exhaustion, HTTP chunk drops, or 0-token empty responses (`finish_reason='stop'`).
* **The Step-Down Ladder**: Engineering the automatic backoff cascade:
  $$\text{xhigh} \;\longrightarrow\; \text{high} \;\longrightarrow\; \text{medium} \;\longrightarrow\; \text{low}$$
* **The Recovery Dividend**: How this single backoff mechanism rescued 55+ previously "dead" problems, lifting Muse from 22.0% in raw one-shot to 44.7% in the combined funnel.

### 3. Gemini 3.8 Flash: The Disciplined Sprinter (Self-Throttling Architecture)
* **Internal Self-Regulation**: Despite running at `reasoning_effort=high`, Flash naturally concludes its reasoning phase within ~2,000 to 4,000 tokens (~60–90 seconds) and transitions immediately to synthesis.
* **Stream Density**: Streaming publication-grade LaTeX at 300+ tokens/sec with near-zero dropped connections or rate-limit friction.
* **Zero Overhead**: Does not need client-side token throttling; the model natively balances reasoning depth against output conciseness.

---

## Section 5: The Harness Governance Layer — Ceilings, Deadlines & Parity
1. **The 3-Hour Ceiling Paradox**:
   * Why we set `TCS_SOLVER_TIMEOUT_SECONDS=10800.0` (3.0 hours) as an overarching multi-turn ceiling.
   * Empirical finding: Across 127 Muse passes, **zero passes ever took > 3.0 hours**, but cutting the ceiling to 1.5 hours would have killed 21 deep, valid passes (16.5% of total wins).
2. **Scientific Parity**:
   * Giving Flash, Muse, and Qwen identical ceilings ensures fairness.
   * Faster models simply finish in minutes without ever touching the ceiling, while complex multi-turn solvers have the runway to finish.
3. **The Runway Extensions**:
   * Overtime gates (+4 turns, +1,800s for $\ge 0.35$ directional proofs).
   * Unconditional synthesis runway (+1,800s) to prevent deep runs from choking during final typesetting.

---

## Section 6: Conclusion — The MoE Divergence & The Future of Frontier Proving

### 1. The Tale of Two Massive MoEs: When 96.9% AIME Meets Reality
* **The Scale vs. Reliability Paradox**:
  Both Meta Muse and Gemini Flash represent the zenith of modern Mixture-of-Experts (MoE) engineering. On paper and on high-school contest benchmarks (AIME: 96.88%), Meta Muse appears invincible. Yet in autonomous mathematical proving, **Gemini Flash delivers a +53% advantage in one-shot pass rate (101 vs. 66 passes) in a fraction of the time**.
* **Why Leaderboards Lie About Proving**:
  Standard benchmarks reward models that can engage in endless internal speculation until a single numerical token emerges. But in research science, **the proof is the product**. If an MoE model cannot govern its expert routing and output stream under deep context pressure, it drops the connection, hits a token ceiling, or runs off the rails.
* **The Real Lesson of Meta Muse**:
  Muse is not "dumber" than Flash—its 96.9% AIME score proves its latent mathematical capacity is staggering. Rather, **Muse is an unguided missile without a harness**. When paired with our multi-turn harness and the adaptive step-down ladder (`xhigh -> high -> medium -> low`), Muse's pass count surged from 66 to **134 passes (44.67%)**. It simply cannot survive in autonomous mode without external governance.

### 2. The Three Architectural Archetypes
Our 300-task benchmark establishes three definitive operational profiles for frontier reasoning models:
1. **The Self-Regulating MoE (Gemini 3.8 Flash)**:
   Naturally converges on proofs in 2,000–4,000 thinking tokens, streams at 300+ tok/s, and requires zero client-side token throttling. The uncontested champion for high-throughput baseline triage.
2. **The High-Entropy Powerhouse (Meta Muse 1.3)**:
   Immense mathematical depth that easily destabilizes into 0-token stream drops or infinite loops. Requires continuous external scaffolding (backoff ladders and multi-agent refereeing) to extract its value.
3. **The Unconstrained Open-Weights Reasoner (Qwen-3.8-27B)**:
   Autoregressively shares thinking and generation tokens. Will burn 65k tokens debating trivialities unless rigidly throttled with hardware token budgets (`TCS_SOLVER_THINKING_BUDGET`), which instantly cuts solve time in half and rescues lost proofs.

### 3. The 2-Step Funnel as the Universal Proving Standard
* The ideal proving architecture does not pit these models against each other as rivals; it organizes them in a strict cognitive hierarchy:
  * **Stage 1 (Flash)**: Clear 35% of the benchmark in minutes at near-zero cost.
  * **Stage 2 (Harness + Muse/Qwen)**: Escalate the remaining 65% of unsolved problems into multi-turn cognitive search, where deep thinking, REPL verification, and adversarial debate can grind down genuine open problems.
* **The Final Rule of Autonomous Reasoning**:
  *Never let an open-weights reasoner think without an explicit budget, and never run a frontier proprietary reasoner without an adaptive backoff ladder.*

---

## Appendix & Artifacts
* Reproducibility repository links and configuration files (`.env.qwen`, `.env.agy_flash`, `.env.example.muse`).
* Full LaTeX proofs of notable benchmark breakthroughs (e.g. Task 234, Task 250, Perrenet lower bound frontier).
* Raw runtime logs, token distributions, and TCS referee audit scorecards.
