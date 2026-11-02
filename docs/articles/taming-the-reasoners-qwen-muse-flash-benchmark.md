# Taming the Reasoners: Lessons from Benchmarking Qwen-3.8-27B, Meta Muse 1.3, and Gemini 3.8 Flash on Research-Level Mathematics

### *An empirical deep-dive into reasoning dynamics, token expenditure, and harness governance across 300 Theoretical Computer Science frontier proofs.*

---

## Executive Summary & Narrative Hook
* **The Premise**: As frontier reasoning models ("thinking LLMs") become standard in AI research, raw benchmark claims obscure real-world execution realities. Putting three fundamentally different reasoning engines—**Qwen-3.8-27B** (local weights on RTX 5090), **Meta Muse 1.3** (`muse-spark-1.3-contributor`), and **Google Gemini 3.8 Flash** (`gemini-3.8-flash-high`)—through the grueling **TCS-300** benchmark reveals radically distinct cognitive personalities.
* **The Discovery**: Model intelligence is not just a function of parameters or leaderboard accuracy; it is governed by **how the model thinks, when it stops thinking, and how the harness manages runaway reasoning loops**.
* **The Central Thesis**:
  * Qwen is prone to runaway self-debate and requires hard external throttling.
  * Muse is deep but brittle under extreme compute, requiring adaptive step-down backoff to keep from dropping off the rails.
  * Flash demonstrates internal self-governance and extreme generation velocity, making it an unprecedented triage primitive.

---

## Section 1: The Arena — TCS-300 Benchmark & The 2-Step Funnel
1. **The Testbed**: 300 publication-grade mathematical lemmas and theorems extracted directly from arXiv preprints in Theoretical Computer Science (TCS), spanning graph theory, quantum complexity, circuit lower bounds, and randomized algorithms.
2. **The 2-Step Funnel Architecture**:
   * *Step 1: Fast Unassisted One-Pass Triage ($T=1$)*: Filter out the "low-hanging" frontier proofs with pure, zero-scaffolding generation.
   * *Step 2: Multi-Turn Cognitive Harness Escalation ($T \le 8$ or $12$)*: Engage persistent Python REPL kernels, multi-agent debate (Architect, Falsifier, Skeptic, Scribe), and the Academic Proof Auditor only on what Step 1 failed to solve.
3. **Evaluation Rigor**: The Official TCS Judge (referee LLM operating calibrated rubrics on $[0.0, 1.0]$, median of 3 replicates for borderline scores, certified pass at $\ge 0.50$, directional soundness at $\ge 0.40$).

---

## Section 2: Head-to-Head Performance & Empirical Scorecard
*(To be populated with finalized benchmark figures upon completion)*

* **Scorecard Comparison Table**:
  * Metrics: Target Total (300), Certified Passes ($\ge 0.50$), Directional Soundness ($\ge 0.40$), Solve Success Rate.
  * Side-by-side: Qwen-3.8-27B vs. Meta Muse 1.3 vs. Gemini-3.8-Flash-High.
* **Wall-Clock Time & Throughput**:
  * Average solve time per task (Flash: ~3.2m | Qwen: ~5.5m throttled vs. 10m unthrottled | Muse: ~8m 1-shot, ~2.9h multi-turn).
  * Effective throughput per hour across concurrent worker pools.
* **Token Expenditure & Verbosity**:
  * Proof size vs. Reasoning token volume.
  * Why generating 140,000 characters of proof often scores *lower* than a disciplined 15,000-character proof.

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

## Section 6: Key Takeaways & The Future of Frontier Proving
1. **Leaderboards vs. Harness Reality**: Why static eval numbers miss the operational dynamics of token ceilings, streaming stability, and thinking budgets.
2. **The 2-Step Funnel as Standard Practice**: Pair a high-speed, self-throttling reasoner (Flash) for wide triage with a deep, heavy-duty cognitive harness (Muse/Qwen) for the stubborn residue.
3. **The Golden Rule of Thinking Governance**: *Never let an open-weights reasoner think without an explicit budget, and never run a frontier proprietary reasoner without an adaptive backoff ladder.*

---

## Appendix & Artifacts
* Reproducibility repository links and configuration files (`.env.qwen`, `.env.agy_flash`, `.env.example.muse`).
* Full LaTeX proofs of notable benchmark breakthroughs (e.g. Task 234, Task 250, Perrenet lower bound frontier).
* Raw runtime logs, token distributions, and TCS referee audit scorecards.
