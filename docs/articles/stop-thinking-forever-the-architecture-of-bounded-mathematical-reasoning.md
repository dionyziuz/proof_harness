# Stop Thinking Forever: The Architecture of Bounded Mathematical Reasoning

### *How configurable constraints, adversarial auditing, and human-in-the-loop checkpoints keep frontier reasoning models focused and converging on hard problems.*

---

## The Long-Horizon Challenge: When Extended Thinking Meets Hard Problems

Long-thinking frontier models have fundamentally transformed automated reasoning. As our own empirical benchmarks on TCS-Bench show, given generous thinking budgets, models like Google Gemini 3.8 Flash, Meta Muse 1.3, and Qwen 2.5-Math can solve an impressive fraction of complex research problems directly in one shot—deriving intricate bounds and complete, publication-grade LaTeX proofs without external assistance.

However, when an advanced model encounters a truly stubborn mathematical obstruction or an open research conjecture, scaling test-time compute with an open-ended token budget does not always guarantee a breakthrough. In fact, without structural guardrails, it can enter an extended cycle of cognitive perseveration.

We have all observed this phenomenon: an autonomous run spins up, analyzes a difficult theorem, posits an ambitious reduction, and hits a subtle mathematical impasse. Because the model's own prior derivation sits directly in its context prefix, it can spend tens of thousands of tokens exploring minor variations of an intermediate lemma that collapsed early on—steadily consuming compute budgets without converging on a sound resolution.

Left unconstrained on the hardest problem cohorts, extended reasoning can suffer from three structural challenges:
1. **The Sunk-Cost Loop**: Once a model commits to a flawed intermediate lemma, it often treats that premise as foundational, spending extended tokens attempting to patch or rescue it rather than taking a step back.
2. **Context Window Degradation**: As exploratory scratchpad reasoning balloons toward context limits, standard token compaction mechanisms can inadvertently compress critical definitions or premises, degrading the downstream derivation.
3. **The Multi-Day Black Hole**: Without clear bounds and stopping criteria, an autonomous agent exploring open research conjectures can run for days without emitting inspectable interim artifacts or decision gates for human researchers.

At **Proof-Harness (`proof_harness`)**, we designed an autonomous neuro-symbolic engine capable of tackling research-grade preprints from arXiv Theoretical Computer Science (**TCS-Bench 300**). In our technical papers, we frequently discuss our first two architectural pillars:
* **The Deterministic Mathematical Kernel**: Airgapped Python REPL execution using `SymPy`, `Z3`, 50-digit `mpmath`, and an epistemic state machine that ruthlessly verifies equations instead of trusting next-token probabilities.
* **The CoALA Continuous Domain Memory**: Vectorized procedural and episodic heuristics that allow the system to compound specialized domain intelligence across runs without weight fine-tuning.

**But those two pillars alone are not enough.** A perfect execution kernel and infinite vector memory will still happily run in circles forever if the reasoning harness lacks **configurable constraints**. 

True mathematical problem-solving is not an unconstrained random walk through theorem space; it is a **bounded, adversarial search under resource limits**. 

This article explores the third, unheralded pillar of Proof-Harness: **The Configurable Bounded Constraint & Human-in-the-Loop (HITL) Governance Engine**.

---

## 1. The Anatomy of Unconstrained Failure: Why Reasoners Need Leashes

To understand why constraints must be hard-coded into the harness architecture, consider what happens when a model like Qwen-27B or Meta Muse attacks a research-grade problem without a computational leash.

```text
  [Unconstrained Reasoning Trap]
  
  Theorem Input ──► Deep Thinking ──► Flawed Lemma ──► REPL Refutation
                          ▲                                  │
                          │                                  ▼
                     Rethink / Rescue ◄── Justify Flaw ◄── Denial
                          │
                   (Repeats for 12 hours / 150k tokens / Context overflow)
```

In standard chat or agent wrappers, the prompt essentially says: *"Prove this theorem. Think as long as you need."*

When faced with a genuinely difficult problem—such as establishing an asymptotic expansion bound on a $d$-regular graph or proving an $\exists\mathbb{R}$-hardness reduction—the model encounters mathematical obstructions. But unlike a human mathematician who stands up, takes a walk, erases the blackboard, and starts with an entirely new reduction, the LLM is trapped by its own autoregressive prefix. 

The flawed derivation is right there in its context window. The model's attention heads attend to its own prior arguments. It enters a state of **epistemic fixation**:
* It misapplies the Lovász Local Lemma because $c_3 > \ln 2$ is false.
* The auditor flags the defect.
* Instead of throwing away the Local Lemma, the model introduces an ad-hoc sub-case.
* The sub-case violates the degree condition.
* The model introduces an ad-hoc graph partition to fix the degree condition.
* The partition breaks independence.
* Four hours later, the model is still trying to salvage a mathematical premise that was dead on arrival.

To prevent these unproductive loops on difficult problems, an autonomous harness acts as a **disciplined, time-budgeted research director**. It enforces strict token leashes, bounded iteration limits, in-the-loop adversarial auditing, automated cognitive pivots, and human-in-the-loop checkpoints.

---

## 2. Pillar III: The Bounded Constraint Architecture

The Proof-Harness governance substrate operates across five deeply integrated control surfaces:

```text
+========================================================================================+
|                       BOUNDED CONSTRAINT & GOVERNANCE SUBSTRATE                        |
+========================================================================================+
| 1. Token Leashes     | TCS_COMPACTOR_MAX_CONTEXT_TOKENS (48,000)                       |
|    & Reserve Buffers | TCS_COMPACTOR_RESERVE_TOKENS     (16,000)                       |
|                      | Prevents mid-derivation truncation & context blowouts           |
+----------------------+-----------------------------------------------------------------+
| 2. Iteration Budgets | Standard Iterations: Turns 1..8                                 |
|    & Overtime Gates  | Overtime Gate: Validity >= 0.35 required to enter Turns 9..12   |
|                      | Stops dead-end problems from running for days                   |
+----------------------+-----------------------------------------------------------------+
| 3. In-Loop Auditing  | Adversarial Auditor (Ephemeral Kernel)                          |
|    & Score Routing   |   • Validity < 0.40  ──► Structural Defect Reprimand            |
|                      |   • Validity >= 0.40 ──► Surgical Repair Mode (Preserve Lemmas) |
|                      |   • Validity >= 0.70 ──► Early Certified Exit                   |
+----------------------+-----------------------------------------------------------------+
| 4. Reductive Pivots  | Anti-Perseveration Engine: Purges refuted premises & kernel     |
|    (Anti-Looping)    | variables if validity <= 0.25 across consecutive iterations     |
+----------------------+-----------------------------------------------------------------+
| 5. Human-in-the-Loop | Granular JSON Checkpoints (`checkpoints/*.ckpt.json`)           |
|    State Persistence | Loss-free SIGINT pause, interim inspection, zero-loss resumption|
+========================================================================================+
```

Let's dissect each mechanism.

---

## 3. The Token Leash: Context Compaction without Information Loss

The first boundary is spatial: the context window.

Modern models support 32k, 64k, or 128k context windows, but context degradation is real. If the context window fills up with rambling intermediate scratchpad code, two disasters strike:
1. **The U-Shaped Attention Dip**: Crucial preconditions stated in the original problem prompt get "lost in the middle."
2. **Mid-Derivation Truncation (`finish_reason='length'`)**: The model hits the hard token ceiling right as it is typing the final proof step, causing an abrupt cutoff. The external judge sees an unclosed bracket or unfinished sentence and awards a zero.

Proof-Harness solves this through **asymmetric dual-buffer context compaction** (`compactor.py`):
```bash
TCS_COMPACTOR_MAX_CONTEXT_TOKENS=48000
TCS_COMPACTOR_RESERVE_TOKENS=16000
TCS_COMPACTOR_KEEP_RECENT_TURNS=2
```

Instead of letting context accumulate monotonically:
* The active context is hard-capped at **48,000 tokens**.
* A guaranteed **16,000-token reserve buffer** is strictly walled off for the final proof generation and synthesis stage.
* When the reasoning trace exceeds the boundary, the compactor executes an **epistemic extraction**: it preserves the original problem prompt verbatim, preserves all verified invariants in the `MathStateMachine`, retains the last 2 recent turns, and compacts earlier exploratory turns into a dense, symbolic summary of proved lemmas and refuted paths.

The model gets infinite conceptual runway without ever hitting a hard context wall.

---

## 4. The Iteration Funnel: 8 Standard Turns and the Overtime Gate

The second boundary is temporal: how many turns does an agent get before the system says *enough*?

In Proof-Harness, problem-solving is structured as a **staged tournament**:

```text
[ Problem Ingest ]
        │
        ▼
┌────────────────────────────────────────────────────────┐
│ STANDARD SOLVER: Turns 1 through 8                    │
│   • Turn 1: Zero-Shot / Exploration (No proof emitted) │
│   • Turns 2..7: REPL testing, lemma building           │
│   • Turn 8: Mandatory Candidate Proof Synthesis        │
└──────────────────────────┬─────────────────────────────┘
                           │
             [ Turn 8 Auditor Evaluation ]
                           │
         ┌─────────────────┴─────────────────┐
         ▼                                   ▼
 [ Validity < 0.35 ]                [ Validity >= 0.35 ]
         │                                   │
         ▼                                   ▼
┌─────────────────────────┐         ┌──────────────────────────────┐
│ FORCED TERMINATION      │         │ OVERTIME ESCALATION          │
│ • Cut losses            │         │ • Unlock Turns 9..12         │
│ • Submit best candidate │         │ • Focused defect resolution  │
│ • Save compute          │         │ • Turn 12 Hard Termination   │
└─────────────────────────┘         └──────────────────────────────┘
```

Why is the **Overtime Gate (`TCS_OVERTIME_VALIDITY_THRESHOLD=0.35`)** so crucial?

Empirically, if an AI agent has spent 8 multi-thousand-token iterations on a theorem and cannot even achieve a directional validity score of **0.35**, the probability of it magically discovering a sound proof in turns 9 through 12 is less than **2%**. 

Allowing an unguided agent to keep guessing past Turn 8 on a completely broken foundation is pure compute waste. The harness **cuts its losses immediately**, submits the best effort to the referee, and moves to the next problem.

Conversely, if the candidate achieves **0.45 validity** at Turn 8, the harness recognizes that the core architecture of the proof is sound and only specific lemmas or constants are flawed. It unlocks **Turns 9 through 12 (Overtime)**, providing the extra runway needed to land the proof.

---

## 5. Dual-Tier Referee Architecture: In-Loop Auditor vs. Zero-Trust Judge

A core innovation in Proof-Harness is the architectural separation between the **In-Loop Adversarial Auditor** and the **External Benchmark Judge**.

Many naive agent harnesses make the fatal mistake of letting the agent evaluate its own work ("Self-Correction"). LLMs are notoriously biased toward their own outputs; an agent that wrote a flawed proof will almost always convince itself that the proof is correct when asked to verify it.

Proof-Harness implements an **airgapped adversarial referee**:

### Tier 1: The In-Loop Adversarial Auditor (`prover_auditor.py`)
* Operates in an ephemeral execution sandbox after every single candidate proof draft.
* **Score-Gated Routing**:
  * **Validity < 0.40 (Structural Failure)**: The auditor emits detailed defect notices (`[ADVERSARIAL PROOF AUDIT - DEFECTS DETECTED]`), highlights unproved assertions, and instructs the solver to rethink its strategy.
  * **0.40 $\le$ Validity < 0.70 (Surgical Repair Mode)**: The harness locks all sound lemmas into an immutable blackboard (`Modular Lemma Immutability`) and strictly forbids full rewrites. It instructs the solver: *"Your previous candidate achieved strong validity (0.45). Do NOT rewrite the proof. Make targeted, surgical repairs directly to the specific defect flagged."*
  * **Validity $\ge$ 0.70 (Certified Sound)**: The proof is accepted early (`Auditor: VALID`). Iterations terminate immediately, saving hundreds of seconds of unnecessary computation.

### Tier 2: The External Benchmark Judge (`core_engine/judging/`)
* Completely isolated from the solving process.
* Evaluates the final candidate proof across **3 independent replicates ($k=3$)** using a zero-trust rubric:
  * Check precondition satisfaction.
  * Check whether constants and asymptotic hierarchies are explicitly supported.
  * Verify absence of circular reasoning and external hallucinations.
* The median of the 3 replicates determines the official benchmark verdict (**$\ge 0.50$ = Certified Pass**, **$\ge 0.40$ = Directionally Sound**).

This dual-tier separation ensures that the internal auditor acts as a coaching referee pushing the agent forward, while the external judge guarantees uncompromised academic integrity.

---

## 6. Cognitive Anti-Perseveration: The Mandatory Reductive Pivot

What happens when an agent gets stuck?

In our benchmark runs on TCS-Bench, we observed a fascinating failure mode: on **Task 31** (`task_2106.02397`), the model attempted an $\exists\mathbb{R}$-hardness reduction for a computational geometry problem (`CCIEXPL`). By Turn 4, the REPL kernel proved that its proposed reduction created an exponential blowup in coordinate precision, completely refuting the approach (validity: 0.20).

In an unconstrained setup, the model would spend the next 20 turns trying to patch the coordinate blowup.

Proof-Harness has an active **Anti-Perseveration Engine**:
```text
⚠️ MANDATORY REDUCTIVE PIVOT (Turn 4):
Your previous reduction attempt reached a fatal mathematical obstacle 
or false premise (validity: 0.20). 

All Python kernel variables, intermediate claims, and subgoals from 
this reduction path have been REFUTED AND PURGED from the state machine.

BANNED PREMISES / REFUTED ROUTES (DO NOT RE-ATTEMPT OR RESCUE):
  ❌ Refuted Reduction: Coordinate precision blowup under delta scaling.

Directive: Abandon this entire reduction branch. You are mathematically 
forbidden from attempting to rescue it. Select an alternative problem 
to reduce from, or pivot to an inductive structural argument.
```

The harness **actively purges the refuted branch**. It wipes the refuted variables from the `MathStateMachine`, explicitly posts the banned premises to the cognitive blackboard, and guides the model to pivot to an alternative mathematical direction.

It is the algorithmic equivalent of a senior research advisor taking away a student's broken whiteboard and saying: *"Stop. That route is dead. Start clean with approach B."*

---

## 7. Human-in-the-Loop (HITL) as a First-Class Citizen

Automated theorem proving is not just for automated benchmarks; it is built for **open research frontiers**. 

When a human mathematician uses Proof-Harness to attack an open conjecture in mathematical physics (like our runs on the Quantum Null Energy Condition on null Cauchy surfaces), the human cannot be an afterthought waiting for a script to finish next Tuesday. The human must be an active collaborator.

Proof-Harness treats **Human-in-the-Loop observability and intervention** as a core architectural primitive:

```text
       [ Live Autonomous Run ]
                 │
                 ├──► Live JSON Checkpoint (`checkpoints/task_xyz.ckpt.json`)
                 │      • Current turn & elapsed seconds
                 │      • Registered state machine objects & lemmas
                 │      • Prior auditor defects & validity scores
                 │
  [ Human Operator Sends SIGINT / C-c ]
                 │
                 ▼
       [ Graceful Suspension ]
                 │
                 ├──► Human inspects intermediate proof & auditor critique
                 ├──► Human validates whether partial lemma is promising
                 │
       [ Resume Pipeline (--resume-all) ]
                 │
                 ▼
       [ 100% Zero-Loss Instant Catch-up ]
```

### 1. Granular On-Disk JSON Checkpoints
After every turn, the harness writes a comprehensive checkpoint file (`checkpoints/{task_id}.ckpt.json`). This file captures:
* The current stage (`exploration`, `synthesis`, `audit`).
* The active `MathStateMachine` invariants and verified Python REPL facts.
* All prior auditor defects and scores.
* The exact conversation history and token counts.

### 2. Loss-Free Graceful Interruption
If a run needs to pause—whether to free up GPU VRAM for another job, reboot the machine, or wait for an unstable remote tunnel to recover—the operator can send a clean `SIGINT` (`Ctrl+C`). The harness finishes its atomic write, flushes logs, and suspends cleanly.

### 3. Instant Resume with Zero Recomputation
When relaunched with `--resume-all`, the pipeline:
1. Instantly scans `tasks/` and `judgments/`—skipping all completed problems in milliseconds without firing a single LLM token.
2. Identifies active checkpoints in `checkpoints/`.
3. Emits `>>> Resumed from checkpoint at stage='exploration', turn=4/8`.
4. Continues solving mid-flight with **zero lost progress**.

This allows human mathematicians to steer multi-day campaigns on open problems with total confidence, inspecting partial proofs at breakfast, tweaking domain vector heuristics at lunch, and letting the bounded solver crunch overtime overnight.

---

## 8. Real-World Case Studies from the Benchmark Trenches

Does this bounded constraint architecture actually work in practice? The empirical data from our 300-problem TCS-Bench runs provides a resounding answer.

### Case 1: The Overtime Clutch Win (Task 299)
* **Problem**: `task_2601.20718_statement_196_82`
* **Turn 1 (Unassisted Baseline)**: Validity **0.22 (FAIL)**.
* **Escalation**: The harness caught the failure, entered multi-turn solving, and climbed through turns 2 and 3.
* **The Overtime Breakthrough**: At Turn 4, the Auditor detected a sound invariant and scored it **0.92**. The proof was immediately accepted and emitted early.
* **Result**: **0.92 PASS** across all judge replicates. A problem that was completely dead in one-shot was converted into a near-perfect publication-grade pass in 960 seconds.

### Case 2: The Auditor-vs-Judge Reality Check (Task 20)
* **Problem**: `task_2101.05549_statement_226_137`
* **Turn 8**: Candidate scored **0.45** by the internal auditor, triggering Overtime.
* **Turns 9 through 12**: The model tried local patches but perseverated on a subtle eigenvalue gap scaling issue.
* **The Boundary Enforcement**: At Turn 12, the harness refused to loop indefinitely. It halted execution, locked the candidate, and submitted it to the external judge.
* **Verdict**: The external referee spotted the unproven eigenvalue bound and scored it **0.18 (FAIL)**. 
* **The Lesson**: Even when a model fails to cross the finish line, the harness **guaranteed termination within 7,040 seconds (~1.9 hrs)** rather than burning 48 hours in a runaway hallucination loop.

### Case 3: The Overnight Milestone (Gemini Flash on TCS-300)
By enforcing these exact bounded constraints—fast single-pass triage followed by bounded 12-turn multi-turn escalation—Google Gemini 3.8 Flash completed all 300 research problems in 36.8 hours, achieving:
* **Headline Certified Passes ($\ge 0.50$)**: **166 / 300 (55.33%)**
* **Directionally Sound ($\ge 0.40$)**: **184 / 300 (61.33%)**
* **Parse or Execution Errors**: **0**

---

## Conclusion: True Intelligence Requires Boundaries

In the early days of automated coding and reasoning, the prevailing assumption was simple: *more compute and longer thinking traces equal better answers*.

Our journey building Proof-Harness on research-grade mathematics proved the opposite. Without guardrails, raw frontier intelligence quickly collapses into cognitive perseveration, infinite loops, and multi-day black holes.

To build an AI that can legitimately assist human researchers on the hardest open problems in mathematics and theoretical physics, we cannot just build bigger models or give them infinite context. We must build **disciplined cognitive harnesses**:
* We must give them **spatial boundaries** through context leashes and reserve buffers.
* We must give them **temporal boundaries** through turn budgets and overtime gates.
* We must give them **epistemic boundaries** through adversarial in-the-loop auditors.
* We must give them **psychological boundaries** through mandatory reductive pivots.
* And we must keep the **human in the loop** through transparent, loss-free checkpoints.

When you tame the reasoning engine with the right constraints, it stops thinking forever—and starts actually solving.

---

*Proof-Harness is open-source under the MIT license at [github.com/dionyziuz/proof_harness](https://github.com/dionyziuz/proof_harness).*
