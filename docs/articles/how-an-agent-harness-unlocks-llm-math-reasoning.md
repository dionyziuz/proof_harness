# How an Agent Harness Unlocks Mathematical Reasoning in a 4-Bit 28B Model

### *Why raw LLMs struggle with research-level math, and how a zero-trust code kernel and vector memory boosted pass rates nearly 6x on consumer hardware.*

---

## The Illusion of the Fluent Mathematician

If you ask an off-the-shelf Large Language Model to solve a research-level math problem, you will usually get back an answer that *looks* brilliant. It uses the right terminology, cites standard theorems, and writes elegant LaTeX. 

Then you check line 4.

Somewhere in the middle of the derivation, an inequality quietly flipped direction. Or a random variable was assumed to stay independent after conditioning. Or a graph property was claimed to hold for all $n$, but fails completely on a 4-vertex cycle. 

Because autoregressive language models predict the next most likely token rather than verifying truth, they excel at sounding like mathematicians while routinely hallucinating algebraic leaps.

The conventional industry response to this challenge has been simple: **make the model bigger**. Throw a multi-hundred-billion-parameter frontier model on a cloud cluster at the problem and hope sheer scale solves the reasoning gap.

We wanted to explore a different question: **What if the bottleneck isn't raw parameter scale, but the harness around the model?**

To test this, we took an open-weight, 4-bit quantized 28B model (`unsloth/Qwen3.8-27B-NVFP4`) running locally on a single consumer GPU (an NVIDIA RTX 5090 with 32GB VRAM), and evaluated it on **300 research-level problems from TCS-Bench**—spanning spectral graph theory, quantum computing, parameterized complexity, and combinatorial algorithms.

When asked to draft proofs unassisted in a single pass, the 27B model struggled, passing only **4.0%** of problems. 

Then we wrapped that exact same 4-bit model inside **Proof Harness**—a specialized neuro-symbolic agent architecture. 

The results were striking:

| Evaluation Tier | Unassisted First Draft (27B) | Full Proof Harness (4-Bit 28B) | Multiplier |
| :--- | :---: | :---: | :---: |
| **All Passing Proofs (Grade C & Up, $\ge 0.40$)** | 4.0% (12 / 300) | **23.7% (71 / 300)** | **5.92× Gain** |
| **Certified Sound Proofs (Grade B & Up, $\ge 0.50$)** | 3.3% (10 / 300) | **21.0% (63 / 300)** | **6.30× Gain** |
| **Publication-Grade Proofs (Grade A & Up, $\ge 0.70$)** | 2.3% (7 / 300) | **18.0% (54 / 300)** | **7.71× Gain** |

Nearly **6x as many passing proofs**, with 18% reaching publication-grade quality—all running on consumer hardware without fine-tuning a single model weight. (While our empirical results were benchmarked on consumer hardware, the harness itself is model-agnostic and can wrap any LLM).

Here is how the harness works and the three core architectural decisions that made it possible.

---

## 1. Pillar I: The Zero-Trust Deterministic Kernel

The first rule of mathematical agents: **never trust an LLM's prose.**

If a language model claims that an inequality $f(n) \le g(n)$ holds for all positive integers, asking the model *"Are you sure?"* is useless. It will simply hallucinate a polite explanation of why it is sure.

Instead, Proof Harness pairs the LLM with a stateful, airgapped Python execution environment equipped with symbolic algebra (`SymPy`), SMT constraint solvers (`Z3`), arbitrary-precision arithmetic (`mpmath`), and graph analysis tools (`NetworkX`).

Whenever the model formulates an intermediate claim, it is forced to verify it in code:

- **Instant Counterexample Search**: Before spending thousands of tokens drafting a 5-page proof around an unverified lemma, the kernel sweeps small canonical instances (cycles, stars, bipartite graphs, boundary values). If a counterexample exists, the conjecture is killed in milliseconds on Turn 1.
- **Asymptotic & Covariance Checking**: Instead of letting the model guess whether $f(n) = \mathcal{O}(g(n))$ or assume two conditioned events remain independent, preloaded diagnostic tools compute exact symbolic limits and run covariance tests to catch subtle dependencies.
- **Empirical Falsification**: If an inequality holds on 2,000 random numerical samples, the kernel reports that no counterexample was found—but explicitly reminds the model that empirical survival is *evidence*, not a formal proof.

By treating every mathematical statement as guilty until proven innocent, the agent catches flawed logic early rather than compounding errors.

---

## 2. Pillar II: Learning by Solving (Vector Memory Without Fine-Tuning)

Standard LLMs suffer from **episodic amnesia**. An agent might spend 15 minutes and 10 iterations figuring out a clever reduction or uncovering a tricky boundary condition, only to forget everything the second the session ends. On the very next problem, it starts from scratch and repeats the exact same mistakes.

Fine-tuning model weights after every problem isn't practical—it is slow, computationally expensive, and frequently causes catastrophic forgetting.

Proof Harness solves this by implementing **continuous domain learning through vector embeddings**:

1. **Strategy Extraction**: When a problem is solved (or when a failure is analyzed), a low-temperature reflection pass distills verified mathematical strategies into concise heuristics (e.g., *"When bounding high-dimensional sumsets, dyadic decomposition prevents recursive dimension blowup"*).
2. **The Anti-Cheating Gate**: A strict validator scrubs problem-specific variables, expected answers, and hardcoded numbers. Only generalizable, universal strategies are allowed through.
3. **Role-Partitioned Retrieval**: Learned heuristics are stored in local vector memory. When the agent later faces a new problem in the same mathematical domain, it retrieves relevant strategies based on semantic similarity.

The result is a system that **compounds expertise over time**. The base model's weights remain completely untouched, but the agent's effective mathematical knowledge grows with every problem it solves.

---

## 3. Pillar III: Cognitive Roles & The Turn-1 Exploration Gate

When humans tackle hard research math, we don't immediately start writing the final proof on page one. We sketch diagrams, test small cases on scratch paper, look for edge cases, and outline potential approaches.

Naive AI agents do the exact opposite: they immediately start generating `\begin{proof}` on Turn 1. Once an LLM commits to an initial algebraic path, it falls into a confirmation bias trap—papering over mistakes with hand-waving arguments rather than backtracking.

To prevent this, Proof Harness divides the proving process across specialized cognitive roles and enforces a strict **Turn-1 Exploration Gate**:

- **Turn 1 is for Exploration Only**: The model is strictly banned from drafting the formal proof on Turn 1. Instead, it must inventory known facts, query vector memory for related strategies, formulate testable conjectures, and run code in the kernel.
- **Role Specialization**: 
  - The **Context Architect** sketches high-level reduction blueprints.
  - The **Logician** tests steps iteratively in the Python REPL.
  - The **Adversarial Skeptic** actively searches for edge-case counterexamples.
  - The **Publication Scribe** writes the formal, rigorous LaTeX only after the mathematical foundation is verified.
- **Stateless Blackboard Revisions**: If an internal referee finds a flaw in a draft, the harness doesn't keep a massive, noisy chat history that confuses the model. It collapses the context into a clean, focused blackboard: the problem statement, the best draft so far, and the specific surgical repair directive.

In our 300-problem benchmark, **100% of all passing proofs were discovered through this governed multi-turn process**. Over 84% of wins were achieved in Turn 4 or later, proving that deep iteration—guided by deterministic tools—is essential for hard reasoning.

---

## Key Takeaways for AI Engineers

Whether you are building agents for mathematics, software engineering, quantitative finance, or scientific research, the architectural lessons from Proof Harness apply broadly:

1. **Don't trust language models to verify their own prose.** Connect your agent to deterministic ground truth—code interpreters, compilers, linters, or SMT solvers.
2. **Enforce an exploration phase.** Never let an agent jump straight to the final output on Turn 1. Force it to inspect inputs, test assumptions, and explore the search space first.
3. **Use vector memory for fast, weight-free learning.** Distilling lessons into curated, validated vector embeddings lets your agents get smarter over time without the cost and instability of continuous model fine-tuning.
4. **Isolate generation from critique.** Have separate roles for drafting solutions and finding flaws. An agent asked to grade its own work will almost always be overly optimistic.
5. **Keep multi-turn context clean.** Long conversation histories accumulate noise. Summarize or reset to a clean "blackboard" containing only the current best candidate and the exact failure to address.

---

## Looking Forward

The frontier of AI reasoning isn't just about training larger foundation models. It is about building the cognitive scaffolding that enables models to think, verify, and learn effectively.

By pairing a lightweight, 4-bit open-weight model with a zero-trust execution kernel and continuous heuristic memory, Proof Harness demonstrates that consumer hardware can tackle problems that previously seemed out of reach.

*For implementation details, evaluation scripts, and the full benchmark harness, check out the [Proof Harness repository on GitHub](https://github.com/dionyziuz/proof_harness). For theoretical analysis and comprehensive ablation studies, our full technical paper is available in the repository documentation.*
