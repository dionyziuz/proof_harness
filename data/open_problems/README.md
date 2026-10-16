# Open Problems in Mathematics Benchmark

This directory contains the 50 open problems in mathematics benchmark created by Epoch AI and highlighted in *Scientific American* ("Mathematicians Name 50 of the Highest-Stakes Problems in Math").

Unlike standard closed benchmarks (such as Project Euler or typical competition math), these problems have **no documented ground-truth solution**. Progress represents genuine advances in mathematical research.

---

## Directory Organization

```
data/open_problems/
├── README.md                  # This overview and usage guide
├── frontiermath50/            # The 50 Epoch AI FrontierMath Open Problems
│   ├── metadata.csv           # Full metadata (field, notability, tags, attribution, status)
│   ├── prompts.csv            # Canonical natural language mathematical prompts
│   ├── problems.json          # Combined JSON schema of all 50 problems
│   └── tasks/                 # 50 individual problem task specifications (.md)
└── lean/                      # 50 Lean 4 formal open mathematical conjectures
    ├── manifest.json          # Index of formal Lean statements and targets
    ├── Erdos1.erdos_1.lean    # Isolated Lean 4 formalization of Erdős Problem 1
    ├── ...                    # All 50 formal Lean 4 open problem files
    └── Erdos1057.erdos_1057.lean
```

---

## 1. FrontierMath 50 (`frontiermath50/`)

Selected by leading mathematicians to test the limits of automated mathematical reasoning:
- **Domains**: Number theory, Diophantine equations, Combinatorics, Graph theory, Arithmetic geometry, Social choice, Discrete geometry, Knot theory, Analysis.
- **Generator-Verifier Gap**: Candidate solutions (finite constructions, counterexamples, algorithm designs, algebraic witnesses) are computationally or logically checkable once found.
- **Featured Conjectures**:
  - `apery-irrationality`: Apéry-style linear recurrences for irrationality of $\zeta(5), \dots$
  - `jones-unknot`: Counterexamples to the Jones unknot conjecture (nontrivial knot with $V(t) = 1$).
  - `lonely-runner`: The Lonely Runner Conjecture on circular tracks.
  - `sum-of-three-cubes`: Integer solutions for $x^3 + y^3 + z^3 = n$ ($n=3, 114$).
  - `committee-election` & `efx`: Core existence and fair allocation in social choice.

---

## 2. Lean 4 Formal Conjectures (`lean/`)

Contains 50 machine-verifiable open problems curated by Thomas Bloom and Epoch AI (`LeanOpenProblems`):
- **Verification Engine**: Lean 4 formal compiler and FRO proof checker (`prime_math.judging.lean_judge`).
- **Structure**: Each file contains:
  1. Required definitions and hypotheses formalized in Mathlib / Lean 4.
  2. The target conjecture theorem ending in `:= by sorry`.
  3. The target disproof theorem (`<id>.disproof : ¬ (type_of% ...) := sorry`).
- **Goal**: An agent must either prove the conjecture or prove its negation without using `sorry` or introducing unapproved axioms.

### Compiling and Proving with Lean Judge
To verify candidate Lean proofs:
```python
from prime_math.judging.lean_judge import LeanJudge

judge = LeanJudge()
result = judge.judge_sync(
    proof_code=candidate_lean_code,
    formal_sketch=lean_spec_text,
    task_id="Erdos1"
)
print("Formally Verified:", result.accepted)
```
