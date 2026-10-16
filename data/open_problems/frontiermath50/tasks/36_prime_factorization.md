# Improve the Best-known Classical Factorization Algorithms

- **Problem ID**: `prime-factorization`
- **Field**: Number theory
- **Notability**: Breakthrough
- **Tags**: Algorithm
- **Prompt Type**: `full_problem`
- **Status**: Open

---

## Problem Statement & Prompt

# Factoring semiprimes faster than the best known classical algorithms

You will be given semiprimes $N = p \cdot q$ to factor, and scored on whether you do so
*faster than the best known classical method could* — by a margin large enough that any
positive score is very likely to be an algorithmic breakthrough, not just a fast
implementation or beefier hardware.

## Submission format

You will submit a single Python file. It may use `sage`. When submitting, you may also
constrain the challenges your program will receive by specifying `logp` and/or `logq`:

| parameter | meaning |
|-----------|---------|
| `logp` | $p$ is chosen with $2^{logp} < p < 2^{logp+1}$ |
| `logq` | $q$ is chosen with $2^{logq} < q < 2^{logq+1}$ |

Both must be between 3 and 3000. If `logp` is given but not `logq`, $q$ is chosen from
the same range as $p$. If `logq` is given but not `logp`, then for each challenge a value
$k$ is chosen randomly with $\min(110, logq) \le k \le logq$ and $p$ is chosen with
$2^k < p < 2^{k+1}$. If neither is given, for each challenge a random $k$ between 110 and
3000 is chosen and $p$ and $q$ are drawn independently from $(2^k, 2^{k+1})$.

## Protocol

The verifier repeatedly:

1. picks a challenge semiprime $N = p \cdot q$ subject to your constraints;
2. runs your program in a fresh process, with `N` in decimal as the sole command-line
   argument, using the same Python interpreter that runs the verifier. That is Sage's
   Python, which is not necessarily the `python` on your `PATH` — find it with
   `sage -c 'import sys; print(sys.executable)'` and test your submission with that
   interpreter, or `import sage` may fail. The wall-clock time of the whole invocation
   is measured, including interpreter startup;
3. reads the last non-empty line of your program's stdout as your answer: the two prime
   factors of `N`, space-separated. Earlier lines of stdout are ignored, so printing
   progress information is fine.

Your program is run on one challenge after another until a total wall-clock budget of 30
minutes is exhausted (at most 20 challenges). Each invocation may use whatever remains of
the 30 minutes, so a program that spends the whole budget on one instance gets exactly
one attempt. If you want to reuse expensive precomputation across challenges, cache it in
a file.

## Scoring

For each correct factorization, the verifier estimates the wall-clock time the best known
classical method would have needed for an instance of that shape — ECM (governed by the
smaller factor) for unbalanced $N$, SIQS/GNFS for balanced $N$ — using a calibrated cost
model of the standard $L$-notation complexities. Your score is the speedup
`classical_time / time_spent`, and it is 0 unless it clears a breakthrough cutoff:

```
cutoff = parallel_allowance · calibration_uncertainty · substantial_speedup
       = 16 · 10 · 1000 ≈ 1.6e5
```

The modeled classical time is single-threaded; `parallel_allowance` discounts the cores
your program may use, `calibration_uncertainty` covers hardware/implementation/model
error, and `substantial_speedup` is the genuine algorithmic gain we insist on. So a
positive score means you factored $N$ over 160000x faster than the best known classical
method plausibly could on this machine. Choose instance sizes large enough that this is
possible at all: an instance the model prices at a few seconds cannot clear the cutoff no
matter how fast you factor it.

If there is sufficient time available, your program may be run on multiple challenges.
It will be considered correct if it clears the cutoff on any challenge within the
available time.

## Example submission format

```
import sys

from sage.all import ZZ, factor

N = ZZ(sys.argv[1])
F = factor(N)
# Your answer: the two prime factors of N
print(" ".join(str(p) for p, e in F))
```
