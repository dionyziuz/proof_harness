# Elliptic Curves with Prescribed Number of Points

- **Problem ID**: `elliptic-curve-point-count`
- **Field**: Arithmetic geometry
- **Notability**: Solid result
- **Tags**: Algorithm
- **Prompt Type**: `full_problem`
- **Status**: Open

---

## Problem Statement & Prompt

# Elliptic curves with a prescribed point count

Given a prime $p$ and a target $N$ in the Hasse interval
($|N - (p+1)| < 2\sqrt{p}$), construct an elliptic curve $E/\mathbb{F}_p$ with
exactly $\#E(\mathbb{F}_p) = N$. The verifier deliberately chooses $(p, N)$ so that
the CM discriminant $D = (p+1-N)^2 - 4p$ has a large squarefree part.

## Submission format

You will submit a single Python file. It may use `sage`. When submitting, you may
also constrain the challenges your program will receive by specifying at most one
of the following (noting the "interestingness" criterion below):

| parameter | meaning |
|-----------|---------|
| `s` | $p$ chosen with $2^s < p < 2^{s+1}$; $s$ between 7 and 300 |
| `p` | you fix the prime $p$ (must be prime, $\ge 128$) |
| `N` | you fix the cardinality $N$ ($\ge 128$); the verifier picks a compatible $p$ |

If none is specified, each challenge is a random instance over the whole range.

## Protocol

The verifier repeatedly:

1. picks a challenge $(p, N)$ subject to your constraint;
2. runs your program in a fresh process with `p` and `N` in decimal as the two
   command-line arguments, using the same Python interpreter that runs the verifier.
   That is Sage's Python, which is not necessarily the `python` on your `PATH` — find
   it with `sage -c 'import sys; print(sys.executable)'` and test your submission with
   that interpreter, or `import sage` may fail;
3. reads the last non-empty line of your program's stdout as your answer: the
   $a$-invariants of a curve $E/\mathbb{F}_p$ with $\#E = N$, either `a b` (for
   $y^2 = x^3 + ax + b$) or `a b c d e` (general Weierstrass form), space-separated.
   Earlier lines of stdout are ignored, so printing progress information is fine.

Your program is run on one challenge after another until a total wall-clock budget of 30
minutes is exhausted (at most 20 challenges). Each invocation may use whatever remains of
the 30 minutes, so a program that spends the whole budget on one instance gets exactly
one attempt. If you want to reuse expensive precomputation across challenges, cache it in
a file.

## Scoring

A solution counts when the returned curve really has $N$ points. Only solves with
$p$ above the size threshold of $2^{80}$ are "interesting".

If there is sufficient time available, your program may be run on multiple challenges.
It will be considered correct if it answers an "interesting" challenge correctly within
the available time.

## Example submission format

```
import sys

from sage.all import ZZ

p, N = ZZ(sys.argv[1]), ZZ(sys.argv[2])
curve = find_curve(p, N)   # you must implement this!
# Your answer: the a-invariants of the elliptic curve,
# either as a pair a,b for y^2 = x^3 + ax + b
# or a 5-tuple a,b,c,d,e for y^2 + axy + cy = x^3 + bx^2 + dx + e
print(" ".join(str(a) for a in curve))
```
