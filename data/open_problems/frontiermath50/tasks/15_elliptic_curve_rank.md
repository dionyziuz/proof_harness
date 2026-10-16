# Elliptic Curves over $\mathbb{Q}$ of Large Rank

- **Problem ID**: `elliptic-curve-rank`
- **Field**: Arithmetic geometry
- **Notability**: Moderately interesting
- **Tags**: Construction - Finite, Record Breaking
- **Prompt Type**: `full_problem`
- **Status**: Solved (2026-08-20)

---

## Problem Statement & Prompt

An elliptic curve E over the rationals can be written in the general Weierstrass form

    E : y^2 + a1 x y + a3 y = x^3 + a2 x^2 + a4 x + a6,   a1, a2, a3, a4, a6 in Z.

By the Mordell-Weil theorem its group of rational points is finitely generated,

    E(Q) ≅ E(Q)_tors ⊕ Z^r,

and the non-negative integer r is the rank of E.

## Task

Find an elliptic curve E over Q of rank at least 30, and exhibit 30 rational
points on E that are linearly independent in E(Q) (equivalently, independent modulo
torsion). These 30 independent points force rank E(Q) >= 30.

## Submission

Write your answer to a file, then submit the path to that file using the `submit_answer`
tool. The file must contain a single JSON object with exactly two keys, "curve" and
"x_coords":

  - "curve": the five coefficients [a1, a2, a3, a4, a6] of the Weierstrass model above,
    each an integer written as a string (e.g. ["1", "0", "0", "-7", "12"]). The
    coefficients must define a non-singular curve (nonzero discriminant).

  - "x_coords": a list of exactly 30 x-coordinates, each the x-coordinate of a
    rational point on E. Give each as a string that is an exact integer (e.g. "5") or
    fraction (e.g. "-3/2"). You only supply x-coordinates: the verifier reconstructs a
    corresponding y (the two choices are the inverse points \(P\) and \(-P\); choosing either
    does not affect independence). Every x MUST be the x-coordinate of a genuinely rational
    point -- the reconstructed y must be rational -- otherwise the point is not in E(Q) and
    the submission is rejected.

The file should contain exactly this JSON object and nothing else. The example below only
illustrates the format; it is not a valid answer:

{"curve": ["0", "0", "0", "0", "1"], "x_coords": ["2", "0", "-1"]}

Approximate or floating-point coordinate values will not be accepted.
