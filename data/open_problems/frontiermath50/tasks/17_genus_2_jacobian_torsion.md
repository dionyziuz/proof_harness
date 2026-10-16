# A Genus 2 Curve over the Rationals with a Rational Torsion Point of Prime Order at Least 31

- **Problem ID**: `genus-2-jacobian-torsion`
- **Field**: Arithmetic geometry
- **Notability**: Moderately interesting
- **Tags**: Construction - Finite, Record Breaking
- **Prompt Type**: `full_problem`
- **Status**: Solved (2026-07-30)

---

## Problem Statement & Prompt

For each squarefree polynomial f(x) in Z[x] of degree 6, the curve

    C_f : y^2 = f(x)

is hyperelliptic of genus 2, and its Jacobian J_f = Jac(C_f) is an abelian surface over Q.

A rational point of J_f can be given in Mumford coordinates by a pair of polynomials (u, v)
over Q with

  - u monic of degree 2,
  - deg v <= 1,
  - u dividing v^2 - f.

Such a pair describes the effective divisor D of degree 2 cut out by u(x) = 0 and y = v(x),
and the corresponding point of J_f is the divisor class [D - K], where K is the canonical
class of C_f: the degree-2 divisor cut out by x = constant, equivalently the sum of the two
points at infinity of the smooth projective model. (K is a rational divisor class even when
the two points at infinity are individually irrational, so [D - K] is a rational point of J_f
for every such pair (u, v).)

# Task

Find a squarefree polynomial f(x) in Z[x] of degree exactly 6, together with a point of
J_f(Q) whose order is a prime number p >= 31.

(There is no loss of generality in taking f to have integer coefficients: f and c^2 * f define
isomorphic curves for any c in Q*, so denominators can be cleared.)

# Submission

Write your answer to a file, then submit the path to that file using the `submit_answer` tool.
The file must contain a single JSON object with exactly three keys, "coeffs", "u", and "v",
each a list of coefficients given as strings, CONSTANT TERM FIRST:

  - "coeffs": the 7 coefficients of f, i.e. ["c0", "c1", ..., "c6"] meaning
    f(x) = c0 + c1 x + ... + c6 x^6. Each must be an integer, and c6 must be nonzero so that f
    has degree exactly 6.
  - "u": the 3 coefficients of u, i.e. ["u0", "u1", "u2"] meaning u(x) = u0 + u1 x + u2 x^2.
  - "v": the 2 coefficients of v, i.e. ["v0", "v1"] meaning v(x) = v0 + v1 x.

Every entry must be a string holding an exact integer such as "5" or an exact fraction such as
"-3/2". Everything is checked with exact arithmetic, so approximate or floating-point values will
not be accepted.

The file should contain exactly this JSON object and nothing else. The example below only
illustrates the format; it is not a valid answer:

{"coeffs": ["1", "0", "1", "0", "1", "0", "1"], "u": ["1", "0", "1"], "v": ["0", "0"]}
