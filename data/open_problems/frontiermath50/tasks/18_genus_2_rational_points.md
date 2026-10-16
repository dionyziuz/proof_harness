# Many Rational Points on a Genus-2 Curve

- **Problem ID**: `genus-2-rational-points`
- **Field**: Diophantine
- **Notability**: Moderately interesting
- **Tags**: Construction - Finite, Record Breaking
- **Prompt Type**: `full_problem`
- **Status**: Solved (2026-09-03)

---

## Problem Statement & Prompt

For each squarefree polynomial f(x) in Z[x] of degree 6, the curve

    C_f : y^2 = f(x)

is hyperelliptic of genus 2. By Faltings' theorem (the Mordell conjecture), a curve of
genus at least 2 over Q has only finitely many rational points. Equivalently, there are
only finitely many x in Q for which f(x) is the square of a rational number.

The rational points of C_f come in two kinds:

  - Affine points: each x in Q with f(x) = y^2 != 0 yields a pair (x, y) and (x, -y); an
    x with f(x) = 0 yields the single point (x, 0).
  - Points at infinity: the smooth projective model has two points at infinity. They are
    rational if and only if the leading coefficient c6 of f is the square of a rational
    (equivalently, a perfect-square integer).

The current record is 642 rational points, achieved in 2008 by Elkies and Stoll with

    y^2 = 82342800 x^6 - 470135160 x^5 + 52485681 x^4 + 2396040466 x^3
          + 567207969 x^2 - 985905640 x + 247747600.

(Its leading coefficient 82342800 is not a perfect square, so this curve has no rational
points at infinity and all 642 of its rational points are affine.)

## Task

Find a squarefree polynomial f(x) in Z[x] of degree exactly 6 such that the curve
y^2 = f(x) has MORE THAN 642 rational points in total (affine points together with any
rational points at infinity).

(There is no loss of generality in taking integer coefficients: f and c^2 * f define
isomorphic curves for any c in Q*, so denominators and square factors of the leading
coefficient can be cleared.)

## Submission

Write your answer to a file, then submit the path to that file using the `submit_answer`
tool. The file must contain a single JSON object with exactly two keys, "coeffs" and
"points":

  - "coeffs": the 7 coefficients of f as a list of integer strings, CONSTANT TERM FIRST,
    i.e. ["c0", "c1", "c2", "c3", "c4", "c5", "c6"] meaning f(x) = c0 + c1 x + ... + c6 x^6.
    Each coefficient must be an integer written as a string (e.g. "0", "-7", "12"). The
    leading coefficient c6 must be nonzero, so that f has degree exactly 6.

  - "points": a list of [x, y] pairs giving the AFFINE rational points, each coordinate a
    string parseable as an exact rational (an integer like "5" or a fraction like "-3/2").
    Every pair must satisfy y^2 = f(x). Do NOT list the points at infinity -- the verifier
    adds them automatically (two of them when c6 is a perfect square, otherwise none).

The file should contain exactly this JSON object and nothing else. The example below only
illustrates the format; its points do not give a record curve:

{"coeffs": ["1", "0", "1", "0", "1", "0", "1"], "points": [["0", "1"], ["0", "-1"], ["1", "2"], ["1", "-2"]]}

Everything is checked exactly with arbitrary-precision rational arithmetic, so approximate
or floating-point values will not be accepted.
