# The No-Three-in-Line Problem

- **Problem ID**: `no-three-in-line`
- **Field**: Discrete geometry
- **Notability**: Moderately interesting
- **Tags**: Construction - Family, Improved Bound
- **Prompt Type**: `full_problem`
- **Status**: Open

---

## Problem Statement & Prompt

Let G_n be the n x n grid of lattice points {1, 2, ..., n} x {1, 2, ..., n}.

A no-three-in-a-line set is a subset S of G_n such that no three points of S lie on a common straight line. Let t(n) be the maximum possible size of such a set.

The trivial upper bound is t(n) <= 2n (more than 2n points force three in some row).

Sets of size 1.5 * n - O(1) are classical, so it is the ratio 3/2 that is the barrier: it is not known whether t(n) / n can exceed 3/2 for arbitrarily large n. Your task is to beat that ratio, which means beating it by a constant factor -- shaving an additive constant off a 1.5 * n - O(1) construction leaves the ratio at 3/2 and does not count.

Concretely: give an algorithm that produces, for arbitrarily large n, a no-three-in-a-line set of MORE than 1.8 * n points.

Solution format:
* Write a Python script defining a function `solution(min_n: int) -> tuple[int, list[list[int]]]`.
* Given a lower bound `min_n`, `solution(min_n)` must return a pair `(n, points)` where:
  - `n` is an integer with `min_n <= n <= 2 * min_n` (you may choose any such n; your family need not include every n),
  - `points` is a list of `[x, y]` pairs of integers with `1 <= x <= n` and `1 <= y <= n`,
  - the points are distinct,
  - no three of the points are collinear, and
  - `len(points) > 1.8 * n` (strictly).
* The verifier calls `solution(min_n)` for many increasing lower bounds `min_n`. The n you return must be STRICTLY INCREASING across these calls.
* Each call to `solution(min_n)` must complete within 1 minute when run on a typical laptop.
* Do not include any code at the file level. You may include a `main` block for testing, but it will not be executed by the verifier.
* An example showing the required format is provided in the file "solution_example.py". (That example returns only a handful of points, far below the threshold; it is there to illustrate the interface, not to solve the problem.)
