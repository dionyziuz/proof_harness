# Hall's Conjecture

- **Problem ID**: `hall-conjecture`
- **Field**: Diophantine
- **Notability**: Moderately interesting
- **Tags**: Counterexample, Record Breaking
- **Prompt Type**: `full_problem`
- **Status**: Open

---

## Problem Statement & Prompt

Hall's conjecture concerns how close a perfect square can be to a perfect cube without being equal to it. For positive integers x and y with

    x^3 != y^2,

define the "Hall ratio"

    r(x, y) = sqrt(x) / |y^2 - x^3|.

A large ratio means y^2 is a remarkably good (but not exact) approximation to x^3 relative to the size of x. The original strong form of Hall's conjecture asserted that this ratio is bounded by an absolute constant. Decades of computation have produced only finitely many large examples: the largest ratio currently known is approximately 46.6. A pair achieving a ratio greater than 100 would be a new record.

## Task

Find positive integers x and y such that

    x^3 != y^2    and    sqrt(x) / |y^2 - x^3| > 100.

## Submission

Submit your two integers as (x, y) using the submit tool. There is no bound on the magnitude of x or y. Your submission is checked exactly with arbitrary-precision integer arithmetic (the ratio condition is tested as x > (100 * |y^2 - x^3|)^2), so an approximate or rounded answer will not be accepted.
