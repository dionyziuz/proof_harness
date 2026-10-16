# A counterexample to Wilf's Conjecture

- **Problem ID**: `numerical-semigroups`
- **Field**: Number theory
- **Notability**: Solid result
- **Tags**: Construction - Finite, Counterexample
- **Prompt Type**: `full_problem`
- **Status**: Open

---

## Problem Statement & Prompt

Your task is to find a counterexample to Wilf's conjecture.

A numerical semigroup S = ⟨n_1, ..., n_k⟩ is the set of all non-negative integer combinations

    a_1 n_1 + ... + a_k n_k        (each a_i a non-negative integer)

of a fixed set of positive integers n_1, ..., n_k (the "generators") whose overall greatest common divisor is 1. Because the gcd is 1, S contains every sufficiently large integer, so only finitely many non-negative integers are missing from S; the missing ones are called the gaps of S.

Associated to S are the following invariants:

- genus g(S): the number of gaps.
- Frobenius number F(S): the largest gap.
- conductor c(S) = F(S) + 1: every integer >= c(S) belongs to S.
- multiplicity m(S): the smallest nonzero element of S (equivalently, the smallest generator).
- embedding dimension e(S): the size of the unique minimal generating set of S.
- n(S): the number of elements of S that are strictly less than c(S). Equivalently, n(S) = c(S) - g(S).

For example, S = ⟨5, 7, 9⟩ has elements 0, 5, 7, 9, 10, 12, 14, 15, 16, ... and gaps 1, 2, 3, 4, 6, 8, 11, 13, so g = 8, F = 13, c = 14, m = 5, e = 3, and n = 6.

Wilf's conjecture (1978) asserts that every numerical semigroup satisfies

    e(S) · n(S) >= c(S).

A counterexample is a single numerical semigroup S with

    e(S) · n(S) < c(S).

Find one.

## Certificate format

Write your answer to a file containing a single JSON object and submit the file path.

The minimal certificate is just the list of generators:

    {"generators": [5, 7, 9]}

The verifier only builds S directly when the multiplicity m (the smallest generator) is at most 200000. For a larger m, building S is too slow, so you must additionally supply the Apéry set of S together with a predecessor for each of its elements:

    {
      "generators": [5, 7, 9],
      "apery": [0, 16, 7, 18, 9],
      "apery_predecessors": [null, [2, 9], [0, 7], [4, 9], [0, 9]]
    }

The Apéry set is the list w_0, w_1, ..., w_{m-1}, where w_r is the smallest element of S congruent to r modulo m; in particular w_0 = 0. It determines S completely, and every invariant above follows from it.

Each entry of `apery_predecessors` is either null (for entry 0) or a pair [prev, gen] certifying w_r = w_{prev} + gen for a submitted generator gen. Following these links back to w_0 = 0 writes each w_r as a sum of generators, proving w_r is in S.

The generators must be positive integers with overall gcd 1. The semigroup must be nontrivial — the smallest generator must be at least 2, equivalently S has at least one gap (S = ⟨1⟩ is the whole set of non-negative integers, for which the invariants degenerate, so it is not accepted). All arithmetic is exact.
