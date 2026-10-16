# Large $\ell$-Rank in Class Groups of Imaginary Quadratic Fields

- **Problem ID**: `class-group-rank`
- **Field**: Algebraic number theory
- **Notability**: Moderately interesting
- **Tags**: Record Breaking
- **Prompt Type**: `full_problem`
- **Status**: Open

---

## Problem Statement & Prompt

Find imaginary quadratic fields K = Q(sqrt(-D)) whose class group Cl(K) has large ℓ-rank.

For a prime ℓ, the ℓ-rank of Cl(K) is the integer r such that the ℓ-torsion subgroup Cl(K)[ℓ] is
isomorphic to (Z/ℓZ)^r. Exhibit fields achieving ℓ-rank at least r for the (ℓ, r) pairs in the
challenge set below. You demonstrate that the ℓ-rank is at least r by giving r independent
class-group elements, each of order exactly ℓ, so that together they generate a subgroup
isomorphic to (Z/ℓZ)^r.

## Challenge set

Each (ℓ, r) pair below is a separate challenge:

    ℓ = 3    r = 9, 10, 11, 12, 13, 14, 15, 16
    ℓ = 5    r = 6, 7, 8, 9, 10, 11, 12
    ℓ = 7    r = 5, 6, 7, 8, 9, 10
    ℓ = 11   r = 4, 5, 6, 7, 8
    ℓ = 13   r = 4, 5, 6, 7, 8

Each challenge you certify earns credit; certifying every pair in the set earns full credit, and
any nonempty subset earns partial credit. Completing more is better.

## Submission

Write your answer to a file, one certificate per line, with fields separated by the pipe
character, and submit the file's path with the submit tool:

    ℓ|r|D|gen_1|gen_2|...|gen_r

- ℓ — the prime. Every generator on the line must have order exactly ℓ in Cl(K).
- r — the number of generators that follow. They must be independent.
- D — a positive integer such that -D is a fundamental discriminant; the field is K = Q(sqrt(-D)).
- gen_i — an element of Cl(K), given as a binary quadratic form of discriminant -D: three
  comma-separated integers "a,b,c" meaning a·x² + b·x·y + c·y², with b² - 4·a·c = -D and a > 0
  (the form must be positive definite).

Only lines whose (ℓ, r) is in the challenge set earn credit; you may include others (for example
the one below) but they are ignored.

### Example line

    3|3|4447704|390,-96,2857|921,-786,1375|346,-68,3217

This is ℓ = 3, r = 3, discriminant -4447704, and three independent order-3 forms — a well-formed
example of the format. (3, 3) is below the challenge set, so it verifies without counting.
