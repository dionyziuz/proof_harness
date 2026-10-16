# Prime-Generating Polynomial Systems in Five Variables

- **Problem ID**: `prime-generating-polynomials`
- **Field**: Number theory
- **Notability**: Moderately interesting
- **Tags**: Construction - Finite, Improved Bound
- **Prompt Type**: `full_problem`
- **Status**: Open

---

## Problem Statement & Prompt

A polynomial with integer coefficients can "generate the primes" in the following sense. Work
with five variables x1, x2, x3, x4, x5 ranging over the nonnegative integers N = {0, 1, 2, ...}.

Find a system of polynomials f_1, ..., f_k in Z[x1, x2, x3, x4, x5] such that, as x ranges over
the common nonnegative integer zeros of the system (the points where f_1(x) = ... = f_k(x) = 0),
the value x1 + 2 ranges over exactly the prime numbers. Concretely:

* (Primes are attained) For every prime p there is a point x in N^5 with x1 = p - 2 and
  f_i(x) = 0 for all i. The offset of 2 lets the smallest prime, 2, correspond to x1 = 0.
* (Composites are excluded) There is no point x in N^5 with f_i(x) = 0 for all i for which
  x1 + 2 is composite.

Constraints on the system:
* 1 <= k <= 20 polynomials.
* Each polynomial has total degree at most 10.
* Every coefficient is a nonzero integer of absolute value at most 10000.
* For each prime p the verifier queries (every queried prime is below 2^32), the witness
  coordinates you produce must be nonnegative integers below 2^1000. This bound is a resource
  limit of the verifier, not part of the mathematical problem: witness(p) for larger primes is
  not subject to it, and composite exclusion is still required over all of N^5.

Polynomial encoding. A polynomial is a list of terms; a term is a list
[coefficient, e1, e2, e3, e4, e5] and denotes coefficient * x1**e1 * x2**e2 * x3**e3 * x4**e4 * x5**e5.
Each exponent vector may appear at most once per polynomial. For example x1 + 2 - x2 is
    [[1, 1, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0], [-1, 0, 1, 0, 0, 0]].

Solution format:
* Write a Python file that defines, at module level:
    - POLYNOMIALS: the list of polynomials, in the encoding above.
    - witness(p): given a prime p, returns the five nonnegative integers [x1, x2, x3, x4, x5]
      with x1 == p - 2 that make every polynomial vanish. It must work for arbitrary primes,
      not only small ones.
* You may optionally define PUBLIC_WITNESSES: a dict mapping the decimal string of each of the
  first 1000 primes to its five witness coordinates. This is an optional convenience and may be
  omitted.
* Your module is loaded in a fresh process, so do not rely on global state persisting between
  calls or on any file-level side effects.
