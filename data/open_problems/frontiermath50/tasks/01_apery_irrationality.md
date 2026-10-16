# Apéry-Style Irrationality Proofs via Linear Recurrences

- **Problem ID**: `apery-irrationality`
- **Field**: Number theory
- **Notability**: Breakthrough
- **Tags**: Construction - Finite
- **Prompt Type**: `full_problem`
- **Status**: Solved (2026-09-17)

---

## Problem Statement & Prompt

The goal of this problem is to prove the irrationality of a particular real number in a specific way.

Let k be an integer with 1 ≤ k ≤ 25, and let P_0(x), ..., P_k(x) be polynomials with integer coefficients, each of degree at most 25, where P_k(n) ≠ 0 for n ≥ k. Let α = (α_0, ..., α_{k-1}) and β = (β_0, ..., β_{k-1}) be tuples of integers.

Let (a_n) and (b_n) denote the sequences defined by the recurrence P_k(n)*u_n = P_{k-1}(n)*u_{n-1} + ... + P_0(n)*u_{n-k} for n ≥ k, when initialized with α and β respectively.

Let L_n denote lcm(1,2,...,n) and let B_n = (L_n)^m*b_n, where m is a fixed positive integer.

Let X be a real number. It is known that if the following hold, then X is irrational.

1. a_n is an integer for all n ≥ 0.

2. B_n is an integer for all n ≥ 0.

3. X ≠ b_n / a_n for all n ≥ 0.

4. |b_n - a_n*X| ≤ C*ρ^n*e^{-mn} for all n ≥ 0, where C and ρ are positive real numbers and ρ < 1.

Your task is to find P_i, α, β, and m that establish the irrationality of ___ in this way.

Your answer is verified numerically, as follows. Conditions 1-4 are checked for all 0 ≤ n < 650, with condition 4 checked using the fixed values C = 100 and ρ = 0.95. An answer that satisfies condition 4 only with a larger C or ρ is rejected; in particular, note that scaling α and β by a common factor scales C by that factor. In addition, every integer in the submission — each polynomial coefficient and each entry of α and β — may have at most 100 decimal digits. Conditions 1 and 2 are checked exactly, but condition 4 is checked in floating point against a value of X carrying 6000 decimal digits. This bounds how sharply the linear form may converge: if |b_n - a_n*X| falls below (|a_n| + |b_n|)*10^-5990 at some tested n, the computed value cannot be distinguished from zero and the answer is rejected. For a typical certificate, where |b_n - a_n*X| is of order 1/a_n, this accommodates m up to about 10. m > 21 is rejected immediately, since no certificate with this m could be checked given this precision.
