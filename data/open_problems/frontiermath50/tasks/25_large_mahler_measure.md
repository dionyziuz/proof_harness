# Littlewood Polynomials with Large Normalized Mahler Measure

- **Problem ID**: `large-mahler-measure`
- **Field**: Analysis
- **Notability**: Moderately interesting
- **Tags**: Construction - Finite, Improved Bound
- **Prompt Type**: `full_problem`
- **Status**: Open

---

## Problem Statement & Prompt

For a polynomial P(x) = sum_{k=0}^d a_k x^k, write e(t) = exp(2*pi*i*t) and recall the Mahler measure

    M_0(P) = exp( integral_0^1 log|P(e(t))| dt ).

A Littlewood polynomial is one whose coefficients all satisfy a_k = +-1. For such a polynomial of degree d, Parseval gives ||P||_2 = sqrt(d+1), and Jensen's inequality yields M_0(P) <= sqrt(d+1). Define the normalized Mahler measure

    b(P) = M_0(P) / sqrt(d+1),

so that 0 < b(P) <= 1. The goal is to construct Littlewood polynomials whose normalized Mahler measure is as large as possible at arbitrarily large degree.

Concretely: find an algorithm that takes a positive integer n as input and produces a Littlewood polynomial of degree strictly greater than n whose normalized Mahler measure satisfies

    b(P) > 0.97.

Solution format:
* Write a Python script defining a function `solution(n: int) -> str`.
* The returned string must be over the alphabet {'+', '-'} and have length d+1 > n+1, where the k-th character (0-indexed) is the sign of the coefficient a_k, listing a_0, a_1, ..., a_d (lowest-order coefficient first). '+' denotes the coefficient +1 and '-' denotes -1.
* Do not include any code at the file level. You may include a `main` block for testing, but it will not be executed by the verifier.
* You may assume n <= 500000; no larger value of n is tested.
* For each tested n the call to solution(n) must complete within 5 minutes on a typical laptop, and the returned degree must not exceed 2000000. Since n is at most 500000, this cap always leaves room for a valid answer.
* An example showing the required format is provided in the file "/solution_example.py".
