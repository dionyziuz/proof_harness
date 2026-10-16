# Euler's Sum of Powers Conjecture

- **Problem ID**: `euler-sum-of-powers`
- **Field**: Diophantine
- **Notability**: Moderately interesting
- **Tags**: Counterexample, Integer Solution
- **Prompt Type**: `full_problem`
- **Status**: Open

---

## Problem Statement & Prompt

In 1769 Leonhard Euler conjectured that for every integer n > 2, no sum of fewer than n positive n-th powers can itself be an n-th power. In its tightest form this says the equation

    a_1^n + a_2^n + ... + a_{n-1}^n = b^n

has no solution in positive integers: with only n-1 terms on the left, an n-th power cannot be reached. Euler's conjecture stood for nearly two centuries. It was finally refuted for n = 5 by Lander and Parkin in 1966, who found

    27^5 + 84^5 + 110^5 + 133^5 = 144^5,

and later for n = 4 by Elkies and Frye. But for every exponent n > 5, no counterexample is known, and whether any exists remains open.

## Task

Find an integer k > 5 and positive integers a_1, ..., a_{k-1}, a_k such that

    a_1^k + a_2^k + ... + a_{k-1}^k = a_k^k.

The length of your list determines the exponent: if you submit k integers, the exponent is k, the first k-1 are summed, and the last is the right-hand side. So a valid submission of k integers is exactly a counterexample to Euler's conjecture for exponent k.

## Submission

Submit your list of integers (a_1, ..., a_k) using the submit tool. Every integer must be positive, as in the statement of the conjecture; zero or negative entries are rejected. There is no bound on their magnitude. Your submission is checked exactly with arbitrary-precision integer arithmetic, so an approximate or rounded answer will not be accepted.
