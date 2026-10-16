# Short Superpermutations over 8, 9, and 10

- **Problem ID**: `superpermutations`
- **Field**: Combinatorics
- **Notability**: Moderately interesting
- **Tags**: Construction - Family, Improved Bound
- **Prompt Type**: `full_problem`
- **Status**: Solved (2026-07-27)

---

## Problem Statement & Prompt

A superpermutation on n symbols is a sequence of integers, each in {1, ..., n}, that contains every one of the n! permutations of (1, 2, ..., n) as a contiguous (consecutive) subsequence. For example, the sequence

    1 2 1

is a superpermutation on 2 symbols, because its two length-2 windows, (1, 2) and (2, 1), are exactly the two permutations of (1, 2). Likewise

    1 2 3 1 2 1 3 2 1

is a superpermutation on 3 symbols of length 9: sliding a window of length 3 across it yields all six permutations of (1, 2, 3).

A central question is how short a superpermutation on n symbols can be. The minimum length is known exactly only for small n (it equals 1! + 2! + ... + n! for n <= 5: 1, 3, 9, 33, 153). For n >= 6 the minimum is unknown, and the best results are upper bounds from explicit constructions.

## Task

Find three short superpermutations:

  * seq8  : a superpermutation on 8 symbols of length at most 46204
  * seq9  : a superpermutation on 9 symbols of length at most 408965
  * seq10 : a superpermutation on 10 symbols of length at most 4037046

Each sequence must use only the symbols 1, ..., n (with n = 8, 9, 10 respectively) and must contain all n! permutations as consecutive windows. All three must satisfy their length bounds for the submission to be fully accepted; a submission in which only some of the sequences are valid earns partial credit.

## Submission

Write your answer to a single text file containing one labelled list per sequence:

    seq8: [1, 2, 3, ...]
    seq9: [1, 2, 3, ...]
    seq10: [1, 2, 3, ...]

Each list is an ordinary bracketed, comma-separated list of integers and may span multiple lines. Submit the absolute path of this file using the submit tool. Verification is exact: a sequence that misses even one permutation, uses a symbol outside its range, or exceeds its length bound will be rejected.
