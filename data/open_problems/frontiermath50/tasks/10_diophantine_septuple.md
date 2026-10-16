# A Rational Diophantine Septuple

- **Problem ID**: `diophantine-septuple`
- **Field**: Diophantine
- **Notability**: Moderately interesting
- **Tags**: Integer Solution
- **Prompt Type**: `full_problem`
- **Status**: Open

---

## Problem Statement & Prompt

A *rational Diophantine m-tuple* is a set of m distinct non-zero rational numbers

    {a_1, ..., a_m}

with the property that a_i * a_j + 1 is the square of a rational number for every pair of distinct indices i != j. Only the off-diagonal products are constrained: a_i^2 + 1 need not be a square.

## Task

Find a rational Diophantine 7-tuple: that is, 7 distinct non-zero rational numbers a_1, ..., a_7 such that a_i * a_j + 1 is the square of a rational number for every pair of distinct indices i != j.

## Submission

Write your 7 rationals to a file as a list of strings. Each entry should be an integer "p" or a fraction "p/q" (no decimal points or exponents). Submit the path to that file using the submit tool. The file should contain exactly that list and nothing else, for example:

    ["1", "3", "8", "120", "-5/2", "7/3", "11"]

No bound is placed on the size of the numerators and denominators; because the values travel in a file rather than on a command line, an entry may have as many digits as it needs.
