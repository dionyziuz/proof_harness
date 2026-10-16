# Existence of EFX Allocations

- **Problem ID**: `efx`
- **Field**: Social choice
- **Notability**: Moderately interesting
- **Tags**: Construction - Finite, Counterexample
- **Prompt Type**: `full_problem`
- **Status**: Open

---

## Problem Statement & Prompt

# An example with no EFX allocation

## Setup

We divide m indivisible goods among n agents. Agent i has an **additive**
valuation: a nonnegative integer value V[i][g] for each good g, and values a
bundle S of goods by the sum of its parts,

    v_i(S) = sum over g in S of V[i][g].

An **allocation** (A_1, ..., A_n) partitions the goods: every good is given to
exactly one agent (bundles may be empty). There are exactly n^m allocations.

## EFX (envy-free up to any good)

An allocation is **EFX** if no agent strongly envies another after the removal of
any single positively-valued good from the envied bundle. Formally, for every
ordered pair of distinct agents i, j and every good g in A_j with V[i][g] > 0:

    v_i(A_i)  >=  v_i(A_j) - V[i][g].

The phrase "up to *any* good" is the crux: the inequality must hold even for the
good whose removal helps i the least (the smallest good in A_j that i values
positively). Equivalently, for every ordered pair (i, j) such that A_j contains
at least one good i values positively:

    v_i(A_i)  >=  v_i(A_j) - min{ V[i][g] : g in A_j and V[i][g] > 0 }.

## Your task

Exhibit an example where an EFX allocation does not exist: choose n and m and
an n-by-m valuation matrix such that **no** allocation of the m goods to the n
agents is EFX. Since EFX allocations always exist for n <= 3, choose n >= 4.

## Submission format

Write your answer to a JSON file: an array of n rows, where row i is a list of m
nonnegative integers giving agent i's values for goods 0, 1, ..., m-1. All rows
must have the same length. Submit that file's path with the submit tool.

Example:

    [[3, 0, 1],
     [1, 2, 2]]

Here agent 0 values the three goods at 3, 0, 1 and agent 1 at 1, 2, 2. (This
example is only to show the format; it is not a counterexample -- an EFX
allocation exists for it.)

Keep n^m small enough that all allocations can be checked within the time
limit.
