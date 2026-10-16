# Condorcet Winning Sets

- **Problem ID**: `condorcet-winning-sets`
- **Field**: Social choice
- **Notability**: Solid result
- **Tags**: Construction - Finite, Counterexample
- **Prompt Type**: `full_problem`
- **Status**: Open

---

## Problem Statement & Prompt

# Condorcet Winning Sets of Size at Least Four

## Setup

An election consists of n voters and m candidates (labelled 1, 2, ..., m). Each
voter submits a strict ranking of all m candidates (a permutation), from most
preferred to least preferred.

Given a committee (a subset S of candidates) and an alternative candidate c not
in S, say that c "overthrows" S if at least half of the voters rank c above
every member of S. Formally, c overthrows S when

    #{ voters v : v ranks c higher than every member of S }  >=  n / 2.

A committee S is a *Condorcet winning set* (CWS) if no alternative candidate
overthrows it -- that is, for every c not in S, strictly fewer than half of the
voters prefer c to the whole committee.

(Equivalently: S is a CWS iff for every candidate c not in S, a strict majority
of voters rank at least one member of S above c. This is the definition of
Elkind, Lang & Saffidine, and the "1/2-undominated" notion of Charikar et al.,
"Six Candidates Suffice to Win a Voter Majority".)

Note that any superset of a Condorcet winning set is again a Condorcet winning
set, so the relevant quantity is the size of the *smallest* CWS. The full
candidate set is always a CWS, so a smallest CWS always exists.

## Your task

Construct an election in which **every Condorcet winning set has size at least
four** -- i.e. there is no Condorcet winning set of size 3 (and hence, by upward
closure, none of size 1 or 2 either).

You choose n (the number of voters) and m >= 4 (the number of candidates).

## Submission format

Write your answer to a file as the n-by-m matrix of rankings, in plain text: n
lines, each with m whitespace-separated integers. Each of the n rows is one
voter's ranking -- the m candidate labels (1 .. m) in order from most preferred
to least preferred -- and must be a permutation of 1 .. m. Submit that file's
path using the submit tool.

Example (n = 3 voters, m = 4 candidates):

    3 1 4 2
    1 2 3 4
    4 2 1 3

Here the first voter prefers candidate 3 > 1 > 4 > 2.

The verifier will check every 3-candidate committee and accept your submission
only if each one is overthrown by some alternative (so no size-3 Condorcet
winning set exists).
