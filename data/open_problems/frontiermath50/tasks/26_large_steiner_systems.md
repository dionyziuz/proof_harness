# Large Steiner Systems

- **Problem ID**: `large-steiner-systems`
- **Field**: Combinatorics
- **Notability**: Solid result
- **Tags**: Construction - Finite, Record Breaking
- **Prompt Type**: `full_problem`
- **Status**: Open

---

## Problem Statement & Prompt

Let [n] be the elements {1,...,n}. Call T a q-subset of [n] if T⊆[n] and the size of T is q.

A (n,q,r)-Steiner system S is a set of q-subsets of [n] such that every r-subset of [n] is contained in exactly 1 element of S.

Your task is to find a (n,q,r)-Steiner system for any n > q > r > 5, where r < 10 and n < 200.

Submit your answer with the parameters n, q, r, and blocks (the list of q-subsets, where each block is a list of q integers from {1, ..., n}).
