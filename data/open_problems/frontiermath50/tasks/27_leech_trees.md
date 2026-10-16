# Leech Trees

- **Problem ID**: `leech-trees`
- **Field**: Combinatorics
- **Notability**: Moderately interesting
- **Tags**: Construction - Finite, Record Breaking
- **Prompt Type**: `full_problem`
- **Status**: Open

---

## Problem Statement & Prompt

A tree with positive-integer edge weights is called a Leech tree if the set of all pairwise path distances between vertices is exactly {1, 2, 3, ..., n(n-1)/2}, where n is the number of vertices. In other words, every integer from 1 up to n(n-1)/2 appears as the path distance between exactly one pair of vertices.

Your task is to construct a Leech tree with 18 or more vertices.

Output your answer as a file named submission.json containing a JSON array of edges, where each edge is a triple [x, y, w]:
- x and y are positive integers representing the two endpoint vertices
- w is a positive integer representing the edge weight

The edges must form a valid tree (connected, no cycles) on n >= 18 vertices with positive-integer edge weights, and the n(n-1)/2 pairwise path distances must be exactly the integers 1, 2, 3, ..., n(n-1)/2.

Example format for a small Leech tree on 3 vertices (distances 1, 2, 3):
[
  [1, 2, 1],
  [2, 3, 2]
]

Your submission.json should follow this same format, scaled up to 18 or more vertices.
