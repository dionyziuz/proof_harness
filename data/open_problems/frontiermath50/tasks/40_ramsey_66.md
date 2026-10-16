# The Ramsey Number $R(6,6)$

- **Problem ID**: `ramsey-66`
- **Field**: Graph theory
- **Notability**: Moderately interesting
- **Tags**: Construction - Finite, Improved Bound
- **Prompt Type**: `full_problem`
- **Status**: Open

---

## Problem Statement & Prompt

The Ramsey number R(s, t) is the smallest integer n such that any 2-coloring of the edges of the complete graph K_n contains either a red clique of size s or a blue clique of size t. The current best known bounds for R(6, 6) are:

  102 ≤ R(6, 6) ≤ 160

To show R(6, 6) > 102, it suffices to exhibit a 2-coloring of the edges of K_n (for some n ≥ 102) such that there is no monochromatic K_6 — that is, no set of 6 vertices whose induced edges are all the same color.

## Task

Find such a 2-coloring. Specifically, provide:
- An integer n ≥ 102.
- A set of "red" edges of K_n (vertices labeled 1 through n). All remaining edges of K_n are "blue".

The coloring must contain no monochromatic K_6 in either color.

## Submission format

Write your answer to a file named `submission.json`, as a JSON object with the following fields:

```json
{
  "n": <int>,
  "red_edges": [[u, v], ...]
}
```

- `n`: the number of vertices; must be at least 102.
- `red_edges`: a list of edges `[u, v]` with 1 ≤ u < v ≤ n. Each edge appears at most once. All edges of K_n not listed here are blue.
