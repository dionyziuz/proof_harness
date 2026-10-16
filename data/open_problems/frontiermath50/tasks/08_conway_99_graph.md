# Conway's 99-Graph Problem

- **Problem ID**: `conway-99-graph`
- **Field**: Graph theory
- **Notability**: Solid result
- **Tags**: Construction - Finite
- **Prompt Type**: `full_problem`
- **Status**: Open

---

## Problem Statement & Prompt

Your task is to find a specific graph.

A graph on n vertices is called srg(n, k, 1, 2) if:
- every vertex has exactly k neighbors,
- every pair of adjacent vertices has exactly 1 common neighbor, and
- every pair of non-adjacent vertices has exactly 2 common neighbors.

For these parameters, k is forced by n. Counting length-2 paths from a fixed vertex gives k(k - 2) = 2(n - k - 1), hence k = sqrt(2(n-1)), so k is not a free parameter.

Find an srg(99, 14, 1, 2) — a graph on 99 vertices satisfying the conditions above with k = 14.  Whether such a graph exists is an open problem, known as Conway's 99-graph problem.

Write your 99-graph construction to a file named `submission.txt`, as a list of edges, one per line, in the format:

  {u, v}

where u and v are the vertex labels (integers from 1 to 99, inclusive). Treat the graph as undirected. List each unordered edge exactly once, with no self-loops. A valid srg(99, 14, 1, 2) would have 693 edges.
