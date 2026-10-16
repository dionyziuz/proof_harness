# The Missing Moore Graph

- **Problem ID**: `missing-moore-graph`
- **Field**: Graph theory
- **Notability**: Solid result
- **Tags**: Construction - Finite
- **Prompt Type**: `full_problem`
- **Status**: Open

---

## Problem Statement & Prompt

Your task is to find the missing Moore graph.

A Moore graph is a graph that achieves the Moore bound: the maximum possible number of vertices for a given degree k and diameter d. For degree k and diameter d=2, the Moore bound is 1 + k + k(k-1) = k^2 + 1 vertices.

Moore graphs of diameter 2 are known to exist for degrees k = 2 (the 5-cycle), k = 3 (the Petersen graph), and k = 7 (the Hoffman-Singleton graph). It was proved by Hoffman and Singleton in 1960 that the only remaining possible degree is k = 57, which would give a graph on 57^2 + 1 = 3250 vertices. Whether this graph exists is one of the oldest open problems in graph theory. It has never been found, and it has never been proved not to exist — hence the name "the missing Moore graph."

A graph is the missing Moore graph if and only if it satisfies all of the following:

1. It has exactly 3250 vertices, labelled 1 through 3250.
2. Every vertex has degree exactly 57 (it is 57-regular).
3. Every pair of non-adjacent vertices has exactly 1 common neighbor (diameter 2).
4. No two adjacent vertices share a common neighbor.

Conditions 3 and 4 together make it a strongly regular graph with parameters srg(3250, 57, 0, 1).

Your task is to construct such a graph and write it to a file. The file should contain exactly one edge per line, in the following format:

  {x, y}

where x and y are integers between 1 and 3250 (inclusive). List each edge once only (do not list both {x, y} and {y, x}). The file should contain exactly 92625 edges (since 3250 * 57 / 2 = 92625).

Example of correctly formatted lines:
  {1, 2}
  {1, 3}
  {2, 57}
