# The Earth–Moon Problem

- **Problem ID**: `earth-moon`
- **Field**: Graph theory
- **Notability**: Moderately interesting
- **Tags**: Construction - Finite, Improved Bound
- **Prompt Type**: `full_problem`
- **Status**: Open

---

## Problem Statement & Prompt

A graph G = (V, E) is *biplanar* if its edge set E can be partitioned into two parts E_1 and
E_2 such that both (V, E_1) and (V, E_2) are planar. (Equivalently, G has thickness at most 2.)

Your task: construct a biplanar graph whose chromatic number is at least 10. Such a graph
would improve the best known lower bound for the Earth-Moon problem from 9 to 10. (Achieving
12, the known upper bound, would resolve the problem completely.)

You must exhibit the graph as a partition of its edges into two planar parts E_1 and E_2,
supply a proper coloring of the graph, and state the chromatic number k you claim for it.
Your submission is accepted if and only if:

  1. Both (V, E_1) and (V, E_2) are planar (so G is biplanar);
  2. k is between 10 and 12 (every biplanar graph has chromatic number at most 12);
  3. the coloring you supply is a proper coloring of G = (V, E_1 ∪ E_2) using at most k colors
     (so the chromatic number of G is at most k); and
  4. G cannot be properly colored with k - 1 colors (so its chromatic number is at least k).

Conditions 3 and 4 together prove that the chromatic number of G is exactly k.

Solution format:
* Write your answer to a JSON file and submit its path.
* The JSON must be an object with these keys:
  - "num_vertices": a positive integer N. The vertices are the integers 0, 1, ..., N-1.
  - "edges_part1": a list of edges [u, v] forming the first planar part E_1.
  - "edges_part2": a list of edges [u, v] forming the second planar part E_2.
  - "coloring": a list of N integers; entry i is the color assigned to vertex i.
  - "chromatic_number": the integer k you claim is the chromatic number of G.
* E_1 and E_2 must be disjoint (together they are the edge set of G). Endpoints are integers
  in [0, N), and edges are undirected.
* Colors may be any integers; only which colors are equal matters. Color G with exactly k
  colors and set "chromatic_number" to that same k. Your choice of k must be at least 10 and
  at most 12. A larger value of k is a stronger result.

Example submission (illustrating the format only — this graph is not biplanar with a high
chromatic number):
{
  "num_vertices": 3,
  "edges_part1": [[0, 1], [1, 2]],
  "edges_part2": [[0, 2]],
  "coloring": [0, 1, 2],
  "chromatic_number": 3
}
