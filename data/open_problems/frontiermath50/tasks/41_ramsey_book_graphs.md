# Book Ramsey Numbers

- **Problem ID**: `ramsey-book-graphs`
- **Field**: Graph theory
- **Notability**: Moderately interesting
- **Tags**: Construction - Family
- **Prompt Type**: `full_problem`
- **Status**: Open

---

## Problem Statement & Prompt

Let B_n denote the triangular book graph on n+2 vertices, i.e. with n triangular "pages".

Find an algorithm that takes n as input and produces a graph on 4n - 2 vertices that does not contain a copy of B_{n-1} and whose complement does not contain a copy of B_n.

Provide the algorithm that takes n as input and outputs the graph as an adjacency string: a binary string that lists whether an edge is present (1) or absent (0) in column major order, zero-indexed. The order the edges should appear in the string is: {0,1}, {0,2}, {1,2}, {0,3}, {1,3}, {2,3}, {0,4}, {1,4}, {2,4}, {3,4}, ... For example, the string 011010 would represent the 4-vertex graph with edges {0,2}, {1,2}, {1,3}.

Solution format:
* Write a Python script defining a function `solution(n: int) -> str`.
* Do not include any code at the file level. You may include a `main` block for testing, but it will not be executed by the verifier.
* For n ≤ 100, each call to `solution(n)` must complete within 1 minute of wall-clock time when run on a typical laptop.
* An example is provided for you in the file "/solution_example.py".
