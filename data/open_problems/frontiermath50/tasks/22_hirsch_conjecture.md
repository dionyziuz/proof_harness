# The Hirsch Conjecture in Dimension 4

- **Problem ID**: `hirsch-conjecture`
- **Field**: Discrete geometry
- **Notability**: Solid result
- **Tags**: Construction - Finite, Improved Bound
- **Prompt Type**: `full_problem`
- **Status**: Open

---

## Problem Statement & Prompt

Let P be a bounded, full-dimensional convex polytope in R^4 with n facets. Its graph G(P) has the
vertices of P as nodes and the edges (1-dimensional faces) of P as arcs. The combinatorial diameter
of P is the largest graph distance between any two vertices of P.

Produce an explicit rational polytope P in R^4 whose combinatorial diameter is strictly greater
than n - 4, where n is its number of facets.

Submit P in BOTH representations, with every coordinate exact (no floating-point values):

  * an H-representation: an integer matrix A (with n rows and 4 columns) and an integer vector b
    (of length n) such that
        P = { x in R^4 : A x <= b },
    where the n rows of (A, b) are exactly the facets of P (each row is a facet; no row is
    redundant, and no facet is omitted);

  * the full V-representation: the complete list of vertices of P, each given as 4 exact rationals.

Format your answer as a single JSON object with these three keys:

  {
    "A": [[a_11, a_12, a_13, a_14], ..., [a_n1, a_n2, a_n3, a_n4]],
    "b": [b_1, ..., b_n],
    "vertices": [[v_11, v_12, v_13, v_14], ..., [v_m1, v_m2, v_m3, v_m4]]
  }

The entries of A and b must be integers. Each vertex coordinate must be either an integer or a
string holding an exact rational, for example "3/2", "-7", or "0". Do not use floating-point
numbers anywhere in the submission.

Write this JSON object to a file and submit the path to that file.
