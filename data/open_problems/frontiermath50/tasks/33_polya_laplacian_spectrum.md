# Pólya's Conjecture for the Dirichlet Laplacian

- **Problem ID**: `polya-laplacian-spectrum`
- **Field**: Spectral geometry
- **Notability**: Major advance
- **Tags**: Construction - Finite, Counterexample
- **Prompt Type**: `full_problem`
- **Status**: Open

---

## Problem Statement & Prompt

Let Omega be a bounded open set in R^2 with Lipschitz boundary, and let

    0 < lambda_1(Omega) <= lambda_2(Omega) <= ...

be the Dirichlet eigenvalues of the positive Laplacian on Omega, counted with
multiplicity. Consider the inequality

    lambda_k(Omega) >= 4*pi*k / |Omega|,

where |Omega| is the area of Omega.

## Task

Find a bounded polygonal domain P in R^2 with rational vertices, together with an
index k >= 1, that violates this inequality:

    |P| * lambda_k(P) < 4*pi*k.

P is an outer simple polygon with zero or more polygonal holes removed from it, so
it may be multiply connected. Supply a strictly positive rational margin epsilon;
the submission is accepted only if the violation holds with room to spare:

    |P| * lambda_k(P) < 4*pi*k - epsilon.

## Submission

Submit the path to a JSON file with this shape:

    {
      "vertices": [[x1, y1], [x2, y2], ..., [xm, ym]],
      "holes": [ [[..], [..], [..]], ... ],
      "k": <positive integer>,
      "epsilon": <positive rational>,
      "refinement": <non-negative integer>
    }

  * "vertices": the outer simple polygon, vertices in cyclic order.
  * "holes" (optional, default none): simple polygons removed from the outer one.
    Each hole must lie strictly inside the outer polygon, and holes must be
    pairwise disjoint.
  * "k", "epsilon": as above.
  * "refinement": mesh refinement level used to bound lambda_k(P); larger values
    give a tighter bound at higher cost.

Each coordinate and epsilon is an exact rational, given as an integer or as a
string that is a fraction "p/q" or a plain decimal such as "0.25". Floating-point
numbers and scientific notation are not accepted.
