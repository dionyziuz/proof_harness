# Sensitivity versus Degree for Boolean Multilinear Polynomials

- **Problem ID**: `degree-sensitivity-boolean`
- **Field**: Combinatorics
- **Notability**: Solid result
- **Tags**: Construction - Finite, Improved Bound
- **Prompt Type**: `full_problem`
- **Status**: Open

---

## Problem Statement & Prompt

Construct a multilinear, integer-coefficient function P, on n variables, with 1 < n ≤ 100, with degree deg(P) > 1, that is Boolean on the cube, i.e., P(x) = 0 or 1 for all x in {0,1 }^n, and that has

|P(0, ..., 0) - P(1, 0, ..., 0)| + |P(0, ..., 0) - P(0, 1, 0, ..., 0)| + ... + |P(0, ..., 0) - P(0, ..., 0, 1)| = deg(P)^a

for some a > log(6)/log(3).

Provide your answer as an expression using variables like x1, x2, etc. For operators use +, -, and *. You may also use parentheses. Example: x1 + x2 + x3 - (x1*x2 + x1*x3 + x2*x3) + x1*x2*x3
