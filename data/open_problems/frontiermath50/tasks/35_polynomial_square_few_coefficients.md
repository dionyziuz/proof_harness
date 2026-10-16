# Dense Polynomials Whose Squares Are Sparse

- **Problem ID**: `polynomial-square-few-coefficients`
- **Field**: Number theory
- **Notability**: Solid result
- **Tags**: Construction - Finite
- **Prompt Type**: `full_problem`
- **Status**: Open

---

## Problem Statement & Prompt

Your task is to find a specific polynomial, if one exists.

A polynomial p(x) = a_0 + a_1*x + ... + a_n*x^n is called *dense* if every coefficient
a_0, a_1, ..., a_n is non-zero (i.e. it has n+1 non-zero coefficients, one at every
degree from 0 to n).

Your goal is to find a dense polynomial p with integer coefficients such that p(x)^2 has
fewer than sqrt(n) non-zero coefficients, where n is the degree of p.

In other words:
  - p has degree n
  - every coefficient of p is a non-zero integer
  - the number of non-zero coefficients of p^2 is strictly less than sqrt(n)

This is an open problem — it is unknown whether such a polynomial exists. You may attempt
either to find one or to argue why it cannot exist.

If you produce a polynomial, write it to a .json file as an array of integers [a_0, a_1, ..., a_n]
where a_0 is the constant term and a_n is the leading coefficient, then submit
the path to that file. Every entry must be a non-zero integer. Use plain integer literals
(e.g. 3, -7), not scientific notation (e.g. 3e0), as the latter is parsed as a float and
will be rejected.

Example file contents (this is just an illustration, not a solution):
  [1, -1, 2, -3, 1]
represents 1 - x + 2x^2 - 3x^3 + x^4, which has degree 4 and 5 non-zero coefficients.
