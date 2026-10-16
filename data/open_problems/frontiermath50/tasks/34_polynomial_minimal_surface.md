# Polynomial Entire Minimal Graphs

- **Problem ID**: `polynomial-minimal-surface`
- **Field**: Analysis
- **Notability**: Solid result
- **Tags**: Construction - Finite
- **Prompt Type**: `full_problem`
- **Status**: Open

---

## Problem Statement & Prompt

Let u : R^n -> R be a real polynomial. The graph {(x, u(x)) : x in R^n} in R^(n+1) is a minimal
surface if and only if u solves the minimal surface equation

    div( grad u / sqrt(1 + |grad u|^2) ) = 0.

Clearing the positive denominator, this is equivalent to the polynomial identity

    F_u := (1 + |grad u|^2) * (Laplacian u)  -  sum_{i,j=1}^n u_i u_j u_ij  ==  0,

where u_i = du/dx_i, u_ij = d^2u/(dx_i dx_j), |grad u|^2 = sum_i u_i^2, and Laplacian u = sum_i u_ii.

Your task: find a non-affine polynomial u in R[x_1, ..., x_n] for some n >= 8 such that F_u is the
zero polynomial (equivalently, a polynomial u of total degree >= 3 with F_u == 0). Such a u would be
a polynomial entire minimal graph. Coefficients may be any real algebraic numbers.

Solution format:
* Write your answer to a JSON file and submit its path.
* The JSON must be an object with these keys:

  - "n": the number of variables (an integer; must be at least 8).

  - "coefficient_field": how the coefficients are given. Either
      * the string "Q" (the rational numbers), in which case each "coeff" below is a single
        rational string such as "3/2" or "-5"; or
      * a number field object
            {"type": "number_field",
             "min_poly": ["a_0", "a_1", ..., "a_d"],   # rational strings, low degree first:
                                                        #   a_0 + a_1 T + ... + a_d T^d
             "isolating_interval": ["a", "b"]}          # rational strings with a < b
        defining theta as the UNIQUE REAL root of min_poly in the open interval (a, b). min_poly
        must be IRREDUCIBLE over Q and of degree >= 2, and (a, b) must bracket exactly one of its
        real roots. Then theta is a real algebraic number and each "coeff" below is a LIST of
        rational strings [c_0, c_1, ..., c_{m-1}] with m <= d, meaning c_0 + c_1*theta + ... .

  - "polynomial": a list of terms, each {"coeff": <coeff>, "exp": [e_1, ..., e_n]}, representing
        u(x) = sum over terms of  coeff * x_1^e_1 * ... * x_n^e_n.
        Each "exp" is a list of exactly n non-negative integers.

Example (illustrating the format only):
{
  "n": 8,
  "coefficient_field": {"type": "number_field", "min_poly": ["-2", "0", "1"],
                        "isolating_interval": ["1", "2"]},
  "polynomial": [
    {"coeff": ["0", "1"], "exp": [1, 0, 0, 0, 0, 0, 0, 0]},
    {"coeff": ["3/2"],    "exp": [0, 1, 0, 0, 0, 0, 0, 0]}
  ]
}
Here min_poly is T^2 - 2 and isolating_interval (1, 2) selects theta = +sqrt(2) (the interval
(-2, -1) would select -sqrt(2)), so this encodes u = sqrt(2)*x_1 + (3/2)*x_2.
