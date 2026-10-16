# The Rank of the $4\times4$ Matrix-Multiplication Tensor

- **Problem ID**: `matrix-multiplication`
- **Field**: Combinatorics
- **Notability**: Moderately interesting
- **Tags**: Algorithm, Record Breaking
- **Prompt Type**: `full_problem`
- **Status**: Open

---

## Problem Statement & Prompt

The matrix multiplication tensor M<4> encodes the bilinear map that multiplies two 4x4 matrices over the complex numbers. A rank-r decomposition of M<4> is a list of r triples (L_s, R_s, P_s) for s = 1, ..., r, where

  - L_s is a linear form on the entries of a 4x4 matrix A,
  - R_s is a linear form on the entries of a 4x4 matrix B,
  - P_s is a 4x4 matrix of coefficients,

such that for ALL 4x4 matrices A and B,

  A B = sum_{s=1}^r L_s(A) * R_s(B) * P_s.

Equivalently, this is an algorithm that multiplies two 4x4 matrices using at most r scalar multiplications (the products L_s(A) * R_s(B)), where additions and multiplications by fixed constants are free.

Your task is to exhibit a decomposition of rank r <= 47.

Coefficients may be complex algebraic numbers. You will certify them exactly by working in a coefficient ring Q[t]/(f), where f(t) is a polynomial with rational coefficients: each coefficient is represented as an element of Q[t]/(f), and choosing any complex root of f recovers the complex decomposition.

Output your answer as a file containing a single JSON object with these keys:

  {
    "r": <integer rank, with r <= 47>,
    "f": [c0, c1, ..., cd],
    "L": [ <r rows, each a list of 16 entries> ],
    "R": [ <r rows, each a list of 16 entries> ],
    "P": [ <r rows, each a list of 16 entries> ]
  }

Format details:

  - "f" lists the coefficients of f(t) = c0 + c1*t + ... + cd*t^d in ascending degree, with cd != 0 and degree d >= 1. (Use f = [0, 1], i.e. f(t) = t, to work over the plain rationals Q.)

  - Each of "L", "R", "P" is an array of exactly r rows, and each row has exactly 16 entries, one per matrix cell in row-major order: column index 4*(i-1) + (j-1) corresponds to the (i, j) cell (rows and columns indexed 1..4). For row s, the entries of L give L_s as the linear form L_s(A) = sum_{i,j} L[s][4*(i-1)+(j-1)] * A_{ij}; likewise R[s] gives R_s(B) = sum_{j,k} R[s][...] * B_{jk}; and P[s] gives the matrix P_s, so (P_s)_{ik} = P[s][4*(i-1)+(k-1)].

  - Each entry is an element of Q[t]/(f), written as a list of rational coefficients [a0, a1, ...] meaning a0 + a1*t + ... (degree strictly less than d). A bare scalar (e.g. "1" or 3) is accepted as a constant. Rational numbers must be written as integers or as exact fraction strings like "3/2" or "-1/4". Floating-point numbers are NOT accepted, since the check is exact; this includes decimal strings like "0.5" and exponent notation like "1e3" (write "1/2" and "1000" instead).
