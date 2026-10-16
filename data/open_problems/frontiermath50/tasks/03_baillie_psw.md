# A Baillie–PSW Pseudoprime

- **Problem ID**: `baillie-psw`
- **Field**: Number theory
- **Notability**: Moderately interesting
- **Tags**: Counterexample, Integer Solution
- **Prompt Type**: `full_problem`
- **Status**: Open

---

## Problem Statement & Prompt

The Baillie-PSW primality test combines two probable-prime tests and declares an odd
integer n > 1 to be prime when it passes both. No composite integer is known to pass the Baillie-PSW test, and it has been verified that none exists below 2^64; whether any composite passes the
Baillie-PSW test is a long-standing open problem. Such a composite is called a Baillie-PSW
pseudoprime.

The two component tests are defined as follows.

Strong Fermat pseudoprime to base 2.
  Write n = d * 2^s + 1 with d odd. We say n is a strong Fermat
  pseudoprime to base 2 if either
      2^d ≡ 1 (mod n),   or
      2^(d * 2^r) ≡ -1 (mod n)  for some integer r with 0 ≤ r < s.
  (This is the strong, or Miller-Rabin, probable-prime test to base 2. See OEIS A001262.)

Strong Lucas pseudoprime (Selfridge Method A).
  Set P = 1. Let D be the first element of the sequence 5, -7, 9, -11, 13, ... (absolute value
  increasing by 2, sign alternating) such that the Jacobi symbol (D / n) equals -1, and set
  Q = (1 - D) / 4. Define the Lucas sequences U_k and V_k by
      U_0 = 0, U_1 = 1,  U_k = P*U_{k-1} - Q*U_{k-2},
      V_0 = 2, V_1 = P,  V_k = P*V_{k-1} - Q*V_{k-2}.
  Write n + 1 = e * 2^t with e odd. We say n is a strong Lucas pseudoprime if either
      U_e ≡ 0 (mod n),   or
      V_(e * 2^r) ≡ 0 (mod n)  for some integer r with 0 ≤ r < t.
  (See OEIS A217255.)

An odd composite n passes the Baillie-PSW test if it is BOTH a strong Fermat pseudoprime to
base 2 AND a strong Lucas pseudoprime.

Your task: find an odd composite integer n that passes the Baillie-PSW test.

Solution format:
* Write your answer to a JSON file and submit its path.
* The JSON must be an object with these integer keys:
  - "factor1", "factor2": two integers, each greater than 1. The candidate is their product
    n = factor1 * factor2; these factors certify that n is composite (they need not be prime).
    There is no bound on the size of the factors; arbitrary-precision arithmetic is used.

The verifier forms n = factor1 * factor2 with both factors > 1 (so n is a certified odd
composite), and then runs both component tests on n. Your submission is accepted if and only if
n is a certified odd composite that passes both tests.

Example submission (illustrating the format only — this n does not pass the test):
{
  "factor1": 23,
  "factor2": 89
}
