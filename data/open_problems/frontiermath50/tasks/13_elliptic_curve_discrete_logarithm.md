# The Elliptic-Curve Discrete Logarithm Problem (ECDLP)

- **Problem ID**: `elliptic-curve-discrete-logarithm`
- **Field**: Number theory
- **Notability**: Major advance
- **Tags**: Algorithm
- **Prompt Type**: `full_problem`
- **Status**: Open

---

## Problem Statement & Prompt

Let p be a prime. You are given a curve

    E/F_p : y^2 = x^3 + a*x + b,

a point P in E(F_p) of prime order n, and a point

    Q = xP,

where xP denotes elliptic-curve scalar multiplication: the point obtained by adding P to
itself x times under the group law on E. (The scalar x here is an integer, distinct from the
affine coordinate variable x in the curve equation above.)

The Elliptic Curve Discrete Logarithm Problem is to recover x from the data (p, a, b, P, Q).

The public challenge is p, a, b, P, Q, n, h, where #E(F_p) = h*n and h is small:

    p = 92031864238688105485400538143121826382268912998421731683072033206257804487869
    a = 66780434313013275323552928748270304481793221171615836760742707201070562072251
    b = 49926646401131159709065076432340744650532242327390476058390067276099410301761
    P = (38644201185363417798191368696978811687737990190927559170532558153854800741642, 44805905068903897561820523795034068749084951450263356538447620424400005028012)
    Q = (54096648540016620506320945354369315155659910460053503439381018376158456438152, 23176800742290374932252879903926227937053123957837615098744936554616726258932)
    n = 18406372847737621097080107628624365276513063062857122317960690589425712086189
    h = 5

## Task

Recover x (the private answer): the integer x with 1 <= x < n such that Q = xP.

## Submission

Submit x by calling the `submit_answer` tool with `secret_x` set to the decimal integer,
written as a string (for example "123456789").
