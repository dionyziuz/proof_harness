# Class Group Algorithms

- **Problem ID**: `quadratic-field-class-number`
- **Field**: Algebraic number theory
- **Notability**: Solid result
- **Tags**: Construction - Family
- **Prompt Type**: `full_problem`
- **Status**: Open

---

## Problem Statement & Prompt

# Imaginary quadratic fields with a prescribed class number

Given a target class number $h$, find a negative fundamental discriminant $D$ such
that the imaginary quadratic field $\mathbb{Q}(\sqrt{D})$ has class number exactly
$h$.

## Submission format

You will submit a single Python file. It may use `sage`. When submitting, you may
also specify a regime `s` (an integer with 2<=s<=100) constraining the challenges
your program will receive (such that 2^s <= h < 2^(s+1) where h is the challenge);
note that for s<90 it won't be scored.

## Protocol

The verifier repeatedly:

1. picks a challenge class number `h`: if you specified `s`, then $2^s \le h < 2^{s+1}$;
   otherwise `h` is chosen randomly over the whole range ($2^{90} \le h < 2^{100}$);
2. runs your program in a fresh process, with `h` in decimal as the sole command-line
   argument, using the same Python interpreter that runs the verifier. That is Sage's
   Python, which is not necessarily the `python` on your `PATH` — find it with
   `sage -c 'import sys; print(sys.executable)'` and test your submission with that
   interpreter, or `import sage` may fail;
3. reads the last non-empty line of your program's stdout as your answer: a negative
   fundamental discriminant `D` >= -100000*h*h whose field has class number `h`.
   Earlier lines of stdout are ignored, so printing progress information is fine
   (`D` without the minus sign is also accepted).

## Verification & scoring

The verifier accepts $D$ when it is a negative fundamental discriminant with
$|D| \le 100000 h^2$ and class number exactly $h$.

If there is sufficient time available, your program may be run on multiple challenges.
It will be considered correct if it answers any of the challenges correctly within
the available time.

The total time limit is 30 minutes (at most 100 challenges are attempted); each
invocation of your program may use whatever remains of it.

## Example submission format

```
import sys

from sage.all import ZZ

h = ZZ(sys.argv[1])
D = find_discriminant(h)   # you need to implement this function!
# Your answer: a negative fundamental discriminant whose field has class number h
print(D)
```
