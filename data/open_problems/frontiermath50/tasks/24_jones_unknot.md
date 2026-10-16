# The Jones Unknot Conjecture

- **Problem ID**: `jones-unknot`
- **Field**: Knot theory
- **Notability**: Major advance
- **Tags**: Construction - Finite, Counterexample
- **Prompt Type**: `full_problem`
- **Status**: Open

---

## Problem Statement & Prompt

Your task is to disprove the Jones unknot conjecture by exhibiting a nontrivial knot whose normalized Jones polynomial is 1.

The Jones unknot conjecture asserts that the unknot is the only knot whose normalized Jones polynomial equals 1. It is a famous open problem. A counterexample is a knot K that is genuinely knotted — it has positive Seifert genus, so it is not the unknot — yet has normalized Jones polynomial exactly 1.

Concretely: produce a planar diagram of a single knot K such that

- the normalized Jones polynomial of K is exactly 1 (the unknot's value), and
- K is not the unknot: its knot Floer homology has positive Seifert genus.

## Diagram format

Give K as a PD (planar diagram) code — a list of crossings, each a 4-entry list of nonnegative integer edge labels listing the four strand ends around the crossing. The four entries are listed counterclockwise starting at the incoming under-strand (SnapPy/Spherogram's `PD_code()` convention); a diagram whose crossings violate this convention is rejected as malformed input. Each edge label must appear exactly twice, and the diagram must be classically planar and represent exactly one knot component. For example, the trefoil is:

    [[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]]

The diagram may have at most 100 crossings, and it must be reduced: it must admit no simplifying Reidemeister I or II move. Equivalently, SnapPy/Spherogram's `Link(pd).simplify(mode='basic')` must return `False` on your diagram — and if it returns `True`, the simplified link it leaves behind is a qualifying diagram of the same knot.

## Compute budget

The crossing limit is not a compute budget. Verification computes the Jones polynomial and knot Floer homology of the diagram you submit, and the cost of both — knot Floer homology especially — is driven by the Morse width/girth of the *diagram*, not by its crossing number: across reduced 55-to-80-crossing diagrams we measured knot Floer runtimes from under a second to several minutes, and the diagrams with more crossings were often the faster ones. The verifier allows 900 seconds for any single one of these computations and 1800 seconds for all of them together.

You control the diagram: present your knot as a low-width diagram — a closed braid on few strands, or a diagram simplified with `Link(pd).simplify(mode='global')` — rather than assuming that a knot within the crossing limit is within reach.

## Submission format

Write your answer to a file whose entire contents are the PD code as a single Python literal (a list of 4-element lists of nonnegative integers), for example:

    [[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]]

Submit the path to that file with the submit tool.
