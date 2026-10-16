# The Smooth Four-Dimensional Poincaré Conjecture

- **Problem ID**: `smooth-poincare-4d`
- **Field**: Topology/Geometry
- **Notability**: Breakthrough
- **Tags**: Construction - Finite, Counterexample
- **Prompt Type**: `full_problem`
- **Status**: Open

---

## Problem Statement & Prompt

Your task is to disprove the smooth 4-dimensional Poincaré conjecture (SPC4) by exhibiting an explicit knot that is a counterexample, together with a certificate proving it.

## Background

A knot K is *slice* if it bounds a smoothly embedded disk in the 4-ball. The Rasmussen s-invariant (and the related invariants s mod 2, s mod 3, the odd Sq^1 invariant, and the sl(3) s-invariant, all computable from Khovanov-type homology) vanish for every slice knot; so a knot with any of these invariants nonzero is not slice.

Two knots K_1 and K_2 have *the same 0-surgery* if the 3-manifolds obtained by 0-framed Dehn surgery are related by a sequence of Kirby moves. Following the Manolescu–Piccirillo program: suppose K_1 and K_2 have the same 0-surgery, K_1 is slice, and K_2 has a nonzero Rasmussen-type invariant. If SPC4 held, the 0-surgery equivalence together with K_1 being slice would force K_2 to be slice too — contradicting the nonzero invariant. Such a pair (K_1, K_2) therefore disproves SPC4.

You are asked to produce such a pair and certify all three facts explicitly.

## What you must certify

1. **Kirby equivalence.** A sequence of framed links, starting from the 0-framed K_1 and ending at the 0-framed K_2, in which each link is obtained from the previous one by a single explicit Kirby move.
2. **K_1 is slice.** An explicit slice-disk movie for K_1: a sequence of births, band moves, and Reidemeister moves taking K_1 to a crossingless unlink.
3. **K_2 is not slice.** This is checked for you, by running KnotJob on K_2 and requiring at least one of the Rasmussen-type invariants above to be nonzero.

KnotJob is available in your environment, so you can check a candidate K_2 yourself before submitting. `java` is on the PATH and the jar is at `/opt/knotjob/KnotJob/KnotJob.jar`. Write the PD code to a `.txt` file as a single line in Python list-of-lists form, for example

```
[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]]
```

then run the same command the verifier uses:

```
java -Xmx16g -jar /opt/knotjob/KnotJob/KnotJob.jar k2.txt -s0 -s2 -s3 -sqo -sl0
```

It writes the five invariants to `k2.txt_s0_s2_s3_sl3s0_sqo` beside the input file. A K_2 whose invariants all vanish will be rejected, so it is worth checking before you build the rest of the certificate.

The cost of this computation depends on the *diagram* you submit, not on the knot: two diagrams of the same knot with the same number of crossings can differ by three orders of magnitude in runtime. The verifier allows 30 minutes for it, and gives up (rejecting the certificate) beyond that. You control the diagram: `k2_pd` only has to be planar-isotopic to the last link of your Kirby sequence, and Reidemeister moves are legal Kirby moves, so you can extend the sequence with R-moves to end on whichever diagram of K_2 you like. If KnotJob is slow on your K_2, simplify that diagram and end the sequence there rather than assuming the invariant is out of reach.

## Diagram conventions

A knot or link is given by a PD (planar diagram) code: a list of crossings, each a list of four nonnegative integer strand labels listed counter-clockwise around the crossing. Every strand label must appear exactly twice. Plain PD codes cannot represent crossingless split unknot components, so where a framed link needs them they are recorded separately (see `split_unknot_framings` below).

## Certificate format

Write your answer to a file named `submission.json` containing a single JSON object:

    {
      "k1_pd": [[...], ...],
      "k2_pd": [[...], ...],
      "kirby_sequence": {
        "links": [
          {"pd": [[...], ...], "framings": [0], "split_unknot_framings": []},
          ...
        ],
        "moves": [ <kirby_move>, ... ]
      },
      "slice_certificate": {
        "initial": [[...], ...],
        "moves": [ <slice_move>, ... ],
        "final_unlink_components": 1
      }
    }

Optional top-level `comment` and `metadata` keys are ignored.

### kirby_sequence

`links` is a list of framed links; each has a `pd` code, a `framings` list (one integer framing per component), and `split_unknot_framings` (framings of any isolated split unknot components not expressible in the PD code). There must be exactly one fewer move than links: `moves[i]` is applied to `links[i]` and its result is compared to `links[i+1]` up to planar isotopy, relabeling, and a component permutation compatible with the framings. Supported Kirby moves:

- `R1`, `R2`, `R3` — Reidemeister moves.
- `handle_slide` — slide one component over a framed push-off of another. The band is written on the *expanded* diagram obtained after adding the framed push-off.
- `pm1_unknot` — add or remove an isolated ±1-framed unknot.

### slice_certificate

`initial` must be K_1 itself (a single-component diagram; extra initial components are rejected — introduce every auxiliary unknot with an explicit `birth`). Each move carries its `type`, the data locating it, and a `result` (the diagram after the move, with explicit `component_labels` where components must be tracked). Supported slice moves are `birth`, `band`, and `R1`/`R2`/`R3`. The movie must be a connected surface and satisfy

    A - B + C = 1

where A is the number of births, B the number of band moves, and C the number of final unlink components.

## What is accepted

The verifier accepts (exit 0) if and only if all of the following hold:

- `k1_pd` and `k2_pd` are each valid PD codes of a single-component knot;
- the first Kirby link is the 0-framed K_1 and the last is the 0-framed K_2, each with no extra split components;
- every Kirby move is locally valid and each result matches the next link up to planar isotopy;
- the slice certificate starts at K_1, every local move is valid, the surface is connected, and A − B + C = 1;
- KnotJob reports at least one nonzero requested Rasmussen-type invariant for K_2.

Every step must be spelled out explicitly; the verifier never applies random or global simplification, and all diagram comparisons are exact.
