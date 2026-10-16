# The Slice–Ribbon Conjecture and the GST Knot

- **Problem ID**: `slice-ribbon`
- **Field**: Knot theory
- **Notability**: Solid result
- **Tags**: Construction - Finite, Counterexample
- **Prompt Type**: `full_problem`
- **Status**: Open

---

## Problem Statement & Prompt

Your task is to prove that the GST knot is ribbon by exhibiting an explicit ribbon disk.

Starting from the diagram K below, exhibit a sequence of Reidemeister moves and ribbon band moves that ends in a crossingless unlink. If your sequence uses n band moves, the final unlink must have exactly n + 1 components (each ribbon band move splits off one component). Such a sequence is exactly a ribbon-disk certificate for K.

## The starting diagram

K is given by this PD (planar diagram) code — a list of 4-tuples, one per crossing, listing the four strand labels around the crossing counter-clockwise. Your certificate must begin from exactly this diagram:

GST_START_PD = [
  [83, 94, 84, 95], [92, 81, 93, 82], [72, 95, 73, 96], [73, 84, 74, 85], [93, 70, 94, 71],
  [82, 71, 83, 72], [2, 10, 3, 9], [17, 5, 18, 4], [7, 19, 8, 18], [8, 23, 9, 24],
  [3, 25, 4, 24], [31, 1, 32, 96], [27, 12, 28, 13], [38, 5, 39, 6], [6, 37, 7, 38],
  [1, 46, 2, 47], [28, 44, 29, 43], [47, 33, 48, 32], [51, 21, 52, 20], [21, 53, 22, 52],
  [34, 53, 35, 54], [50, 35, 51, 36], [54, 50, 55, 49], [58, 20, 59, 19], [22, 60, 23, 59],
  [33, 60, 34, 61], [57, 36, 58, 37], [61, 49, 62, 48], [55, 63, 56, 62], [63, 57, 64, 56],
  [13, 69, 14, 68], [42, 69, 43, 70], [29, 75, 30, 74], [76, 12, 77, 11], [15, 79, 16, 78],
  [77, 27, 78, 26], [40, 79, 41, 80], [75, 44, 76, 45], [30, 86, 31, 85], [87, 11, 88, 10],
  [16, 90, 17, 89], [88, 26, 89, 25], [39, 90, 40, 91], [86, 45, 87, 46], [65, 80, 66, 81],
  [64, 91, 65, 92], [67, 15, 68, 14], [66, 41, 67, 42],
]

## Certificate format

Write your answer to a file containing a single Python literal — a dictionary:

    {
        "initial": [...],                 # must equal GST_START_PD (lists or tuples both fine)
        "moves": [ <move>, <move>, ... ], # the explicit sequence
        "final_unlink_components": n + 1, # optional; inferred if omitted
    }

Each move is a dictionary carrying its `type`, the data identifying where it is applied, and a `result`: the PD code of the diagram *after* the move. The verifier applies the move to the current diagram itself and checks your `result` agrees up to planar isotopy. A result may be a PD code, `{"pd": [...]}`, or — only for the final crossingless unlink — `{"unlink": n}` (or a plain `[]`, whose component count is then inferred). A PD code cannot record a crossingless (split, unknotted) component, so if a band splits off a trivial circle while other components still have crossings, write that intermediate result as `{"pd": [...], "unlink": k}`, meaning the PD diagram together with `k` extra crossingless unknot components.

Supported moves:

    {"type": "R1", "direction": "remove", "crossing": i, "side": s, "result": [...]}
    {"type": "R1", "direction": "add", "strand": (c, s), "hand": "left"|"right", "result": [...]}
    {"type": "R2", "direction": "remove", "crossing": i, "side": s, "result": [...]}
    {"type": "R2", "direction": "add", "strands": [(c1, s1), (c2, s2)], "result": [...]}
    {"type": "R3", "triple": [(c1, s1), (c2, s2), (c3, s3)], "result": [...]}
    {"type": "band", "band": <spec>, "result": [...]}

Crossing/strand references are (crossing_index, strand_index) with strand_index in {0,1,2,3}. An R3 triple names the three corners of the triangular face the move is applied across; the corners may be listed in any order. A band spec is a Spherogram compressed band string (e.g. "223505_1_1") or a triple accepted by `spherogram.links.bands.core.Band`.
