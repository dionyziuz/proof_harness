# A shape with Heesch number 7

- **Problem ID**: `heesch-number`
- **Field**: Topology/Geometry
- **Notability**: Solid result
- **Tags**: Construction - Finite, Record Breaking
- **Prompt Type**: `full_problem`
- **Status**: Open

---

## Problem Statement & Prompt

# Record-Setting Heesch Polyform

## Goal
Submit one connected, hole-free unmarked polyform with at most **200 cells**, compact enough for the verifier (`span_x + span_y <= 29`), and a witness patch showing **Heesch number at least 7**. The submitted polyform should not admit an isohedral tiling.

## Submission File
Return a single plain-text file containing one tile record in the compact `heesch-sat` text format.

The record must have this structure:

```text
GRID x1 y1 x2 y2 ... xn yn
~ hc hh P
N
level1 <a,b,c,d,e,f>
level2 <a,b,c,d,e,f>
...
levelN <a,b,c,d,e,f>
```

`P` is the number of witness patches; it is `1` in the common case (`hh == hc`) and `2` when you also certify `hh == hc + 1`. The block above shows a single patch. When `P = 2`, append a second count line `M` followed by `M` placement lines for the hole-allowed patch.

## Line-by-Line Format
`GRID x1 y1 x2 y2 ... xn yn`

`GRID` is the grid abbreviation. Allowed abbreviations include `O` (polyomino/square grid), `H` (polyhex/hexagonal grid), and `I` (polyiamond/triangular grid). Use `O` unless you intentionally submit a different supported polyform family. The remaining integers are occupied cell coordinates, not boundary vertices and not an outline walk. For example:

```text
O 0 0 1 0
```

describes a domino: one cell at `(0,0)` and one adjacent cell at `(1,0)`.

Every coordinate pair must name a cell of the grid you chose. `O` and `H` have a cell at every pair, but the other families do not: in `I`, for instance, a cell needs `x` and `y` to be congruent mod 3 and congruent to 0 (upward triangle) or 1 (downward triangle), so `I 1 0` names no cell. The verifier rejects a shape that lists a position which is not a cell, and likewise a placement whose transform is not a symmetry of the grid and so moves the tile off the grid's cells.

`~ hc hh P`

This status line declares a non-tiler witness record.

- `hc` is the claimed hole-free Heesch number.
- `hh` is the claimed Heesch number when holes are allowed in the outer corona. A non-tiler always has `hh == hc` or `hh == hc + 1`. If you are not demonstrating the hole variant, set `hh = hc`.
- `P` is the number of witness patches that follow. Set `P = 1` when `hh == hc` (just the hole-free patch). Set `P = 2` when `hh == hc + 1`: give the hole-free patch reaching corona `hc` first, then a second patch reaching corona `hh` whose outer corona may contain holes.
- Use `7 <= hc <= 12`. Set `hh = hc` (with `P = 1`) unless you also want to certify the hole variant `hh = hc + 1` (with `P = 2`). Acceptance depends only on the recomputed hole-free `hc` reaching the threshold.

`N`

The number of placed copies of the tile in the following patch, including the central copy. Each patch begins with its own count line and is followed by that many placement lines.

`level <a,b,c,d,e,f>`

Each placement line gives one copy of the tile.

- `level` is the corona index of that copy: `0` for the central seed, `1` for the first surrounding corona, `2` for the second, and so on.
- `<a,b,c,d,e,f>` is the affine lattice transform applied to every occupied cell `(x,y)` of the base tile:

```text
x' = a*x + b*y + c
y' = d*x + e*y + f
```

The matrix part `[[a,b],[d,e]]` should be a grid symmetry: a rotation and/or reflection valid for the selected grid. The translation part is `(c,f)`.

The central tile should normally be:

```text
0 <1,0,0,0,1,0>
```

## Constraints
- Use one tile record per submission file.
- Every listed coordinate pair must be a cell of the chosen grid (see the shape line above).
- The shape must be connected and hole-free.
- The shape must have at most 200 occupied cells.
- The shape must be compact enough for the current verifier: its coordinate bounding-box spans must satisfy `span_x + span_y <= 29`. Absolute translation does not matter; the solver normalizes position before classification. More elongated shapes may be mathematically interesting, but this verifier rejects them to avoid a known fixed-bitmap SAT failure mode.
- The witness patch must include all copies needed to demonstrate coronas through level `hc`.
- The patch must have no overlapping placed cells.
- Keep coordinates reasonably small so automated checking can finish.
- The verifier independently recomputes the classification with a maximum search level of 12. A
  shape that reaches that search bound is classified as inconclusive and is not accepted, even if
  its submitted witness is valid.

## Toy Example
This example is not a solution to the challenge; it only illustrates syntax.

```text
O 0 0 1 0
~ 1 1 1
3
0 <1,0,0,0,1,0>
1 <1,0,0,0,1,1>
1 <1,0,0,0,1,-1>
```

Submit your own record-setting high-Heesch construction instead of the toy example.
