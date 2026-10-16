# Conway's Thrackle Conjecture

- **Problem ID**: `thrackles`
- **Field**: Graph theory
- **Notability**: Solid result
- **Tags**: Construction - Finite
- **Prompt Type**: `full_problem`
- **Status**: Open

---

## Problem Statement & Prompt

# Task: find a counterexample to Conway's thrackle conjecture

## Background

A **thrackle** is an embedding of a graph in the plane satisfying the following property: every pair of distinct edges meets **exactly once**. Two edges can meet in one of two ways:

- **At a shared endpoint** — if two edges share a vertex, that vertex counts as their one meeting.  They must not intersect anywhere else.
- **At an interior crossing** — if two edges share no endpoints, they must cross transversely at exactly one interior point of both arcs.

Each edge is a **Jordan arc**: a continuous, non-self-intersecting curve between its two endpoints.

**Conway's thrackle conjecture** states that any thrackle has at most as many edges as vertices.  This conjecture has been open since the 1960s.

## Task

Find a counterexample: a valid thrackle whose number of edges is strictly greater than its number of vertices.

## Submission format

Write your answer to a file named `submission.json`, as a JSON object with the following structure:

```json
{
  "vertices": {
    "<name>": [x, y]
  },
  "edges": {
    "<name>": {
      "endpoints": ["<v1>", "<v2>"],
      "waypoints": [[x, y], ...]
    }
  }
}
```

- **vertices**: a mapping from vertex names to 2-D coordinates `[x, y]`.
- **edges**: a mapping from edge names to arc descriptions.
  - `endpoints`: a list of exactly two vertex names.
  - `waypoints`: an ordered list of interior points along the arc, from `v1` to `v2`.  The arc is treated as the polyline `v1 → waypoints[0] → … → v2`.  Omit `waypoints` (or set it to `[]`) for a straight-line edge.
  - The submitted polyline for each edge must itself be a Jordan arc: consecutive points must be distinct, and the polyline must not self-intersect, self-touch, or overlap itself.

Coordinates must be finite JSON numbers, parsed exactly as rational values (e.g. `0`, `1`, `-3`, `0.5`, `1.25`, `1e-9`).  Vertex and edge names may be any non-empty strings.  No two vertices may share the same coordinates.

**General-position rule**: no two polyline vertices (graph vertices or waypoints) from different edges may coincide, unless they are the same named graph vertex that both edges legitimately share as an endpoint.  In particular, crossings between edges must occur at smooth interior points of both arcs, not at a waypoint of either edge.  Any coincidence of polyline vertices is treated as a geometric defect and rejected.
