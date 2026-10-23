"""Generic Mathematical Verification & Empirical Diagnostics Toolkit.

Provides domain-agnostic, rigorous computational verification functions preloaded into
the Python REPL to stress-test mathematical assertions, asymptotic bounds, probabilistic
dependencies, and combinatorial invariants without problem-specific heuristics.
"""

from __future__ import annotations

import math
import random
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

try:
    import sympy as sp
except ImportError:
    sp = None

try:
    import networkx as nx
except ImportError:
    nx = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import z3
except ImportError:
    z3 = None


def check_asymptotic_dominance(
    f: Union[str, Any],
    g: Union[str, Any],
    var: str = "n",
) -> Dict[str, Any]:
    """Rigorous symbolic comparison of asymptotic growth rates between f(n) and g(n) as n -> oo.
    
    Determines whether f = o(g), f = O(g), f = Theta(g), f = Omega(g), or f = omega(g).
    """
    if sp is None:
        return {"error": "SymPy not available"}
    
    try:
        n = sp.Symbol(var, positive=True, real=True)
        local_dict = {var: n}
        f_sym = sp.sympify(f, locals=local_dict) if isinstance(f, str) else f.subs(sp.Symbol(var), n)
        g_sym = sp.sympify(g, locals=local_dict) if isinstance(g, str) else g.subs(sp.Symbol(var), n)
        
        ratio = f_sym / g_sym
        lim = sp.limit(ratio, n, sp.oo)
        
        is_o = bool(lim == 0)
        is_omega = bool(lim == sp.oo or lim == -sp.oo)
        is_Theta = bool(lim != 0 and not is_omega and hasattr(lim, 'is_finite') and lim.is_finite)
        is_O = bool(not is_omega)
        
        res = {
            "f": str(f_sym),
            "g": str(g_sym),
            "ratio": str(ratio),
            "limit_n_to_inf": str(lim),
            "is_o": is_o,
            "is_O": is_O,
            "is_Theta": is_Theta,
            "is_omega": is_omega,
        }
        if is_o:
            res["conclusion"] = f"{f_sym} = o({g_sym}) (strictly dominated, ratio -> 0)"
        elif is_Theta:
            res["conclusion"] = f"{f_sym} = Theta({g_sym}) (same asymptotic scale, ratio -> {lim})"
        elif is_omega:
            res["conclusion"] = f"{f_sym} = omega({g_sym}) (DIVERGES faster than {g_sym}, ratio -> oo)"
        elif is_O:
            res["conclusion"] = f"{f_sym} = O({g_sym}) (upper-bounded by {g_sym})"
        else:
            res["conclusion"] = f"Limit evaluated to {lim}"
        return res
    except Exception as e:
        return {"error": f"Asymptotic analysis failed: {e}"}


def check_event_covariance(
    sampler_fn: Callable[[], Tuple[int, int]],
    n_trials: int = 2000,
) -> Dict[str, Any]:
    """Tests empirical dependency between two binary indicator events A and B.
    
    sampler_fn() must return a tuple (indicator_A, indicator_B) where each is 0 or 1.
    Evaluates Pr[A], Pr[B], Pr[A and B], Cov(A, B), and Pearson correlation.
    """
    if n_trials < 50:
        n_trials = 50
        
    count_A = 0
    count_B = 0
    count_AB = 0
    
    for _ in range(n_trials):
        a, b = sampler_fn()
        ia = 1 if a else 0
        ib = 1 if b else 0
        count_A += ia
        count_B += ib
        count_AB += (ia * ib)
        
    p_A = count_A / n_trials
    p_B = count_B / n_trials
    p_AB = count_AB / n_trials
    
    cov = p_AB - (p_A * p_B)
    var_A = p_A * (1.0 - p_A)
    var_B = p_B * (1.0 - p_B)
    
    corr = 0.0
    if var_A > 1e-9 and var_B > 1e-9:
        corr = cov / math.sqrt(var_A * var_B)
        
    is_dependent = abs(corr) > 0.05
    
    return {
        "trials": n_trials,
        "P(A)": round(p_A, 5),
        "P(B)": round(p_B, 5),
        "P(A and B)": round(p_AB, 5),
        "P(A)*P(B)": round(p_A * p_B, 5),
        "covariance": round(cov, 6),
        "correlation": round(corr, 4),
        "are_dependent": is_dependent,
        "assessment": (
            f"DEPENDENCE DETECTED (corr={corr:.4f}, cov={cov:.6f}). Naive product independence P(A cap B) = P(A)P(B) is INVALID."
            if is_dependent
            else f"Consistent with near-independence (corr={corr:.4f}, cov={cov:.6f})."
        ),
    }


def find_graph_counterexample(
    predicate_fn: Callable[[Any], bool],
    max_n: int = 7,
) -> Dict[str, Any]:
    """Exhaustively tests a structural graph predicate against standard small graph families.
    
    predicate_fn(G) should return True if the invariant holds, False if a counterexample is found.
    Tests paths, cycles, trees, complete graphs, bipartite graphs, and random graphs on n <= max_n.

    If predicate_fn raises on some graph, that graph is NOT counted as satisfying the invariant:
    the result is reported as inconclusive ("inconclusive": True, "verified": False) with the
    first few errors listed, unless a genuine counterexample is found elsewhere.
    """
    if nx is None:
        return {"error": "NetworkX not available"}
        
    graphs_tested = 0
    predicate_errors: list = []

    def _record_error(graph_name: str, exc: BaseException) -> None:
        predicate_errors.append({"graph_family": graph_name, "error": f"{type(exc).__name__}: {exc}"[:300]})
    
    # 1. Paths, Cycles, Complete Graphs
    for n in range(2, max_n + 1):
        test_graphs = [
            (f"Path P_{n}", nx.path_graph(n)),
            (f"Cycle C_{n}", nx.cycle_graph(n)),
            (f"Complete K_{n}", nx.complete_graph(n)),
            (f"Star S_{n}", nx.star_graph(n - 1)),
        ]
        for name, G in test_graphs:
            graphs_tested += 1
            try:
                if not predicate_fn(G):
                    return {
                        "counterexample_found": True,
                        "graph_family": name,
                        "vertices": list(G.nodes()),
                        "edges": list(G.edges()),
                        "graphs_tested": graphs_tested,
                        "status": f"Falsified on {name} with V={len(G.nodes())}, E={len(G.edges())}",
                    }
            except Exception as exc:
                _record_error(name, exc)

    # 2. Complete Bipartite Graphs
    for n1 in range(1, max_n // 2 + 1):
        for n2 in range(1, max_n - n1 + 1):
            G = nx.complete_bipartite_graph(n1, n2)
            graphs_tested += 1
            try:
                if not predicate_fn(G):
                    return {
                        "counterexample_found": True,
                        "graph_family": f"K_{n1},{n2}",
                        "vertices": list(G.nodes()),
                        "edges": list(G.edges()),
                        "graphs_tested": graphs_tested,
                        "status": f"Falsified on K_{n1},{n2}",
                    }
            except Exception as exc:
                _record_error(f"K_{n1},{n2}", exc)

    # 3. Random Erdős-Rényi Graphs
    rng = random.Random(42)
    for n in range(3, max_n + 1):
        for p in [0.2, 0.4, 0.6, 0.8]:
            for _ in range(5):
                G = nx.erdos_renyi_graph(n, p, seed=rng.randint(1, 100000))
                graphs_tested += 1
                try:
                    if not predicate_fn(G):
                        return {
                            "counterexample_found": True,
                            "graph_family": f"Erdős-Rényi G({n}, {p})",
                            "vertices": list(G.nodes()),
                            "edges": list(G.edges()),
                            "graphs_tested": graphs_tested,
                            "status": f"Falsified on random graph G({n}, {p})",
                        }
                except Exception as exc:
                    _record_error(f"Erdős-Rényi G({n}, {p})", exc)

    if predicate_errors:
        first = predicate_errors[0]
        return {
            "counterexample_found": False,
            "verified": False,
            "inconclusive": True,
            "graphs_tested": graphs_tested,
            "predicate_errors": len(predicate_errors),
            "error_examples": predicate_errors[:3],
            "status": (
                f"INCONCLUSIVE: predicate raised on {len(predicate_errors)}/{graphs_tested} test graphs "
                f"(first: {first['graph_family']}: {first['error']}). Invariant NOT verified; fix the predicate."
            ),
        }

    return {
        "counterexample_found": False,
        "verified": True,
        "inconclusive": False,
        "graphs_tested": graphs_tested,
        "predicate_errors": 0,
        "status": f"Invariant holds across all {graphs_tested} test graphs up to n={max_n}.",
    }


def detect_smt_routing_or_constraint_problem(task_prompt: str) -> bool:
    """Strictly detects whether a problem statement involves discrete routing, disjoint
    paths/cycles/subcubes, parity terminal pairing, or hypercube \\ll parameter hierarchies
    where Z3 SMT verification provides high leverage.

    Scoped strictly to the Target Problem statement so unrelated problem families
    are completely untouched.
    """
    if not task_prompt:
        return False
    m = re.search(r"Target Problem[^\n:]*:\s*(.*)", task_prompt, re.S | re.I)
    target = m.group(1) if m else task_prompt

    has_disjoint_routing = bool(
        re.search(
            r"\b(?:vertex-disjoint|edge-disjoint|edge-decomposition|disjoint\s+(?:paths?|cycles?|subcubes?|matchings?|copies)"
            r"|slice-cover|slice\s+cover|bonded\s+molecule|terminal\s+vertices"
            r"|absorbing\s+(?:path|cube|structure))\b",
            target,
            re.I,
        )
    )
    has_hypercube_packing = bool(
        re.search(r"\\(?:mathcal\{Q\}|cQ)\^\{?n\}?|Q\^n|hypercube", target, re.I)
        and re.search(
            r"\b(?:hamilton\s+cycles?|perfect\s+matchings?|subcubes?|subgraph|paths?|chains?|trees?|absorber|slice|edge-decomposition)\b",
            target,
            re.I,
        )
    )
    has_ll_hierarchy = bool(
        "\\ll" in target
        and re.search(r"\\(?:mathcal\{Q\}|cQ)\^\{?n\}?|hypercube|subcube|disjoint|hamilton|matching", target, re.I)
    )
    return bool(has_disjoint_routing or has_hypercube_packing or has_ll_hierarchy)


def smt_disjoint_paths(
    graph_or_edges: Any,
    terminal_pairs: Sequence[Tuple[Any, Any]],
    required_edges: Optional[Union[Sequence[Tuple[Any, Any]], Dict[int, Sequence[Tuple[Any, Any]]]]] = None,
    forbidden_vertices: Optional[Iterable[Any]] = None,
    require_spanning: bool = False,
    vertices: Optional[Iterable[Any]] = None,
    directed: bool = False,
    timeout_ms: int = 5000,
) -> Dict[str, Any]:
    """Solves exact vertex-disjoint path routing via Z3 SMT.

    Parameters
    ----------
    graph_or_edges:
        A NetworkX Graph/DiGraph, an iterable of (u, v) edges, or an adjacency dict {u: [v, ...]}.
    terminal_pairs:
        Sequence of (s_k, t_k) start/end terminal pairs for each path k = 0..m-1.
    required_edges:
        Optional list of (u, v) edges that must be traversed by some path, or a dict
        mapping path index k -> list of (u, v) edges that path k specifically must traverse.
    forbidden_vertices:
        Optional set/list of vertices that may not be visited by any path.
    require_spanning:
        If True, every non-forbidden vertex in the graph must be visited by exactly one path.
    vertices:
        Optional explicit vertex list (useful if graph_or_edges is an edge list with isolated vertices).
    directed:
        Whether edges are directed (auto-detected if graph_or_edges is a nx.DiGraph).
    timeout_ms:
        Z3 solver timeout in milliseconds (default 5000).
    """
    if z3 is None:
        return {"error": "z3-solver not available"}

    try:
        # 1. Normalize graph into vertex list and directed edge set
        is_directed = directed
        raw_edges: List[Tuple[Any, Any]] = []
        node_set = set(vertices) if vertices is not None else set()

        if nx is not None and hasattr(graph_or_edges, "edges") and hasattr(graph_or_edges, "nodes"):
            is_directed = bool(directed or graph_or_edges.is_directed())
            node_set.update(graph_or_edges.nodes())
            raw_edges = list(graph_or_edges.edges())
        elif isinstance(graph_or_edges, dict):
            for u, nbrs in graph_or_edges.items():
                node_set.add(u)
                for v in nbrs:
                    node_set.add(v)
                    raw_edges.append((u, v))
        else:
            for u, v in graph_or_edges:
                node_set.add(u)
                node_set.add(v)
                raw_edges.append((u, v))

        for s_k, t_k in terminal_pairs:
            node_set.add(s_k)
            node_set.add(t_k)

        forbidden_set = set(forbidden_vertices) if forbidden_vertices is not None else set()
        for k_idx, (s_k, t_k) in enumerate(terminal_pairs):
            if s_k in forbidden_set or t_k in forbidden_set:
                return {
                    "status": "unsat",
                    "satisfiable": False,
                    "paths": None,
                    "reason": f"Terminal of path {k_idx} ({s_k}, {t_k}) is in forbidden_vertices.",
                }

        active_nodes = [v for v in node_set if v not in forbidden_set]
        node_to_idx = {v: i for i, v in enumerate(active_nodes)}
        n_nodes = len(active_nodes)
        m_paths = len(terminal_pairs)

        if m_paths == 0:
            return {
                "status": "sat" if not (require_spanning and n_nodes > 0) else "unsat",
                "satisfiable": not (require_spanning and n_nodes > 0),
                "paths": [],
                "total_vertices_used": 0,
                "spanning": n_nodes == 0,
            }

        # Check terminal disjointness across different paths
        seen_terminals: Dict[Any, int] = {}
        for k_idx, (s_k, t_k) in enumerate(terminal_pairs):
            for term in (s_k, t_k):
                if term in seen_terminals and seen_terminals[term] != k_idx:
                    return {
                        "status": "unsat",
                        "satisfiable": False,
                        "paths": None,
                        "reason": f"Terminal {term} is shared between path {seen_terminals[term]} and path {k_idx}.",
                    }
                seen_terminals[term] = k_idx

        dir_edges_set = set()
        for u, v in raw_edges:
            if u in forbidden_set or v in forbidden_set or u == v:
                continue
            dir_edges_set.add((node_to_idx[u], node_to_idx[v]))
            if not is_directed:
                dir_edges_set.add((node_to_idx[v], node_to_idx[u]))
        dir_edges = sorted(dir_edges_set)

        out_edges: Dict[int, List[int]] = {i: [] for i in range(n_nodes)}
        in_edges: Dict[int, List[int]] = {i: [] for i in range(n_nodes)}
        for u_i, v_i in dir_edges:
            out_edges[u_i].append(v_i)
            in_edges[v_i].append(u_i)

        solver = z3.Solver()
        solver.set("timeout", int(timeout_ms))

        # Variables:
        # used[k][i]: Bool, whether path k visits vertex i
        # flow[k, u_i, v_i]: Bool, whether path k traverses directed edge u_i -> v_i
        # pos[i]: Int in [0, n_nodes - 1], topological order along the path visiting i
        used = [[z3.Bool(f"u_{k}_{i}") for i in range(n_nodes)] for k in range(m_paths)]
        pos = [z3.Int(f"pos_{i}") for i in range(n_nodes)]
        flow: Dict[Tuple[int, int, int], Any] = {}
        for k in range(m_paths):
            for u_i, v_i in dir_edges:
                flow[(k, u_i, v_i)] = z3.Bool(f"f_{k}_{u_i}_{v_i}")

        # Vertex disjointness & potential bounds
        for i in range(n_nodes):
            solver.add(pos[i] >= 0, pos[i] < n_nodes)
            visit_count = z3.Sum([z3.If(used[k][i], 1, 0) for k in range(m_paths)])
            if require_spanning:
                solver.add(visit_count == 1)
            else:
                solver.add(visit_count <= 1)

        # Flow conservation & subtour elimination per path
        for k in range(m_paths):
            s_i = node_to_idx[terminal_pairs[k][0]]
            t_i = node_to_idx[terminal_pairs[k][1]]

            for u_i, v_i in dir_edges:
                f_var = flow[(k, u_i, v_i)]
                solver.add(z3.Implies(f_var, z3.And(used[k][u_i], used[k][v_i], pos[v_i] == pos[u_i] + 1)))
                if not is_directed and (v_i, u_i) in dir_edges_set:
                    solver.add(z3.Not(z3.And(f_var, flow[(k, v_i, u_i)])))

            if s_i == t_i:
                solver.add(used[k][s_i])
                solver.add(pos[s_i] == 0)
                for i in range(n_nodes):
                    if i != s_i:
                        solver.add(z3.Not(used[k][i]))
                    for v_i in out_edges[i]:
                        solver.add(z3.Not(flow[(k, i, v_i)]))
            else:
                solver.add(used[k][s_i], used[k][t_i])
                solver.add(pos[s_i] == 0)

                for i in range(n_nodes):
                    out_sum = z3.Sum([z3.If(flow[(k, i, v_i)], 1, 0) for v_i in out_edges[i]]) if out_edges[i] else z3.IntVal(0)
                    in_sum = z3.Sum([z3.If(flow[(k, u_i, i)], 1, 0) for u_i in in_edges[i]]) if in_edges[i] else z3.IntVal(0)

                    if i == s_i:
                        solver.add(out_sum == 1, in_sum == 0)
                    elif i == t_i:
                        solver.add(in_sum == 1, out_sum == 0)
                    else:
                        solver.add(out_sum == z3.If(used[k][i], 1, 0))
                        solver.add(in_sum == z3.If(used[k][i], 1, 0))

        # Required edges constraints
        if required_edges:
            if isinstance(required_edges, dict):
                for k_req, edge_list in required_edges.items():
                    for u, v in edge_list:
                        if u not in node_to_idx or v not in node_to_idx:
                            return {
                                "status": "unsat",
                                "satisfiable": False,
                                "paths": None,
                                "reason": f"Required edge ({u}, {v}) uses forbidden or unknown vertex.",
                            }
                        u_i, v_i = node_to_idx[u], node_to_idx[v]
                        lits = []
                        if (k_req, u_i, v_i) in flow:
                            lits.append(flow[(k_req, u_i, v_i)])
                        if not is_directed and (k_req, v_i, u_i) in flow:
                            lits.append(flow[(k_req, v_i, u_i)])
                        if not lits:
                            return {
                                "status": "unsat",
                                "satisfiable": False,
                                "paths": None,
                                "reason": f"Required edge ({u}, {v}) for path {k_req} is not in the graph.",
                            }
                        solver.add(z3.Or(lits))
            else:
                for u, v in required_edges:
                    if u not in node_to_idx or v not in node_to_idx:
                        return {
                            "status": "unsat",
                            "satisfiable": False,
                            "paths": None,
                            "reason": f"Required edge ({u}, {v}) uses forbidden or unknown vertex.",
                        }
                    u_i, v_i = node_to_idx[u], node_to_idx[v]
                    lits = []
                    for k in range(m_paths):
                        if (k, u_i, v_i) in flow:
                            lits.append(flow[(k, u_i, v_i)])
                        if not is_directed and (k, v_i, u_i) in flow:
                            lits.append(flow[(k, v_i, u_i)])
                    if not lits:
                        return {
                            "status": "unsat",
                            "satisfiable": False,
                            "paths": None,
                            "reason": f"Required edge ({u}, {v}) is not in the graph.",
                        }
                    solver.add(z3.Or(lits))

        chk = solver.check()
        if chk == z3.sat:
            model = solver.model()
            paths: List[List[Any]] = []
            total_used = 0
            for k in range(m_paths):
                nodes_in_k = []
                for i in range(n_nodes):
                    if z3.is_true(model.evaluate(used[k][i], model_completion=True)):
                        p_val = model.evaluate(pos[i], model_completion=True).as_long()
                        nodes_in_k.append((p_val, active_nodes[i]))
                nodes_in_k.sort(key=lambda pair: pair[0])
                path_k = [v for _, v in nodes_in_k]
                paths.append(path_k)
                total_used += len(path_k)
            return {
                "status": "sat",
                "satisfiable": True,
                "paths": paths,
                "total_vertices_used": total_used,
                "total_active_vertices": n_nodes,
                "spanning": total_used == n_nodes,
            }
        elif chk == z3.unsat:
            return {
                "status": "unsat",
                "satisfiable": False,
                "paths": None,
                "reason": "Z3 proved UNSAT: no vertex-disjoint path routing satisfies all constraints.",
            }
        else:
            return {
                "status": "unknown",
                "satisfiable": False,
                "paths": None,
                "reason": f"Z3 solver returned unknown (timeout={timeout_ms}ms).",
            }
    except Exception as exc:
        return {"error": f"smt_disjoint_paths failed: {exc}"}


def smt_slice_terminal_pairing(
    num_atoms: int,
    terminal_pairs: Sequence[Tuple[Tuple[int, int], Tuple[int, int]]],
    r_edge_atoms: Optional[Sequence[int]] = None,
    max_segments_per_atom: int = 25,
    forbid_two_segments_with_r_edge: bool = True,
    timeout_ms: int = 5000,
) -> Dict[str, Any]:
    """Verifies and synthesizes inter-atom bond transitions and bipartite parity balance
    across a linear chain of bipartite atoms A_0, ..., A_{t-1} (or 1-indexed A_1..A_t).

    Models slice-cover / bonded-molecule path routing in hypercube path-system constructions:
    - Each atom A_i is a bipartite graph with two parity classes {0, 1}.
    - Any local path segment covering/traversing A_i requires 1 endpoint of parity 0 and
      1 endpoint of parity 1 inside A_i (so total parity-0 endpoints == total parity-1 endpoints
      in every atom A_i).
    - Between adjacent atoms A_i and A_{i+1}, inter-atom bond edges connect parity 0 in A_i
      to parity 1 in A_{i+1}, and parity 1 in A_i to parity 0 in A_{i+1}.

    Parameters
    ----------
    num_atoms:
        Number of atoms t >= 1 in the molecule/chain.
    terminal_pairs:
        Sequence of ((atom_s, parity_s), (atom_e, parity_e)) for each global path j = 0..m-1.
        Atom indices can be 0-based (0..t-1) or 1-based (1..t); 1-based is auto-normalized if max == t.
        Parities must be in {0, 1} (or {-1, +1}, which is auto-normalized to {0, 1}).
    r_edge_atoms:
        Optional sequence of atom indices that contain a mandatory R-edge (an internal bipartite
        edge between parity 0 and parity 1 that must be traversed).
    max_segments_per_atom:
        Maximum allowed path segments inside any single atom (default 25).
    forbid_two_segments_with_r_edge:
        If True, forbids an atom containing an R-edge from having exactly 2 path segments
        when bipartite defect rules forbid k=2 with a required edge.
    timeout_ms:
        Z3 solver timeout in milliseconds.
    """
    if z3 is None:
        return {"error": "z3-solver not available"}

    try:
        if num_atoms < 1:
            return {"error": "num_atoms must be >= 1"}

        raw_atoms = [a for (a, _), (b, _) in terminal_pairs for a in (a, b)]
        if r_edge_atoms:
            raw_atoms.extend(r_edge_atoms)
        one_based = bool(raw_atoms and min(raw_atoms) >= 1 and max(raw_atoms) == num_atoms)

        def _norm_atom(a: int) -> int:
            return a - 1 if one_based else a

        def _norm_parity(p: int) -> int:
            if p in (0, 1):
                return int(p)
            if p == -1:
                return 0
            return int(p) % 2

        norm_terminals: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        for (a_s, p_s), (a_e, p_e) in terminal_pairs:
            na_s, na_e = _norm_atom(a_s), _norm_atom(a_e)
            if not (0 <= na_s < num_atoms and 0 <= na_e < num_atoms):
                return {"error": f"Atom index out of range [0, {num_atoms - 1}]: ({a_s}, {a_e})"}
            norm_terminals.append(((na_s, _norm_parity(p_s)), (na_e, _norm_parity(p_e))))

        norm_r_atoms = [_norm_atom(a) for a in (r_edge_atoms or [])]

        solver = z3.Solver()
        solver.set("timeout", int(timeout_ms))

        # Fixed terminal counts in each atom i for parity 0 and parity 1
        fixed_p0 = [0] * num_atoms
        fixed_p1 = [0] * num_atoms
        for (a_s, p_s), (a_e, p_e) in norm_terminals:
            if p_s == 0:
                fixed_p0[a_s] += 1
            else:
                fixed_p1[a_s] += 1
            if p_e == 0:
                fixed_p0[a_e] += 1
            else:
                fixed_p1[a_e] += 1

        # For each boundary i between atom i and atom i+1 (0 <= i < num_atoms - 1):
        # bond_01[i]: number of inter-atom bond edges used from (atom i, parity 0) <-> (atom i+1, parity 1)
        # bond_10[i]: number of inter-atom bond edges used from (atom i, parity 1) <-> (atom i+1, parity 0)
        bond_01 = [z3.Int(f"bond_01_{i}") for i in range(max(0, num_atoms - 1))]
        bond_10 = [z3.Int(f"bond_10_{i}") for i in range(max(0, num_atoms - 1))]

        # Net flow requirement across boundary i:
        # At least enough bond crossings to connect paths whose start and end atoms straddle boundary i,
        # and preserving global connectivity parity across the cut {0..i} vs {i+1..t-1}.
        for i in range(num_atoms - 1):
            solver.add(bond_01[i] >= 0, bond_01[i] <= max_segments_per_atom)
            solver.add(bond_10[i] >= 0, bond_10[i] <= max_segments_per_atom)

            straddling = sum(
                1
                for (a_s, _), (a_e, _) in norm_terminals
                if (a_s <= i < a_e) or (a_e <= i < a_s)
            )
            total_cross = bond_01[i] + bond_10[i]
            solver.add(total_cross >= max(1, straddling))
            solver.add(total_cross % 2 == straddling % 2)

        # In each atom i, total endpoints of parity 0 must equal total endpoints of parity 1:
        # Each segment in atom i has 1 endpoint of parity 0 and 1 endpoint of parity 1.
        segments = [z3.Int(f"segments_{i}") for i in range(num_atoms)]
        for i in range(num_atoms):
            p0_terms = [z3.IntVal(fixed_p0[i])]
            p1_terms = [z3.IntVal(fixed_p1[i])]
            if i > 0:
                # From boundary i-1: bond_01[i-1] enters atom i at parity 1; bond_10[i-1] enters atom i at parity 0
                p1_terms.append(bond_01[i - 1])
                p0_terms.append(bond_10[i - 1])
            if i < num_atoms - 1:
                # From boundary i: bond_01[i] leaves atom i at parity 0; bond_10[i] leaves atom i at parity 1
                p0_terms.append(bond_01[i])
                p1_terms.append(bond_10[i])

            tot_p0 = z3.Sum(p0_terms)
            tot_p1 = z3.Sum(p1_terms)
            solver.add(tot_p0 == tot_p1)
            solver.add(segments[i] == tot_p0)
            solver.add(segments[i] >= 1)
            solver.add(segments[i] <= max_segments_per_atom)

        for r_atom in norm_r_atoms:
            if forbid_two_segments_with_r_edge:
                solver.add(segments[r_atom] != 2)

        chk = solver.check()
        if chk == z3.sat:
            model = solver.model()
            atom_summary = []
            for i in range(num_atoms):
                seg_val = model.evaluate(segments[i], model_completion=True).as_long()
                atom_summary.append(
                    {
                        "atom": i + 1 if one_based else i,
                        "segments": seg_val,
                        "parity_0_endpoints": seg_val,
                        "parity_1_endpoints": seg_val,
                        "has_r_edge": i in norm_r_atoms,
                    }
                )
            bonds_summary = []
            for i in range(num_atoms - 1):
                b01 = model.evaluate(bond_01[i], model_completion=True).as_long()
                b10 = model.evaluate(bond_10[i], model_completion=True).as_long()
                bonds_summary.append(
                    {
                        "between_atoms": (i + 1, i + 2) if one_based else (i, i + 1),
                        "parity_0_to_1": b01,
                        "parity_1_to_0": b10,
                    }
                )
            return {
                "status": "sat",
                "satisfiable": True,
                "atoms": atom_summary,
                "bonds": bonds_summary,
            }
        elif chk == z3.unsat:
            return {
                "status": "unsat",
                "satisfiable": False,
                "atoms": None,
                "bonds": None,
                "reason": "Z3 proved UNSAT: parity balance or atom segment constraints cannot be satisfied.",
            }
        else:
            return {
                "status": "unknown",
                "satisfiable": False,
                "reason": f"Z3 returned unknown (timeout={timeout_ms}ms).",
            }
    except Exception as exc:
        return {"error": f"smt_slice_terminal_pairing failed: {exc}"}


def smt_check_parameter_hierarchy(
    variables: Union[Sequence[str], Dict[str, str]],
    constraints: Sequence[Any],
    hierarchy_order: Optional[Sequence[str]] = None,
    min_ratio: float = 2.0,
    timeout_ms: int = 5000,
) -> Dict[str, Any]:
    """Verifies feasibility of asymptotic parameter hierarchies (0 < a << b << c <= 1),
    degree budgets, and exponent inequalities using Z3.

    Parameters
    ----------
    variables:
        List of variable names (defaults to Real) or dict mapping {name: 'real' | 'int'}.
    constraints:
        Sequence of Python inequality/equation strings (e.g. "2 * k_cycles <= k",
        "exp_bound >= 10") or native Z3 BoolRef expressions.
    hierarchy_order:
        Optional sequence of variable names in ascending << order: [v_1, v_2, ..., v_m]
        enforces 0 < v_1 and v_{i+1} >= min_ratio * v_i for all i.
    min_ratio:
        Minimum separation factor for consecutive variables in hierarchy_order (default 2.0).
    timeout_ms:
        Z3 solver timeout in milliseconds.
    """
    if z3 is None:
        return {"error": "z3-solver not available"}

    try:
        solver = z3.Solver()
        solver.set("timeout", int(timeout_ms))

        z3_vars: Dict[str, Any] = {}
        if isinstance(variables, dict):
            for name, sort in variables.items():
                if str(sort).lower().startswith("int"):
                    z3_vars[name] = z3.Int(name)
                elif str(sort).lower().startswith("bool"):
                    z3_vars[name] = z3.Bool(name)
                else:
                    z3_vars[name] = z3.Real(name)
        else:
            for name in variables:
                z3_vars[name] = z3.Real(name)

        if hierarchy_order:
            for v_name in hierarchy_order:
                if v_name not in z3_vars:
                    z3_vars[v_name] = z3.Real(v_name)
            for idx, v_name in enumerate(hierarchy_order):
                solver.add(z3_vars[v_name] > 0)
                if idx + 1 < len(hierarchy_order):
                    next_v = z3_vars[hierarchy_order[idx + 1]]
                    solver.add(next_v >= z3.RealVal(min_ratio) * z3_vars[v_name])

        eval_env: Dict[str, Any] = {
            "And": z3.And,
            "Or": z3.Or,
            "Not": z3.Not,
            "Implies": z3.Implies,
            "If": z3.If,
            "Distinct": z3.Distinct,
            "Sum": z3.Sum,
            "z3": z3,
        }
        eval_env.update(z3_vars)

        for c in constraints:
            if isinstance(c, str):
                parsed = eval(c, {"__builtins__": {}}, eval_env)
                solver.add(parsed)
            else:
                solver.add(c)

        chk = solver.check()
        if chk == z3.sat:
            model = solver.model()
            witness: Dict[str, str] = {}
            for name, var in z3_vars.items():
                witness[name] = str(model.evaluate(var, model_completion=True))
            return {
                "status": "sat",
                "satisfiable": True,
                "witness": witness,
            }
        elif chk == z3.unsat:
            return {
                "status": "unsat",
                "satisfiable": False,
                "witness": None,
                "reason": "Z3 proved UNSAT: parameter hierarchy / degree / exponent constraints are mutually contradictory.",
            }
        else:
            return {
                "status": "unknown",
                "satisfiable": False,
                "witness": None,
                "reason": f"Z3 returned unknown (timeout={timeout_ms}ms).",
            }
    except Exception as exc:
        return {"error": f"smt_check_parameter_hierarchy failed: {exc}"}


def detect_topological_or_structural_graph_problem(task_prompt: str) -> bool:
    """Strictly detects whether the Target Problem involves planar/outerplanar embeddings,
    dual graphs, cut-vertices/biconnected blocks, cycle chords, or induced holes/paths.
    """
    if not task_prompt:
        return False
    m = re.search(r"Target Problem[^\n:]*:\s*(.*)", task_prompt, re.S | re.I)
    target = m.group(1) if m else task_prompt
    return bool(
        re.search(
            r"\b(?:planar|outerplanar|embedding|outer\s+face|dual\s+graph|genus|crossing\s+number"
            r"|biconnected|cut-vertex|cut\s+vertex|articulation\s+point|induced\s+path|induced\s+cycle"
            r"|hole-free|chordless|chords?)\b|C_\{\\geq",
            target,
            re.I,
        )
    )


def planar_dual_and_faces(G: Any) -> Dict[str, Any]:
    """Computes the combinatorial planar embedding, faces, exact planar dual multigraph G*,
    articulation points (cut-vertices), and biconnected components of a NetworkX graph G.
    """
    if nx is None:
        return {"error": "NetworkX not available"}
    try:
        UG = G.to_undirected() if G.is_directed() else G
        is_planar, emb_or_kuratowski = nx.check_planarity(UG, counterexample=True)
        cut_vertices = list(nx.articulation_points(UG)) if len(UG) > 0 else []
        blocks = [set(c) for c in nx.biconnected_components(UG)] if len(UG) > 0 else []

        if not is_planar:
            return {
                "is_planar": False,
                "kuratowski_edges": list(emb_or_kuratowski.edges()),
                "articulation_points": cut_vertices,
                "biconnected_components": blocks,
            }

        emb = emb_or_kuratowski
        visited_half_edges = set()
        faces: List[List[Any]] = []
        half_edge_to_face: Dict[Tuple[Any, Any], int] = {}

        for u in emb.nodes():
            for v in emb.neighbors_cw_order(u):
                if (u, v) not in visited_half_edges:
                    face_nodes = list(emb.traverse_face(u, v))
                    face_idx = len(faces)
                    faces.append(face_nodes)
                    k = len(face_nodes)
                    for i in range(k):
                        he = (face_nodes[i], face_nodes[(i + 1) % k])
                        visited_half_edges.add(he)
                        half_edge_to_face[he] = face_idx

        dual_G = nx.MultiGraph()
        for f_idx, f_nodes in enumerate(faces):
            dual_G.add_node(f_idx, boundary_vertices=f_nodes, length=len(f_nodes))

        for u, v in UG.edges():
            f1 = half_edge_to_face.get((u, v))
            f2 = half_edge_to_face.get((v, u))
            if f1 is not None and f2 is not None:
                dual_G.add_edge(f1, f2, primal_edge=(u, v))

        dual_simple = nx.Graph(dual_G)
        dual_cut_vertices = list(nx.articulation_points(dual_simple)) if len(dual_simple) > 0 else []

        return {
            "is_planar": True,
            "num_faces": len(faces),
            "faces": faces,
            "dual_graph": dual_G,
            "articulation_points": cut_vertices,
            "dual_articulation_points": dual_cut_vertices,
            "biconnected_components": blocks,
        }
    except Exception as exc:
        return {"error": f"planar_dual_and_faces failed: {exc}"}


def check_cycle_chords_and_holes(
    G: Any = None,
    cycle_vertices: Optional[Sequence[Any]] = None,
    chords: Optional[Sequence[Tuple[Any, Any]]] = None,
    min_hole_length: int = 4,
) -> Dict[str, Any]:
    """Generic analyzer for induced holes (chordless cycles C_{>=t}) and pairwise topological
    relations of chords ((L_1, R_1), (L_2, R_2)) along a cycle or path ordering.

    Classifies chord pairs into:
    - 'crossing': L_1 < L_2 < R_1 < R_2 or L_2 < L_1 < R_2 < R_1 (interleaving endpoints; note both L and R are strictly increasing!)
    - 'nested': L_1 < L_2 < R_2 < R_1 or L_2 < L_1 < R_1 < R_2 (laminar / one nested inside the other)
    - 'disjoint': R_1 < L_2 or R_2 < L_1 (non-overlapping intervals along the ordering)
    - 'sharing_endpoint': {L_1, R_1} & {L_2, R_2} != empty
    """
    try:
        result: Dict[str, Any] = {}
        pos_map: Dict[Any, int] = {}
        if cycle_vertices is not None:
            pos_map = {v: idx for idx, v in enumerate(cycle_vertices)}

        extracted_chords: List[Tuple[int, int, Tuple[Any, Any]]] = []
        if chords is not None:
            for u, v in chords:
                pu = pos_map[u] if u in pos_map else int(u)
                pv = pos_map[v] if v in pos_map else int(v)
                L, R = (pu, pv) if pu <= pv else (pv, pu)
                extracted_chords.append((L, R, (u, v)))
        elif G is not None and cycle_vertices is not None:
            k = len(cycle_vertices)
            cyc_set = set(cycle_vertices)
            for u, v in G.edges():
                if u in cyc_set and v in cyc_set:
                    pu, pv = pos_map[u], pos_map[v]
                    diff = abs(pu - pv)
                    if diff > 1 and diff < k - 1:
                        L, R = (pu, pv) if pu <= pv else (pv, pu)
                        extracted_chords.append((L, R, (u, v)))

        if extracted_chords:
            crossing_pairs = []
            nested_pairs = []
            disjoint_pairs = []
            shared_endpoint_pairs = []
            m = len(extracted_chords)
            for i in range(m):
                L1, R1, c1 = extracted_chords[i]
                for j in range(i + 1, m):
                    L2, R2, c2 = extracted_chords[j]
                    if L1 in (L2, R2) or R1 in (L2, R2):
                        shared_endpoint_pairs.append((c1, c2))
                    elif (L1 < L2 < R1 < R2) or (L2 < L1 < R2 < R1):
                        crossing_pairs.append((c1, c2))
                    elif (L1 < L2 and R2 < R1) or (L2 < L1 and R1 < R2):
                        nested_pairs.append((c1, c2))
                    else:
                        disjoint_pairs.append((c1, c2))
            result["num_chords"] = m
            result["crossing_pairs"] = crossing_pairs
            result["nested_pairs"] = nested_pairs
            result["disjoint_pairs"] = disjoint_pairs
            result["shared_endpoint_pairs"] = shared_endpoint_pairs

        if G is not None and nx is not None:
            holes = [c for c in nx.chordless_cycles(G) if len(c) >= min_hole_length]
            result["holes"] = holes
            result["num_holes"] = len(holes)
            result["is_hole_free"] = len(holes) == 0

        return result
    except Exception as exc:
        return {"error": f"check_cycle_chords_and_holes failed: {exc}"}


def detect_quantum_or_spectral_operator_problem(task_prompt: str) -> bool:
    """Strictly detects whether the Target Problem involves quantum states/channels,
    density matrices, Uhlmann fidelity, partial trace, or PSD / Hermitian operator inequalities.
    """
    if not task_prompt:
        return False
    m = re.search(r"Target Problem[^\n:]*:\s*(.*)", task_prompt, re.S | re.I)
    target = m.group(1) if m else task_prompt
    return bool(
        re.search(
            r"\\succeq|\\preceq|\b(?:quantum|qubit|unitary|hermitian|density\s+matrix|cptp|kraus"
            r"|stinespring|uhlmann|fidelity|partial\s+trace|positive\s+semidefinite|psd"
            r"|eigenvalues?|singular\s+values?|operator\s+norm|schatten|kronecker)\b|\\otimes",
            target,
            re.I,
        )
    )


def random_density_matrix(dim: int = 2, rank: Optional[int] = None, rng: Any = None) -> Any:
    """Generates a random complex Hermitian Positive Semidefinite density matrix rho (Tr(rho) = 1, rho >= 0)."""
    if np is None:
        raise RuntimeError("NumPy not available")
    gen = rng if rng is not None else np.random.default_rng()
    k = rank if rank is not None else dim
    G = gen.standard_normal((dim, k)) + 1j * gen.standard_normal((dim, k))
    W = G @ G.conj().T
    return W / np.trace(W).real


def random_unitary(dim: int = 2, rng: Any = None) -> Any:
    """Generates a Haar-random d x d unitary matrix U (U U^dagger = I)."""
    if np is None:
        raise RuntimeError("NumPy not available")
    gen = rng if rng is not None else np.random.default_rng()
    Z = (gen.standard_normal((dim, dim)) + 1j * gen.standard_normal((dim, dim))) / math.sqrt(2.0)
    Q, R = np.linalg.qr(Z)
    d = np.diag(R)
    ph = np.where(np.abs(d) > 1e-12, d / np.abs(d), 1.0 + 0j)
    return Q * ph


def partial_trace(rho: Any, dims: Tuple[int, int] = (2, 2), keep: int = 0) -> Any:
    """Computes the partial trace of a bipartite operator rho on C^{d_A} x C^{d_B}.
    keep=0 returns Tr_B(rho) (on subsystem A); keep=1 returns Tr_A(rho) (on subsystem B).
    """
    if np is None:
        raise RuntimeError("NumPy not available")
    dA, dB = dims
    arr = np.asarray(rho, dtype=complex).reshape(dA, dB, dA, dB)
    if keep == 0:
        return np.einsum("ijik->jk", arr)
    elif keep == 1:
        return np.einsum("ijil->jl", np.swapaxes(np.swapaxes(arr, 0, 1), 2, 3))
    raise ValueError("keep must be 0 (keep subsystem A) or 1 (keep subsystem B)")


def uhlmann_fidelity(rho: Any, sigma: Any) -> float:
    """Computes Uhlmann fidelity F(rho, sigma) = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2 in [0, 1]."""
    if np is None:
        raise RuntimeError("NumPy not available")
    rho_m = np.asarray(rho, dtype=complex)
    sigma_m = np.asarray(sigma, dtype=complex)
    vals, vecs = np.linalg.eigh((rho_m + rho_m.conj().T) / 2.0)
    sqrt_vals = np.sqrt(np.maximum(vals.real, 0.0))
    sqrt_rho = (vecs * sqrt_vals) @ vecs.conj().T
    inner = sqrt_rho @ sigma_m @ sqrt_rho
    inner_vals = np.linalg.eigvalsh((inner + inner.conj().T) / 2.0)
    tr_sqrt = float(np.sum(np.sqrt(np.maximum(inner_vals.real, 0.0))))
    return tr_sqrt * tr_sqrt


def check_psd_inequality(
    matrix_diff_fn: Callable[..., Any],
    dim: int = 4,
    n_trials: int = 200,
    tol: float = 1e-8,
    seed: int = 42,
) -> Dict[str, Any]:
    """Empirically tests a Loewner / Positive Semidefinite operator inequality A >= B
    by checking whether M = A - B has all eigenvalues >= -tol across random density matrices
    and Haar unitaries.

    matrix_diff_fn can accept:
    - 0 args: lambda: A - B
    - 1 arg:  lambda rho: f(rho)
    - 2 args: lambda rho, sigma: f(rho, sigma)
    - 3 args: lambda rho, sigma, U: f(rho, sigma, U)
    """
    if np is None:
        return {"error": "NumPy not available"}
    import inspect

    try:
        rng = np.random.default_rng(seed)
        sig = inspect.signature(matrix_diff_fn)
        n_params = len(
            [
                p
                for p in sig.parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty
            ]
        )

        min_eig_seen = float("inf")
        for trial in range(n_trials):
            if n_params == 0:
                M = matrix_diff_fn()
            elif n_params == 1:
                M = matrix_diff_fn(random_density_matrix(dim, rng=rng))
            elif n_params == 2:
                M = matrix_diff_fn(random_density_matrix(dim, rng=rng), random_density_matrix(dim, rng=rng))
            else:
                args = [random_density_matrix(dim, rng=rng) for _ in range(n_params - 1)] + [random_unitary(dim, rng=rng)]
                M = matrix_diff_fn(*args)

            M_arr = np.asarray(M, dtype=complex)
            H = (M_arr + M_arr.conj().T) / 2.0
            eigs = np.linalg.eigvalsh(H)
            min_ev = float(np.min(eigs.real))
            if min_ev < min_eig_seen:
                min_eig_seen = min_ev
            if min_ev < -tol:
                return {
                    "holds": False,
                    "counterexample_found": True,
                    "trial": trial,
                    "min_eigenvalue": min_ev,
                    "eigenvalues": [float(x) for x in eigs.real],
                    "status": f"Falsified on trial {trial}: min eigenvalue = {min_ev:.6e} < 0 (not PSD).",
                }
        return {
            "holds": True,
            "counterexample_found": False,
            "trials": n_trials,
            "min_eigenvalue": min_eig_seen,
            "status": f"PSD inequality holds across all {n_trials} trials (min eigenvalue = {min_eig_seen:.6e}).",
        }
    except Exception as exc:
        return {"error": f"check_psd_inequality failed: {exc}"}


def check_quantum_channel_claim(
    predicate_fn: Callable[..., bool],
    dim: int = 2,
    n_random: int = 40,
    seed: int = 42,
) -> Dict[str, Any]:
    """Tests a mathematical predicate about quantum CPTP channels E(rho) = sum_k K_k rho K_k^dagger
    across canonical families (unitary, depolarizing, dephasing/phase-damping, amplitude-damping,
    constant replacement, and random Stinespring/Kraus channels).

    predicate_fn can accept:
    - 1 arg:  predicate_fn(apply_fn) where apply_fn(rho) -> E(rho)
    - 2 args: predicate_fn(apply_fn, kraus_ops)
    - 3 args: predicate_fn(apply_fn, kraus_ops, channel_name)
    """
    if np is None:
        return {"error": "NumPy not available"}
    import inspect

    try:
        rng = np.random.default_rng(seed)
        sig = inspect.signature(predicate_fn)
        n_params = len(
            [
                p
                for p in sig.parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty
            ]
        )

        channels: List[Tuple[str, List[Any]]] = []
        I_d = np.eye(dim, dtype=complex)
        channels.append(("Identity", [I_d]))

        U = random_unitary(dim, rng=rng)
        channels.append(("Random Unitary", [U]))

        if dim == 2:
            X = np.array([[0, 1], [1, 0]], dtype=complex)
            Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
            Z = np.array([[1, 0], [0, -1]], dtype=complex)
            for p in (0.1, 0.3, 0.5):
                # Phase-damping / dephasing channel: (1-p) I rho I + p Z rho Z
                channels.append((f"Dephasing(p={p})", [math.sqrt(1 - p) * I_d, math.sqrt(p) * Z]))
                # Depolarizing channel
                channels.append(
                    (
                        f"Depolarizing(p={p})",
                        [
                            math.sqrt(1 - 3 * p / 4) * I_d,
                            math.sqrt(p / 4) * X,
                            math.sqrt(p / 4) * Y,
                            math.sqrt(p / 4) * Z,
                        ],
                    )
                )
                # Amplitude-damping channel
                K0 = np.array([[1, 0], [0, math.sqrt(1 - p)]], dtype=complex)
                K1 = np.array([[0, math.sqrt(p)], [0, 0]], dtype=complex)
                channels.append((f"AmplitudeDamping(gamma={p})", [K0, K1]))

        # Constant replacement channel E(rho) = |0><0| Tr(rho)
        rep_ops = []
        for j in range(dim):
            K_j = np.zeros((dim, dim), dtype=complex)
            K_j[0, j] = 1.0
            rep_ops.append(K_j)
        channels.append(("Replacement(|0><0|)", rep_ops))

        # Random Kraus channels via Haar unitary on C^{dim * r} (Stinespring dilation)
        for idx in range(n_random):
            r_kraus = int(rng.integers(2, dim * dim + 1))
            U_big = random_unitary(dim * r_kraus, rng=rng)
            isometry = U_big[:, :dim]  # (dim * r_kraus) x dim with V^dagger V = I_dim
            k_ops = [isometry[k * dim : (k + 1) * dim, :] for k in range(r_kraus)]
            channels.append((f"RandomCPTP_{idx}(rank={r_kraus})", k_ops))

        for name, k_ops in channels:
            def _apply(rho: Any, _ops: List[Any] = k_ops) -> Any:
                r_m = np.asarray(rho, dtype=complex)
                return sum(K @ r_m @ K.conj().T for K in _ops)

            if n_params <= 1:
                ok = bool(predicate_fn(_apply))
            elif n_params == 2:
                ok = bool(predicate_fn(_apply, k_ops))
            else:
                ok = bool(predicate_fn(_apply, k_ops, name))

            if not ok:
                return {
                    "holds": False,
                    "counterexample_found": True,
                    "channel_family": name,
                    "kraus_rank": len(k_ops),
                    "status": f"Falsified on quantum channel '{name}' (Kraus rank={len(k_ops)}).",
                }

        return {
            "holds": True,
            "counterexample_found": False,
            "channels_tested": len(channels),
            "status": f"Claim holds across all {len(channels)} canonical and random CPTP quantum channels.",
        }
    except Exception as exc:
        return {"error": f"check_quantum_channel_claim failed: {exc}"}


