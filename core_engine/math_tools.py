"""Generic Mathematical Verification & Empirical Diagnostics Toolkit.

Provides domain-agnostic, rigorous computational verification functions preloaded into
the Python REPL to stress-test mathematical assertions, asymptotic bounds, probabilistic
dependencies, and combinatorial invariants without problem-specific heuristics.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, Tuple, Union

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
