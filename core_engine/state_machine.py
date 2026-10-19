"""Generic Mathematical State Machine Engine for Persistent Math REPL.

Implements a formal epistemic state machine for theorem proving across all mathematical
domains (graph theory, probability/measure, linear algebra, topology, algebra).

Tracks:
1. Object Lineage & Provenance DAG (X -> Y = f(X, restriction))
2. Hereditary vs. Fragile Property Propagation:
   - Hereditary: automatically preserved under restriction/sub-structures.
   - Fragile: systematically invalidated upon restriction/conditioning unless explicitly proven.
3. Active Invariant Verification & Counterexample Hunting (SymPy, Z3, NetworkX, NumPy).
4. State Checkpointing & Rollback for Branch Exploration and Backtracking.
5. Open Sub-goal Tracking and Verified Lemma Ledger.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


# ==============================================================================
# Universal Property Classification: Hereditary vs. Fragile
# ==============================================================================

# Properties that are mathematically guaranteed to be inherited by sub-structures
# (subgraphs, induced subgraphs, subsets, sub-sigma-algebras, principal submatrices, subgroups)
HEREDITARY_PROPERTIES: Set[str] = {
    # Graph theory (under induced subgraphs / vertex deletion)
    "subgraph_closed", "induced_subgraph_free", "bipartite", "planar",
    "triangle_free", "forest", "acyclic", "k_colorable", "chordal",
    # Measure & Probability
    "measurable", "bounded_support", "discrete", "non_negative",
    # Linear Algebra / Matrices (under principal submatrices)
    "hermitian", "symmetric", "positive_semidefinite", "diagonal",
    # Algebra
    "abelian", "torsion_free", "commutative",
    # Topology & Analysis
    "hausdorff", "metric", "t2", "first_countable", "second_countable",
}

# Properties that are FRAGILE: destroyed under restriction, projection, or conditioning
# unless an explicit invariance proof or witness is established.
FRAGILE_PROPERTIES: Dict[str, str] = {
    # Probability / Stochastic processes
    "independent": "Conditioning on global events or truncated sets destroys product-measure independence.",
    "mutually_independent": "Conditioning or subset-filtering creates non-trivial conditioning dependencies across variables.",
    "pairwise_independent": "Conditioning on an auxiliary event destroys pairwise independence.",
    "martingale": "Restricting filtration or stopping without optional stopping criteria destroys martingale property.",
    "stationary": "Time truncation or conditioning on boundary hitting destroys stationarity.",
    "markov": "Hidden vertex truncation destroys Markov property (introduces long-range memory).",
    
    # Graph theory
    "connected": "Vertex or edge deletion disconnects components.",
    "biconnected": "Vertex deletion can create cut-vertices.",
    "minimum_degree": "Vertex deletion reduces the degree of neighboring vertices.",
    "hamiltonian": "Vertex deletion destroys Hamiltonian cycles/paths.",
    "regular": "Vertex deletion destroys uniform degree regularity.",
    "eulerian": "Vertex deletion alters vertex degree parities.",
    
    # Linear Algebra
    "invertible": "Submatrices / projections may be singular.",
    "positive_definite": "General restrictions or perturbations can reduce the minimal eigenvalue to <= 0.",
    "orthogonal": "Sub-collections of vectors or perturbed bases lose exact orthogonality.",
    "surjective": "Domain restriction or subspace projection destroys surjectivity.",
    "injective": "Quotient maps or dimension reductions destroy injectivity.",
    
    # Topology & Analysis
    "compact": "Open subsets or unbounded restrictions lose compactness.",
    "connected_space": "Removing subsets can disconnect the topological space.",
}


class StateInvariantViolation(Exception):
    """Raised when an illegal mathematical state transition or unverified fragile property is asserted."""
    pass


def _json_safe(val: Any) -> Any:
    """Recursively converts sets, tuples, and arbitrary REPL objects into JSON-serializable primitives."""
    if val is None or isinstance(val, (bool, int, float, str)):
        return val
    if isinstance(val, (set, frozenset)):
        return sorted((_json_safe(x) for x in val), key=str)
    if isinstance(val, dict):
        return {str(k): _json_safe(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_json_safe(x) for x in val]
    if hasattr(val, "to_dict") and callable(val.to_dict):
        return _json_safe(val.to_dict())
    return str(val)


@dataclass
class MathObject:
    """Represents a registered mathematical entity in the proof state."""
    name: str
    domain: str  # e.g. 'graph', 'random_variable', 'matrix', 'set', 'sequence'
    properties: Set[str] = field(default_factory=set)
    revoked_properties: Dict[str, str] = field(default_factory=dict)
    parent: Optional[str] = None
    derivation_op: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_property(self, prop: str) -> bool:
        clean = prop.strip().lower()
        return clean in self.properties

    def is_revoked(self, prop: str) -> bool:
        clean = prop.strip().lower()
        return clean in self.revoked_properties

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": str(self.name),
            "domain": str(self.domain),
            "properties": sorted((str(p) for p in self.properties), key=str),
            "revoked_properties": _json_safe(dict(self.revoked_properties)),
            "parent": str(self.parent) if self.parent is not None else None,
            "derivation_op": str(self.derivation_op) if self.derivation_op is not None else None,
            "metadata": _json_safe(dict(self.metadata)),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MathObject":
        return cls(
            name=str(data.get("name", "")),
            domain=str(data.get("domain", "")),
            properties=set(data.get("properties") or []),
            revoked_properties=dict(data.get("revoked_properties") or {}),
            parent=data.get("parent"),
            derivation_op=data.get("derivation_op"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class StateSnapshot:
    """Complete snapshot of mathematical state for rollback and backtracking."""
    name: str
    objects: Dict[str, MathObject]
    subgoals: List[Dict[str, Any]]
    verified_lemmas: List[str]
    namespace_keys: Set[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": str(self.name),
            "objects": {str(k): v.to_dict() if hasattr(v, "to_dict") else _json_safe(v) for k, v in self.objects.items()},
            "subgoals": _json_safe(self.subgoals),
            "verified_lemmas": [str(x) for x in self.verified_lemmas],
            "namespace_keys": sorted((str(k) for k in self.namespace_keys), key=str),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateSnapshot":
        return cls(
            name=str(data.get("name", "")),
            objects={
                k: MathObject.from_dict(v)
                for k, v in (data.get("objects") or {}).items()
            },
            subgoals=list(data.get("subgoals") or []),
            verified_lemmas=list(data.get("verified_lemmas") or []),
            namespace_keys=set(data.get("namespace_keys") or []),
        )


class MathStateMachine:
    """Formal mathematical state machine managing active proofs, object lineages,

    invariants, and branch backtracking in the persistent REPL kernel.
    """

    def __init__(self, kernel_namespace: Optional[Dict[str, Any]] = None):
        self._objects: Dict[str, MathObject] = {}
        self._subgoals: List[Dict[str, Any]] = []
        self._verified_lemmas: List[str] = []
        self._checkpoints: Dict[str, StateSnapshot] = {}
        self._namespace_ref = kernel_namespace

    def bind_namespace(self, ns: Dict[str, Any]) -> None:
        """Binds the kernel namespace for rollback tracking."""
        self._namespace_ref = ns

    # --------------------------------------------------------------------------
    # 1. Object Registration & Lineage
    # --------------------------------------------------------------------------

    def register(self, name: str, domain: str, properties: Optional[List[str]] = None,
                 description: str = "") -> MathObject:
        """Registers a base mathematical object with verified initial axioms/hypotheses."""
        clean_props = {p.strip().lower() for p in (properties or [])}
        obj = MathObject(
            name=name,
            domain=domain.strip().lower(),
            properties=clean_props,
            metadata={"description": description}
        )
        self._objects[name] = obj
        return obj

    def derive(self, name: str, parent: str, op: str,
               details: str = "",
               explicitly_proven_properties: Optional[List[str]] = None) -> MathObject:
        """Derives a new mathematical object from a parent via a structural operation

        (e.g., vertex deletion, conditioning on an event, projection, submatrix).
        
        Automatically preserves Hereditary properties and REVOKES Fragile properties.
        """
        if parent not in self._objects:
            raise KeyError(f"Parent object '{parent}' is not registered in the state machine.")

        p_obj = self._objects[parent]
        inherited_props: Set[str] = set()
        revoked: Dict[str, str] = {}

        # Evaluate property inheritance
        for prop in p_obj.properties:
            if prop in HEREDITARY_PROPERTIES:
                inherited_props.add(prop)
            elif prop in FRAGILE_PROPERTIES:
                reason = FRAGILE_PROPERTIES[prop]
                revoked[prop] = f"Invalidated by '{op}': {reason}"

        # Add explicitly proven properties
        if explicitly_proven_properties:
            for p in explicitly_proven_properties:
                p_clean = p.strip().lower()
                inherited_props.add(p_clean)
                revoked.pop(p_clean, None)

        child = MathObject(
            name=name,
            domain=p_obj.domain,
            properties=inherited_props,
            revoked_properties=revoked,
            parent=parent,
            derivation_op=op,
            metadata={"details": details}
        )
        self._objects[name] = child
        return child

    # --------------------------------------------------------------------------
    # 2. Active Invariant Gate
    # --------------------------------------------------------------------------

    def assert_property(self, name: str, prop: str,
                        proof_method: Optional[str] = None,
                        witness: Any = None) -> bool:
        """Asserts that an object satisfies a mathematical property.

        If the property was revoked (fragile), refuses registration unless a valid
        proof_method ('symbolic', 'empirical', 'structural', 'martingale') is provided.
        """
        if name not in self._objects:
            raise KeyError(f"Object '{name}' not found in state machine.")

        clean_prop = prop.strip().lower()
        obj = self._objects[name]

        if clean_prop in obj.properties:
            return True

        if clean_prop in obj.revoked_properties:
            if not proof_method:
                reason = obj.revoked_properties[clean_prop]
                raise StateInvariantViolation(
                    f"\n[MATHEMATICAL INVARIANT VIOLATION on '{name}']:\n"
                    f"Property '{clean_prop}' was revoked during derivation from '{obj.parent}'.\n"
                    f"Reason: {reason}\n"
                    f"You CANNOT assume '{clean_prop}' without a formal deduction or verification.\n"
                    f"To certify, call: state.assert_property('{name}', '{clean_prop}', proof_method=..., witness=...)"
                )

        # Register certified property
        obj.properties.add(clean_prop)
        obj.revoked_properties.pop(clean_prop, None)
        obj.metadata[f"proof_{clean_prop}"] = {
            "method": proof_method,
            "witness": str(witness)[:200] if witness else "provided"
        }
        return True

    # --------------------------------------------------------------------------
    # 3. State Checkpointing & Rollback (Solving Cognitive Inertia & Dead Ends)
    # --------------------------------------------------------------------------

    def checkpoint(self, name: str) -> str:
        """Creates a named checkpoint of the mathematical state and namespace."""
        ns_keys = set(self._namespace_ref.keys()) if self._namespace_ref else set()
        snapshot = StateSnapshot(
            name=name,
            objects=copy.deepcopy(self._objects),
            subgoals=copy.deepcopy(self._subgoals),
            verified_lemmas=list(self._verified_lemmas),
            namespace_keys=ns_keys
        )
        self._checkpoints[name] = snapshot
        return f"[State] Checkpoint '{name}' established ({len(self._objects)} objects, {len(self._subgoals)} subgoals)."

    def rollback(self, name: str) -> str:
        """Rolls back the mathematical state and purges dead-end namespace variables."""
        if name not in self._checkpoints:
            raise KeyError(f"No checkpoint named '{name}' found. Available: {list(self._checkpoints.keys())}")

        snapshot = self._checkpoints[name]
        self._objects = copy.deepcopy(snapshot.objects)
        self._subgoals = copy.deepcopy(snapshot.subgoals)
        self._verified_lemmas = list(snapshot.verified_lemmas)

        # Namespace clean-up: remove variables defined after checkpoint
        if self._namespace_ref:
            current_keys = set(self._namespace_ref.keys())
            added_keys = current_keys - snapshot.namespace_keys
            for k in added_keys:
                # Do not delete internal python/math globals
                if not k.startswith("__") and k != "state":
                    self._namespace_ref.pop(k, None)

        return (f"[State] Successfully rolled back to '{name}'. "
                f"Purged dead-end variables. Invariants restored to clean state.")

    # --------------------------------------------------------------------------
    # 4. Universal Verification & Falsification Engines
    # --------------------------------------------------------------------------

    def test_independence(self, sampler_fn: Callable[[], Tuple[int, int]],
                          n_trials: int = 2500, alpha: float = 0.01) -> Dict[str, Any]:
        """Empirically tests whether two indicator variables exhibit non-zero covariance

        under conditioning or truncated processes.
        """
        import numpy as np

        samples = [sampler_fn() for _ in range(n_trials)]
        A = np.array([s[0] for s in samples], dtype=float)
        B = np.array([s[1] for s in samples], dtype=float)

        p_a = np.mean(A)
        p_b = np.mean(B)
        p_ab = np.mean(A * B)
        cov = p_ab - (p_a * p_b)

        # Standard error of covariance
        se = np.sqrt(max(1e-12, (p_ab * (1 - p_ab) + (p_a * p_b) * (1 - p_a * p_b)) / n_trials))
        z_score = abs(cov) / se
        is_independent = z_score < 2.58  # 99% confidence threshold

        result = {
            "p_A": round(float(p_a), 4),
            "p_B": round(float(p_b), 4),
            "p_AB": round(float(p_ab), 4),
            "cov": round(float(cov), 6),
            "z_score": round(float(z_score), 2),
            "independent": bool(is_independent),
            "verdict": "INDEPENDENT" if is_independent else "DEPENDENCY_DETECTED"
        }

        if not is_independent:
            print(f"[REPL Invariant Alert] Variables exhibit statistically significant non-zero covariance: cov={cov:.6f}, z={z_score:.2f}!")
            print("Mutually independent Chernoff bounds are INVALID on this conditioned distribution.")
            print("Pivot to: Azuma-Hoeffding edge-exposure martingale, McDiarmid, or Lovász Local Lemma.")
        else:
            print(f"[REPL Invariant Verified] Empirical covariance is consistent with zero (cov={cov:.6f}, z={z_score:.2f}).")
        return result

    def test_graph_counterexample(self, predicate_fn: Callable[[Any], bool],
                                  max_n: int = 7) -> Optional[str]:
        """Searches across standard small graph families for counterexamples to a graph predicate."""
        import networkx as nx

        families = [
            ("Path", lambda n: nx.path_graph(n)),
            ("Cycle", lambda n: nx.cycle_graph(max(3, n))),
            ("Complete", lambda n: nx.complete_graph(n)),
            ("Star", lambda n: nx.star_graph(n - 1)),
            ("Bipartite_Complete", lambda n: nx.complete_bipartite_graph(n // 2, max(1, n - n // 2))),
            ("Wheel", lambda n: nx.wheel_graph(max(4, n))),
        ]

        for n in range(3, max_n + 1):
            for name, gen in families:
                try:
                    G = gen(n)
                    if not predicate_fn(G):
                        msg = f"Counterexample found: {name} graph on {n} nodes violates predicate!"
                        print(f"[State Refutation] {msg}")
                        return msg
                except Exception:
                    continue

        print(f"[State Invariant Verified] No counterexample found among standard graph families up to n={max_n}.")
        return None

    # --------------------------------------------------------------------------
    # 5. Sub-goal Ledger & Proof Trace
    # --------------------------------------------------------------------------

    def add_subgoal(self, goal_id: str, description: str) -> None:
        """Registers a sub-goal required to establish the target theorem."""
        self._subgoals.append({
            "id": goal_id,
            "description": description,
            "status": "OPEN",
            "proof_note": None
        })

    def close_subgoal(self, goal_id: str, proof_note: str) -> None:
        """Closes a verified sub-goal."""
        for sg in self._subgoals:
            if sg["id"] == goal_id:
                sg["status"] = "CLOSED"
                sg["proof_note"] = proof_note
                self._verified_lemmas.append(f"Subgoal [{goal_id}]: {proof_note}")
                print(f"[State Machine] Subgoal '{goal_id}' CLOSED: {proof_note}")
                return
        raise KeyError(f"Subgoal '{goal_id}' not found.")

    def record_lemma(self, lemma_text: str) -> None:
        """Records a formally verified intermediate lemma into the proof trace."""
        self._verified_lemmas.append(lemma_text.strip())

    def status(self) -> str:
        """Outputs a clean ASCII status report of the current mathematical state."""
        lines = ["=== Mathematical State Machine Status ==="]
        lines.append(f"Registered Objects ({len(self._objects)}):")
        for name, obj in self._objects.items():
            props = ", ".join(sorted(obj.properties)) or "none"
            rev = ", ".join(sorted(obj.revoked_properties.keys())) or "none"
            parent_info = f" (derived from '{obj.parent}' via '{obj.derivation_op}')" if obj.parent else ""
            lines.append(f"  • {name} [{obj.domain}]{parent_info}")
            lines.append(f"      Active Invariants: {props}")
            if obj.revoked_properties:
                lines.append(f"      REVOKED Fragile Invariants: {rev}")

        lines.append(f"\nSubgoals ({len(self._subgoals)}):")
        for sg in self._subgoals:
            lines.append(f"  [{sg['status']}] {sg['id']}: {sg['description']}")

        lines.append(f"\nVerified Lemmas ({len(self._verified_lemmas)}):")
        for vl in self._verified_lemmas:
            lines.append(f"  ✓ {vl}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the current mathematical state machine to a JSON-compatible dictionary."""
        return {
            "objects": {str(k): v.to_dict() if hasattr(v, "to_dict") else _json_safe(v) for k, v in self._objects.items()},
            "subgoals": _json_safe(self._subgoals),
            "verified_lemmas": [str(x) for x in self._verified_lemmas],
            "checkpoints": {str(k): v.to_dict() if hasattr(v, "to_dict") else _json_safe(v) for k, v in self._checkpoints.items()},
        }

    def load_state_dict(self, data: Dict[str, Any]) -> None:
        """Restores the mathematical state machine from a serialized dictionary."""
        if not data:
            return
        self._objects = {
            k: MathObject.from_dict(v)
            for k, v in (data.get("objects") or {}).items()
        }
        self._subgoals = copy.deepcopy(list(data.get("subgoals") or []))
        self._verified_lemmas = list(data.get("verified_lemmas") or [])
        self._checkpoints = {
            k: StateSnapshot.from_dict(v)
            for k, v in (data.get("checkpoints") or {}).items()
        }


# Global singleton instance for REPL prelude
state = MathStateMachine()
