"""Persistent Mathematical REPL Kernel for Prime Math.

Maintains a long-lived execution environment with preloaded symbolic and numerical
libraries (SymPy, Z3, mpmath, NumPy, SciPy). State, variable bindings, and lemmas
persist across turns within an isolated mathematical sandbox.
"""

from __future__ import annotations

import ast
import ctypes
import functools
import io
import itertools
import os
import pydoc
import socket as _socket_mod
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Set, Tuple

# Ensure pydoc and subprocess pagers never block on interactive terminal input (e.g. less)
pydoc.pager = pydoc.plainpager
os.environ.setdefault("PAGER", "cat")

from .research import oeis_lookup, read_paper, search_math
from .token_shield import sanitize_traceback, shield_output, summarize_value


# ==============================================================================
# Thread-local runtime network guard
# ==============================================================================
#
# Installed ONCE at module import. The wrappers are no-ops unless the *current
# thread* is inside a sandboxed exec (flag set by PersistentMathKernel.execute when
# `_network_blocked_during_exec()` is True, i.e. TCSCompliantKernel/AuditorKernel with
# allow_internet=False). The harness's own LLM HTTP calls run on the same worker
# threads but outside exec, so they are unaffected.
#
# Limitations (defense-in-depth only, not a real sandbox): threads or subprocesses
# spawned by model code do not inherit the flag. The AST denylist in prover_kernel.py
# blocks the obvious spawn routes; true isolation needs a child process + `unshare -n`.

_sandbox_state = threading.local()
_NET_FAMILIES = frozenset(
    fam for fam in (getattr(_socket_mod, "AF_INET", None), getattr(_socket_mod, "AF_INET6", None))
    if fam is not None
)
_SANDBOX_NET_MSG = (
    "[SANDBOX] Network access is disabled inside the REPL for this run "
    "(allow_internet=False). Work only from the problem statement and given context."
)


def _in_sandboxed_exec() -> bool:
    """True iff the current thread is executing model code with network blocked."""
    return bool(getattr(_sandbox_state, "in_sandboxed_exec", False))


def _install_socket_guard() -> None:
    """Wrap socket connect/sendto/create_connection exactly once per process."""
    if getattr(_socket_mod, "_prime_math_guard_installed", False):
        return

    sock_cls = _socket_mod.socket
    orig_connect = sock_cls.connect
    orig_connect_ex = sock_cls.connect_ex
    orig_sendto = sock_cls.sendto
    orig_create_connection = _socket_mod.create_connection

    def _check(sock: Any) -> None:
        if _in_sandboxed_exec() and getattr(sock, "family", None) in _NET_FAMILIES:
            raise PermissionError(_SANDBOX_NET_MSG)

    @functools.wraps(orig_connect)
    def guarded_connect(self, *args, **kwargs):
        _check(self)
        return orig_connect(self, *args, **kwargs)

    @functools.wraps(orig_connect_ex)
    def guarded_connect_ex(self, *args, **kwargs):
        _check(self)
        return orig_connect_ex(self, *args, **kwargs)

    @functools.wraps(orig_sendto)
    def guarded_sendto(self, *args, **kwargs):
        _check(self)
        return orig_sendto(self, *args, **kwargs)

    @functools.wraps(orig_create_connection)
    def guarded_create_connection(*args, **kwargs):
        if _in_sandboxed_exec():
            raise PermissionError(_SANDBOX_NET_MSG)
        return orig_create_connection(*args, **kwargs)

    sock_cls.connect = guarded_connect
    sock_cls.connect_ex = guarded_connect_ex
    sock_cls.sendto = guarded_sendto
    _socket_mod.create_connection = guarded_create_connection
    _socket_mod._prime_math_guard_installed = True


_install_socket_guard()

MAX_COMBINATORIAL_STEPS = 200_000


def _in_repl_exec() -> bool:
    """True iff the current thread is executing model code inside PersistentMathKernel.execute."""
    return bool(getattr(_sandbox_state, "in_repl_exec", False))


def _install_itertools_guard() -> None:
    """Wrap itertools.combinations, permutations, combinations_with_replacement, and product
    so runaway combinatorial loops or list(combinations(range(3000), 3)) calls inside REPL
    exec fail fast in <20ms instead of spinning CPU or exhausting unified memory.
    """
    if getattr(itertools, "_prime_math_guard_installed", False):
        return

    for _fn_name in ("combinations", "permutations", "combinations_with_replacement", "product"):
        _orig_fn = getattr(itertools, _fn_name, None)
        if _orig_fn is None:
            continue

        def _make_guarded(orig_f: Any, name: str):
            @functools.wraps(orig_f)
            def _guarded(*args, **kwargs):
                it = orig_f(*args, **kwargs)
                if not _in_repl_exec():
                    return it

                def _bounded_gen():
                    for idx, item in enumerate(it, 1):
                        if idx > MAX_COMBINATORIAL_STEPS:
                            raise RuntimeError(
                                f"[TCS-BENCH REPL CEILING] itertools.{name} exceeded {MAX_COMBINATORIAL_STEPS:,} iterations. "
                                "Restrict test instance size (e.g. N <= 20, k <= 4) or use Z3 / preloaded kernel helpers instead of brute-force enumeration."
                            )
                        yield item

                return _bounded_gen()

            return _guarded

        setattr(itertools, _fn_name, _make_guarded(_orig_fn, _fn_name))

    itertools._prime_math_guard_installed = True


_install_itertools_guard()


# ==============================================================================
# Trace-tampering pre-check (the timeout watchdog depends on sys.settrace)
# ==============================================================================

_TRACE_TAMPER_NAMES = frozenset({
    "settrace", "setprofile", "f_trace", "f_trace_lines", "f_trace_opcodes",
})


def _detect_trace_tampering(tree: ast.AST) -> Optional[str]:
    """Return an error message if code references trace/profile hooks, else None."""
    for node in ast.walk(tree):
        hit: Optional[str] = None
        if isinstance(node, ast.Name) and node.id in _TRACE_TAMPER_NAMES:
            hit = node.id
        elif isinstance(node, ast.Attribute) and node.attr in _TRACE_TAMPER_NAMES:
            hit = node.attr
        elif isinstance(node, ast.alias) and node.name.split(".")[-1] in _TRACE_TAMPER_NAMES:
            hit = node.name
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            for name in _TRACE_TAMPER_NAMES:
                if name in node.value:
                    hit = name
                    break
        if hit:
            return (
                f"[SANDBOX] Code references '{hit}': changing interpreter trace/profile hooks is "
                "disabled because the REPL timeout watchdog depends on them. Remove that call and re-run."
            )
    return None


# ==============================================================================
# Optional process-wide memory cap (NOT applied by default)
# ==============================================================================

def apply_process_memory_cap(gb: float) -> Tuple[int, int]:
    """Opt-in: cap this process's virtual address space at `gb` GiB via RLIMIT_AS.

    NOT called anywhere by default. Intended for a dedicated child process that runs
    REPL code (e.g. a future subprocess kernel), or a standalone benchmark worker on this
    unified-memory APU where a runaway allocation can starve the GPU and freeze the machine.

    Caveats:
    - Process-wide: it limits the harness itself (LLM client, TurboVec, etc.), not just exec'd code.
    - RLIMIT_AS counts *virtual* address space. numpy/OpenBLAS thread arenas, ROCm/Vulkan and
      large mmaps reserve far more VA than RSS, so a cap close to real usage causes spurious
      MemoryError. Leave generous headroom (measure VmPeak in /proc/self/status first).
    - Only the soft limit is lowered (hard limit unchanged), so it can be restored with the
      returned previous limits: resource.setrlimit(resource.RLIMIT_AS, previous).
    - Linux/Unix only.

    Returns the previous (soft, hard) RLIMIT_AS limits.
    """
    import resource

    if not gb or gb <= 0:
        raise ValueError(f"apply_process_memory_cap: gb must be > 0, got {gb!r}")
    limit = int(gb * (1024 ** 3))
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY and limit > hard:
        limit = hard
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    return soft, hard


# ==============================================================================
# Honest, explicitly empirical relational check (replaces the old Relational.__bool__ patch)
# ==============================================================================

def _default_samples() -> Tuple[Any, ...]:
    import sympy as sp
    R = sp.Rational
    return (-10, -3, -2, -1, R(-1, 2), 0, R(1, 3), R(1, 2), 1, 2, 3, 5, 7, 10, 100)


def _sample_compatible(sym: Any, val: Any) -> bool:
    """Skip sample values that contradict a symbol's declared assumptions."""
    try:
        if sym.is_integer and val.is_integer is False:
            return False
        if sym.is_positive and val.is_positive is False:
            return False
        if sym.is_nonnegative and val.is_negative:
            return False
        if sym.is_negative and val.is_negative is False:
            return False
        if sym.is_nonpositive and val.is_positive:
            return False
        if sym.is_zero is False and val.is_zero:
            return False
    except Exception:
        return False
    return True


def _truth_value(expr: Any) -> Optional[bool]:
    import sympy as sp

    if isinstance(expr, bool):
        return expr
    if isinstance(expr, sp.logic.boolalg.BooleanTrue):
        return True
    if isinstance(expr, sp.logic.boolalg.BooleanFalse):
        return False
    try:
        evaluated = expr.doit()
    except Exception:
        return None
    if isinstance(evaluated, sp.logic.boolalg.BooleanTrue):
        return True
    if isinstance(evaluated, sp.logic.boolalg.BooleanFalse):
        return False
    return None


def holds_on_samples(
    rel: Any,
    samples: Optional[Iterable[Any]] = None,
    max_points: int = 2000,
) -> Optional[bool]:
    """EMPIRICAL spot-check of a SymPy relation on a finite grid of sample points.

    THIS IS NOT A PROOF. It only substitutes sample values for the free symbols.

    Returns:
        False -- a concrete counterexample was found among the samples (definitive refutation).
        True  -- no counterexample among the evaluable samples (evidence only, NOT a proof:
                 the relation may still fail elsewhere).
        None  -- inconclusive: nothing could be evaluated (non-real values, division by zero,
                 non-boolean input, or no samples compatible with symbol assumptions).

    Samples respect declared symbol assumptions (e.g. positive=True, integer=True). With several
    free symbols the Cartesian product of samples is used, capped at `max_points` points.
    For actual proofs use sp.simplify/sp.factor on lhs - rhs, sp.solve_univariate_inequality,
    sp.reduce_inequalities, or z3, and reason about the result.
    """
    import sympy as sp

    if isinstance(rel, bool):
        return rel
    try:
        rel = sp.sympify(rel)
    except Exception:
        return None
    if not isinstance(rel, sp.logic.boolalg.Boolean):
        return None
    syms = sorted(rel.free_symbols, key=str)
    if not syms:
        return _truth_value(rel)

    try:
        values = [sp.sympify(v) for v in (samples if samples is not None else _default_samples())]
    except Exception:
        return None
    per_sym = [[v for v in values if _sample_compatible(s, v)] for s in syms]
    if any(not vs for vs in per_sym):
        return None

    evaluated = 0
    for point in itertools.islice(itertools.product(*per_sym), max(1, int(max_points))):
        try:
            truth = _truth_value(rel.subs(dict(zip(syms, point))))
        except Exception:
            truth = None  # e.g. "Invalid comparison of non-real" or NaN comparison
        if truth is False:
            return False
        if truth is True:
            evaluated += 1
    return True if evaluated else None


class ThreadAwareStream(io.TextIOBase):
    """Thread-safe stream wrapper that directs output to thread-local buffers

    when active, falling back to the original underlying stream. Prevents cross-task
    REPL stdout/stderr contamination in concurrent multi-worker runs.
    """

    def __init__(self, target_stream: Any):
        self._target = target_stream
        self._local = threading.local()

    def set_redirect(self, buf: io.StringIO) -> None:
        self._local.buf = buf

    def clear_redirect(self) -> None:
        if hasattr(self._local, "buf"):
            del self._local.buf

    def write(self, s: str) -> int:
        buf = getattr(self._local, "buf", None)
        if buf is not None:
            return buf.write(s)
        if self._target and hasattr(self._target, "write"):
            return self._target.write(s)
        return len(s)

    def flush(self) -> None:
        buf = getattr(self._local, "buf", None)
        if buf is not None:
            return buf.flush()
        if self._target and hasattr(self._target, "flush"):
            return self._target.flush()

    def isatty(self) -> bool:
        if getattr(self._local, "buf", None) is not None:
            return False
        if self._target and hasattr(self._target, "isatty"):
            return self._target.isatty()
        return False

    def fileno(self) -> int:
        if self._target and hasattr(self._target, "fileno"):
            return self._target.fileno()
        raise io.UnsupportedOperation("fileno")


_stdout_proxy: Optional[ThreadAwareStream] = None
_stderr_proxy: Optional[ThreadAwareStream] = None
_stream_lock = threading.Lock()


def ensure_stream_proxies() -> tuple[ThreadAwareStream, ThreadAwareStream]:
    """Ensures sys.stdout and sys.stderr are wrapped with thread-aware isolation proxies."""
    global _stdout_proxy, _stderr_proxy
    with _stream_lock:
        if _stdout_proxy is None or sys.stdout is not _stdout_proxy:
            orig_out = sys.stdout if not isinstance(sys.stdout, ThreadAwareStream) else _stdout_proxy._target
            _stdout_proxy = ThreadAwareStream(orig_out)
            sys.stdout = _stdout_proxy
        if _stderr_proxy is None or sys.stderr is not _stderr_proxy:
            orig_err = sys.stderr if not isinstance(sys.stderr, ThreadAwareStream) else _stderr_proxy._target
            _stderr_proxy = ThreadAwareStream(orig_err)
            sys.stderr = _stderr_proxy
        return _stdout_proxy, _stderr_proxy


@dataclass
class ExecutionResult:

    success: bool
    stdout: str
    stderr: str
    result_value: Any
    result_repr: Optional[str]
    error: Optional[str]
    execution_time: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": bool(self.success),
            "stdout": self.stdout or "",
            "stderr": self.stderr or "",
            "result_repr": self.result_repr,
            "error": self.error,
            "execution_time": float(self.execution_time or 0.0),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionResult":
        return cls(
            success=bool(data.get("success", False)),
            stdout=str(data.get("stdout") or ""),
            stderr=str(data.get("stderr") or ""),
            result_value=None,
            result_repr=data.get("result_repr"),
            error=data.get("error"),
            execution_time=float(data.get("execution_time") or 0.0),
        )


class TimeoutException(BaseException):
    """Raised when user code execution exceeds allowed runtime.

    Subclasses BaseException (not Exception) so the common LLM pattern
    `try: ... except Exception: pass` inside REPL code cannot swallow the timeout.
    """
    pass


class PersistentMathKernel:
    """Persistent Python execution environment with math preludes."""

    def __init__(self, default_timeout: float = 300.0):
        self.default_timeout = default_timeout
        self.namespace: Dict[str, Any] = {}
        self._prelude_keys: Set[str] = set()
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        """Resets the namespace back to the default mathematical prelude."""
        with self._lock:
            self._search_call_count = 0
            self.namespace.clear()
            self.namespace["__name__"] = "__main__"
            self._load_prelude()

    def _load_prelude(self) -> None:
        """Injects standard mathematical libraries and symbols into the session."""
        prelude_code = """
import math
import cmath
import itertools
import collections
import functools
import operator
import heapq
import random
from fractions import Fraction
from decimal import Decimal
from math import isqrt, comb, perm, factorial, prod
from collections import Counter, defaultdict, deque
from itertools import permutations, combinations, combinations_with_replacement, product, islice, chain, accumulate, groupby
from functools import lru_cache, cache

# Symbolic mathematics
import sympy as sp
from sympy import (
    Eq, solve, solveset, simplify, expand, factor, cancel,
    diff, integrate, summation, limit, series,
    isprime, factorint, divisors, divisor_count, gcd, lcm,
    primerange, primepi, nextprime, prevprime, primefactors,
    mod_inverse, totient, binomial, fibonacci, lucas, catalan, bell, harmonic,
    mobius, jacobi_symbol, legendre_symbol, sieve,
    npartitions, partition,
    continued_fraction, continued_fraction_periodic, continued_fraction_convergents,
    continued_fraction_iterator, continued_fraction_reduce,
    Matrix, Point, Line, Circle, Polygon,
    Rational, Integer, Float, oo, I, pi, E, exp, log, sin, cos, sqrt,
)

# Common symbolic variables
x, y, z, n, k, a, b, c, m, t = sp.symbols('x y z n k a b c m t')

# SymPy Expr ergonomics: allow p.degree(x) directly on Expr/Add objects
try:
    sp.Expr.degree = lambda self, *args, **kwargs: sp.degree(self, *args, **kwargs)
except Exception:
    pass

# NOTE: SymPy Relational truthiness is intentionally NOT patched. `bool(x >= 0)` raises
# TypeError for undecidable relations; use holds_on_samples(rel) (empirical only) or a real proof.


# SMT & Constraint Solving
try:
    import z3
    from z3 import (
        Int, Ints, Real, Reals, Bool, Bools,
        Solver, Optimize, sat, unsat, unknown,
        And, Or, Not, Implies, If, Distinct, Sum,
    )
    if not hasattr(z3.Solver, "constraints"):
        z3.Solver.constraints = property(lambda self: self.assertions())
except ImportError:
    pass

# Numerical and Arbitrary Precision
import numpy as np
import scipy
import scipy.special as sp_special
import mpmath as mp
mp.dps = 50  # 50 decimal digits of precision by default

# Ergonomic aliases for common LLM assumptions on numpy
if not hasattr(np, "gamma"):
    np.gamma = sp_special.gamma
if not hasattr(np, "comb"):
    np.comb = sp_special.comb
if not hasattr(np, "factorial"):
    np.factorial = sp_special.factorial
if not hasattr(np, "erf"):
    np.erf = sp_special.erf
if not hasattr(np, "beta"):
    np.beta = sp_special.beta
if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid

# Graph Theory and Combinatorics
try:
    import networkx as nx
    if not hasattr(nx, "Tree"):
        nx.Tree = nx.Graph
    if not hasattr(nx, "random_tree") and hasattr(nx, "random_labeled_tree"):
        nx.random_tree = nx.random_labeled_tree
    if not hasattr(nx, "vertex_connectivity") and hasattr(nx, "node_connectivity"):
        nx.vertex_connectivity = nx.node_connectivity
    if not hasattr(nx.Graph, "is_connected"):
        nx.Graph.is_connected = lambda self: nx.is_connected(self)
except ImportError:
    pass


# Accelerated Number Theory Utilities
MAX_DIAGNOSTIC_SIEVE_LIMIT = 200000

def fast_sieve(limit: int) -> np.ndarray:
    \"\"\"Returns numpy array of all primes <= limit using boolean array sieve.\"\"\"
    if limit > MAX_DIAGNOSTIC_SIEVE_LIMIT:
        raise RuntimeError(
            f"ExecutionBlockedByRule11: fast_sieve limit {limit:,} exceeds diagnostic ceiling ({MAX_DIAGNOSTIC_SIEVE_LIMIT:,}). "
            "Escalating empirical prime sieving cannot achieve high precision (statistical error O(1/sqrt(x))). "
            "Rule 11 requires formulating and evaluating the exact analytical formula, Euler product, or algebraic invariants."
        )
    if limit < 2:
        return np.array([], dtype=int)
    is_p = np.ones(limit + 1, dtype=bool)
    is_p[:2] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_p[i]:
            is_p[i*i::i] = False
    return np.nonzero(is_p)[0]

def fast_totients(limit: int) -> np.ndarray:
    \"\"\"Computes Euler's totient phi(n) for all 0 <= n <= limit in O(N log log N).\"\"\"
    if limit > MAX_DIAGNOSTIC_SIEVE_LIMIT:
        raise RuntimeError(
            f"ExecutionBlockedByRule11: fast_totients limit {limit:,} exceeds diagnostic ceiling ({MAX_DIAGNOSTIC_SIEVE_LIMIT:,}). "
            "Rule 11 requires formulating and evaluating exact analytical formulas or Euler products."
        )
    phi = np.arange(limit + 1)
    for p in range(2, limit + 1):
        if phi[p] == p:
            phi[p::p] -= phi[p::p] // p
    return phi

def fast_spf(limit: int) -> np.ndarray:
    \"\"\"Computes Smallest Prime Factor (SPF) for all 0 <= n <= limit in O(limit) vectorized steps.\"\"\"
    if limit > MAX_DIAGNOSTIC_SIEVE_LIMIT:
        raise RuntimeError(
            f"ExecutionBlockedByRule11: fast_spf limit {limit:,} exceeds diagnostic ceiling ({MAX_DIAGNOSTIC_SIEVE_LIMIT:,}). "
            "Rule 11 requires formulating and evaluating exact analytical formulas or Euler products."
        )
    spf = np.arange(limit + 1, dtype=np.int32)
    spf[4::2] = 2
    for i in range(3, int(limit**0.5) + 1, 2):
        if spf[i] == i:
            sl = slice(i * i, limit + 1, i)
            m = spf[sl]
            multiples = np.arange(i * i, limit + 1, i, dtype=np.int32)
            m[m == multiples] = i
    return spf

_orig_primerange = primerange
def primerange(a, b=None):
    high = b if b is not None else a
    if high > MAX_DIAGNOSTIC_SIEVE_LIMIT:
        raise RuntimeError(
            f"ExecutionBlockedByRule11: primerange({a}, {b}) upper bound {high:,} exceeds diagnostic ceiling ({MAX_DIAGNOSTIC_SIEVE_LIMIT:,}). "
            "Escalating empirical prime sieving cannot achieve high precision. Formulate and evaluate the exact analytical formula or Euler product."
        )
    return _orig_primerange(a, b)

def is_square(n: int) -> bool:
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n
"""
        exec(prelude_code, self.namespace)

        def rate_limited_search_math(*args, **kwargs):
            self._search_call_count += 1
            if self._search_call_count > 10:
                return [{
                    "title": "[RESEARCH CAP REACHED]",
                    "url": "",
                    "snippet": (
                        "[RESEARCH CAP REACHED (Max 10 searches)]: Literature search quota exhausted. "
                        "STOP querying search engines. As per Rule 16, pivot immediately to empirical computation "
                        "and numerical extrapolation with Python (Euler-Gauss Protocol) to determine the integer floor."
                    ),
                }]
            return search_math(*args, **kwargs)

        def rate_limited_read_paper(*args, **kwargs):
            if self._search_call_count > 10:
                return (
                    "[RESEARCH CAP REACHED]: Literature reading quota exhausted. "
                    "Pivot immediately to Python computation and extrapolation."
                )
            return read_paper(*args, **kwargs)

        self._live_research_tools: Dict[str, Any] = {
            "oeis_lookup": oeis_lookup,
            "search_math": rate_limited_search_math,
            "read_paper": rate_limited_read_paper,
        }
        self.namespace.update(self._live_research_tools)

        # Explicitly empirical relational spot-check (replaces the old unsound
        # Relational.__bool__ monkeypatch). Returns True/False/None; True is NOT a proof.
        self.namespace["holds_on_samples"] = holds_on_samples

        # z3 calls are C-level and cannot be interrupted by the settrace watchdog,
        # so bound them with a solver timeout equal to the kernel timeout.
        self._set_z3_timeout(self.default_timeout)

        _builtins_range = range
        def guarded_range(*args):
            if len(args) == 1:
                start, stop, step = 0, args[0], 1
            elif len(args) == 2:
                start, stop, step = args[0], args[1], 1
            elif len(args) == 3:
                start, stop, step = args[0], args[1], args[2]
            else:
                return _builtins_range(*args)

            if step != 0:
                count = max(0, (stop - start + (step - 1 if step > 0 else step + 1)) // step)
                if count > 200_000:
                    raise RuntimeError(
                        f"ExecutionBlockedByRule11: range with {count:,} elements exceeds diagnostic ceiling (200,000). "
                        "Escalating empirical simulations or brute-force loops cannot achieve high precision (statistical error O(1/sqrt(x))). "
                        "Transition immediately to formulating and evaluating the exact analytical formula, Euler product, or algebraic invariants."
                    )
            return _builtins_range(*args)

        self.namespace["range"] = guarded_range

        # Formal Mathematical State Machine
        from .state_machine import MathStateMachine, StateInvariantViolation
        self._state_machine = MathStateMachine(self.namespace)
        self.namespace["state"] = self._state_machine
        self.namespace["StateInvariantViolation"] = StateInvariantViolation

        # Lean 4 Verification Runner
        from .lean_runner import run_lean
        def lean_check(code: str, timeout: float = 30.0) -> Dict[str, Any]:
            """Lightweight runner to check a Lean 4 formal proof or snippet."""
            return run_lean(code, timeout=timeout).to_dict()

        self.namespace["lean_check"] = lean_check
        self.namespace["run_lean"] = lean_check

        self._prelude_keys = set(self.namespace.keys())
        self._prelude_keys.add("__builtins__")

    def execute_lean(self, code: str, timeout_seconds: Optional[float] = None):
        """Executes and verifies a Lean 4 code snippet using the local Lean compiler."""
        from .lean_runner import run_lean
        timeout = timeout_seconds or 30.0
        return run_lean(code, timeout=timeout)

    @staticmethod
    def _set_z3_timeout(timeout_seconds: Optional[float]) -> None:
        """Set z3's process-global solver timeout (ms) if z3 is importable. Best effort."""
        if not timeout_seconds or timeout_seconds <= 0:
            return
        try:
            import z3  # noqa: WPS433 (optional dependency)
            z3.set_param("timeout", int(timeout_seconds * 1000))
        except Exception:
            pass

    def _network_blocked_during_exec(self) -> bool:
        """Whether the thread-local socket guard should be armed while model code runs.

        The base kernel never blocks; TCSCompliantKernel / AuditorKernel override this to
        return `not self.allow_internet`.
        """
        return False

    @staticmethod
    def _error_result(error: str, execution_time: float = 0.0) -> ExecutionResult:
        return ExecutionResult(
            success=False, stdout="", stderr="", result_value=None,
            result_repr=None, error=error, execution_time=execution_time,
        )

    def execute(self, code: str, timeout_seconds: Optional[float] = None) -> ExecutionResult:
        """Executes a code string within the persistent namespace."""
        timeout = timeout_seconds or self.default_timeout
        code = code.strip()
        if not code:
            return ExecutionResult(
                success=True,
                stdout="",
                stderr="",
                result_value=None,
                result_repr=None,
                error=None,
                execution_time=0.0,
            )

        # Parse and pre-check outside the watchdog: syntax errors and trace tampering
        # are reported as ordinary error results, never raised.
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as exc:
            return self._error_result(sanitize_traceback(exc))
        tamper = _detect_trace_tampering(tree)
        if tamper:
            return self._error_result(tamper)

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        start_time = time.perf_counter()
        result_val = None
        result_repr = None
        err_msg = None
        success = True
        block_network = self._network_blocked_during_exec()

        # `with` guarantees the lock is released on every path, including re-raised
        # KeyboardInterrupt.
        with self._lock:
            # We enforce execution timeout using both an asynchronous exception injection
            # (which reliably aborts tight while loops and CPython bytecode execution even
            # when specializing interpreter optimization bypasses sys.settrace) and a watchdog trace.
            timer_expired = threading.Event()
            target_ident = threading.get_ident()

            def interrupt_hook():
                timer_expired.set()
                try:
                    ctypes.pythonapi.PyThreadState_SetAsyncExc(
                        ctypes.c_ulong(target_ident),
                        ctypes.py_object(TimeoutException),
                    )
                except Exception:
                    pass

            timer = threading.Timer(timeout, interrupt_hook)
            timer.daemon = True

            # Trace function to safely abort runaway loops. It keeps raising on every
            # subsequent event, so even `except BaseException: pass` only delays the abort
            # by one handler line.
            def trace_dispatch(frame, event, arg):
                if timer_expired.is_set():
                    raise TimeoutException(
                        f"Execution timed out after {timeout:.1f}s. "
                        "Check for infinite loops or expensive brute-force search."
                    )
                return trace_dispatch

            old_trace = sys.gettrace()
            prev_net_flag = _in_sandboxed_exec()
            prev_repl_flag = _in_repl_exec()
            self._set_z3_timeout(timeout)

            out_proxy, err_proxy = ensure_stream_proxies()
            out_proxy.set_redirect(stdout_buffer)
            err_proxy.set_redirect(stderr_buffer)
            try:
                try:
                    timer.start()
                    _sandbox_state.in_sandboxed_exec = block_network
                    _sandbox_state.in_repl_exec = True
                    sys.settrace(trace_dispatch)

                    # If the final AST node is an expression (e.g. `solve(eq, x)`),
                    # execute the preceding statements, then evaluate and capture the last expression.
                    if tree.body and isinstance(tree.body[-1], ast.Expr):
                        last_expr_node = tree.body.pop()
                        if tree.body:
                            compiled_stmts = compile(tree, filename="<prime_math>", mode="exec")
                            exec(compiled_stmts, self.namespace)

                        compiled_expr = compile(
                            ast.Expression(body=last_expr_node.value),
                            filename="<prime_math>",
                            mode="eval",
                        )
                        result_val = eval(compiled_expr, self.namespace)
                        if result_val is not None:
                            result_repr = summarize_value(result_val, max_len=500)
                    else:
                        compiled_code = compile(tree, filename="<prime_math>", mode="exec")
                        exec(compiled_code, self.namespace)
                finally:
                    # Disarm the watchdog and network guard BEFORE any handler runs, so a
                    # timeout firing during error formatting cannot escape execute().
                    sys.settrace(old_trace)
                    _sandbox_state.in_sandboxed_exec = prev_net_flag
                    _sandbox_state.in_repl_exec = prev_repl_flag
                    timer.cancel()
            except TimeoutException as exc:
                success = False
                if not str(exc):
                    exc = TimeoutException(
                        f"Execution timed out after {timeout:.1f}s. "
                        "Check for infinite loops or expensive brute-force search."
                    )
                err_msg = sanitize_traceback(exc)
            except KeyboardInterrupt:
                raise
            except SystemExit as exc:
                # Model scripts often end with `sys.exit(1)` / `exit()`. Re-raising would kill
                # the whole benchmark worker, so report it as the end of this cell instead.
                if exc.code not in (None, 0):
                    success = False
                    err_msg = (
                        f"SystemExit({exc.code!r}): code called exit()/sys.exit(); "
                        "execution of this cell stopped (REPL state preserved)."
                    )
            except BaseException as exc:
                success = False
                err_msg = sanitize_traceback(exc)
            finally:
                out_proxy.clear_redirect()
                err_proxy.clear_redirect()

        exec_time = time.perf_counter() - start_time
        clean_stdout = shield_output(stdout_buffer.getvalue())
        clean_stderr = shield_output(stderr_buffer.getvalue())

        return ExecutionResult(
            success=success,
            stdout=clean_stdout,
            stderr=clean_stderr,
            result_value=result_val,
            result_repr=result_repr,
            error=err_msg,
            execution_time=round(exec_time, 4),
        )

    def get_user_variables(self) -> Dict[str, str]:
        """Returns non-prelude variables and their concise string previews."""
        with self._lock:
            user_vars = {}
            for k, v in self.namespace.items():
                if k.startswith("_") or k in self._prelude_keys:
                    continue
                user_vars[k] = summarize_value(v)
            return user_vars
