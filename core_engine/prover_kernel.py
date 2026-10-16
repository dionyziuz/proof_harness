"""Execution Kernels and AST Safety Ceilings for Theorem Proving.

Provides:
- TCSCompliantKernel: Sandboxed persistent execution kernel with network AST inspection,
  combinatorial ceiling protection, and domain tool preludes.
- AuditorKernel: Ephemeral, isolated execution kernel dedicated strictly to empirical
  witness falsification and proof auditing.

Network isolation (when allow_internet is False) is layered:
1. Research tools (search_math / read_paper / oeis_lookup) are replaced by protocol stubs.
2. An AST denylist rejects network imports and common escape hatches (__import__, importlib,
   os.system/popen, subprocess, sys.modules, exec/eval of import strings).
3. A thread-local runtime guard (installed in kernel.py) makes socket connects raise
   PermissionError while model code is executing on the current thread.
None of this is a real OS sandbox; see kernel.py for limitations.
"""

from __future__ import annotations

import ast
from typing import Any, List, Optional

from .kernel import ExecutionResult, PersistentMathKernel

_NETWORK_MODULES = frozenset({
    "http", "httpx", "requests", "urllib", "urllib3", "socket", "_socket", "socketserver", "ssl",
    "ftplib", "smtplib", "telnetlib", "imaplib", "poplib", "nntplib", "ldap",
    "websockets", "websocket", "aiohttp", "xmlrpc", "webbrowser", "pycurl",
})

# Modules that are escape hatches for the AST import scan (dynamic import / process spawn).
_ESCAPE_MODULES = frozenset({
    "importlib", "subprocess", "builtins", "pty", "ctypes",
})

# Attribute calls on `os` that spawn processes (could shell out to curl/wget).
_OS_SPAWN_ATTRS = frozenset({"system", "popen", "startfile", "fork", "forkpty"})
_OS_SPAWN_PREFIXES = ("exec", "spawn", "posix_spawn")

_TCS_PROTOCOL_MSG = (
    "[TCS-BENCH PROTOCOL: internet access DISABLED for this run] "
    "This benchmark prohibits all external sources (the task's source paper included). "
    "Work only from the problem statement and context given in the task prompt."
)


def _protocol_block(what: str) -> str:
    return (
        f"[TCS-BENCH PROTOCOL] {what} is disabled: "
        f"all internet access is prohibited for this run. "
        f"Work only from the problem statement and given context."
    )


def _string_constants(node: ast.AST) -> List[str]:
    return [
        n.value for n in ast.walk(node)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    ]


def _is_blocked_module(name: str) -> bool:
    root = name.split(".", 1)[0]
    return root in _NETWORK_MODULES or root in _ESCAPE_MODULES


def _scan_network_ast(code: str) -> Optional[str]:
    """Return a protocol error if `code` references network modules or escape hatches, else None."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        # 1. Import statements
        names: List[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names = [node.module]
                if node.module.split(".", 1)[0] == "os":
                    for alias in node.names:
                        if alias.name in _OS_SPAWN_ATTRS or alias.name.startswith(_OS_SPAWN_PREFIXES):
                            return _protocol_block(f"'from os import {alias.name}'")
                if node.module == "sys" and any(a.name == "modules" for a in node.names):
                    return _protocol_block("'sys.modules' access")
        for name in names:
            if _is_blocked_module(name):
                return _protocol_block(f"import of '{name}'")

        # 2. Bare names: importlib, subprocess, socket, __builtins__ ...
        #    (__import__ is handled precisely in the Call branch below)
        if isinstance(node, ast.Name):
            if node.id in ("importlib", "subprocess", "socket", "__builtins__", "builtins"):
                return _protocol_block(f"use of '{node.id}'")

        # 3. Attribute access: os.system / os.popen / sys.modules / X.subprocess ...
        if isinstance(node, ast.Attribute):
            base = node.value.id if isinstance(node.value, ast.Name) else None
            if base == "os" and (node.attr in _OS_SPAWN_ATTRS or node.attr.startswith(_OS_SPAWN_PREFIXES)):
                return _protocol_block(f"'os.{node.attr}'")
            if base == "sys" and node.attr == "modules":
                return _protocol_block("'sys.modules' access")
            if node.attr in ("subprocess", "importlib", "__import__"):
                return _protocol_block(f"'.{node.attr}' access")

        # 4. Calls: __import__(...), exec/eval/compile of strings containing imports
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fname = node.func.id
            if fname == "__import__":
                arg0 = node.args[0] if node.args else None
                if not (isinstance(arg0, ast.Constant) and isinstance(arg0.value, str)) or _is_blocked_module(arg0.value):
                    return _protocol_block("'__import__' of a network/dynamic module")
            elif fname in ("exec", "eval", "compile"):
                for s in (c for arg in node.args for c in _string_constants(arg)):
                    if "import" in s:
                        return _protocol_block(f"'{fname}' of a string containing an import")

        # 5. Any string literal that names __import__ (e.g. getattr(builtins_obj, "__import__"))
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value == "__import__":
            return _protocol_block("string reference to '__import__'")
    return None


class _NetworkIsolationMixin:
    """Shared allow_internet handling for TCSCompliantKernel and AuditorKernel.

    `allow_internet` is a property: flipping it after construction (prover.py does this per
    solve) immediately swaps research tools between live and stubbed versions, and the
    runtime socket guard / AST denylist read it on every execute().
    """

    _allow_internet: bool = False

    @property
    def allow_internet(self) -> bool:
        return self._allow_internet

    @allow_internet.setter
    def allow_internet(self, value: bool) -> None:
        self._allow_internet = bool(value)
        if getattr(self, "_prelude_ready", False):
            self._apply_network_policy()

    def _network_blocked_during_exec(self) -> bool:
        return not self._allow_internet

    def _apply_network_policy(self) -> None:
        """(Re)install live research tools or protocol stubs according to allow_internet."""
        if self._allow_internet:
            live = getattr(self, "_live_research_tools", None) or {}
            self.namespace.update(live)
            return

        def stub_search(*args, **kwargs):
            return [{
                "title": "[TCS-BENCH PROTOCOL: internet access disabled]",
                "url": "",
                "snippet": _TCS_PROTOCOL_MSG,
            }]

        def stub_read_paper(*args, **kwargs):
            return _TCS_PROTOCOL_MSG

        def stub_oeis(*args, **kwargs):
            return _TCS_PROTOCOL_MSG

        self.namespace["search_math"] = stub_search
        self.namespace["read_paper"] = stub_read_paper
        self.namespace["oeis_lookup"] = stub_oeis

    def _forbidden_import(self, code: str) -> Optional[str]:
        """Return a protocol error if `code` uses network modules / escape hatches and internet
        is disabled, else None."""
        if self._allow_internet:
            return None
        return _scan_network_ast(code)


class TCSCompliantKernel(_NetworkIsolationMixin, PersistentMathKernel):
    """Harness kernel with optional web research tools and safety ceilings.

    When allow_internet is False (default for TCS-Bench), replaces search_math / read_paper
    with protocol stubs and AST-scans code for network imports to prevent benchmark contamination.
    When allow_internet is True (for open research theorems), allows live research queries.
    """

    def __init__(self, default_timeout: float = 30.0, allow_internet: bool = False):
        self._prelude_ready = False
        self.allow_internet = allow_internet
        super().__init__(default_timeout)

    def _combinatorial_ceiling(self, code: str) -> Optional[str]:
        """Guard against runaway factorial or combinatorial loops (e.g. permutations on n >= 7)."""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func_name = ""
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        func_name = node.func.attr
                    if func_name == "permutations" and node.args:
                        arg0 = node.args[0]
                        if isinstance(arg0, ast.Call) and getattr(arg0.func, "id", "") == "range" and arg0.args:
                            if isinstance(arg0.args[0], ast.Constant) and isinstance(arg0.args[0].value, int) and arg0.args[0].value >= 7:
                                return "[TCS-BENCH REPL CEILING] permutations on n >= 7 is prohibited (O(n!) complexity). Restrict instance size to n <= 6."
        except Exception:
            pass
        return None

    def execute(self, code: str, timeout_seconds: Optional[float] = None) -> ExecutionResult:
        blocked = self._forbidden_import(code.strip()) or self._combinatorial_ceiling(code.strip())
        if blocked:
            return ExecutionResult(
                success=False, stdout="", stderr="", result_value=None,
                result_repr=None, error=blocked, execution_time=0.0
            )
        return super().execute(code, timeout_seconds or self.default_timeout)

    def _load_prelude(self) -> None:
        super()._load_prelude()
        self._apply_network_policy()

        try:
            import networkx as _nx
            if not hasattr(_nx, "has_edge"):
                _nx.has_edge = lambda G, u, v: G.has_edge(u, v)
        except ImportError:
            pass

        try:
            from .math_tools import check_asymptotic_dominance, check_event_covariance, find_graph_counterexample
            self.namespace["check_asymptotic_dominance"] = check_asymptotic_dominance
            self.namespace["check_event_covariance"] = check_event_covariance
            self.namespace["find_graph_counterexample"] = find_graph_counterexample
        except Exception:
            pass
        # The base prelude already created self._state_machine / namespace["state"]
        # (bound to this same namespace dict), so no second construction is needed.

        self._prelude_keys = set(self.namespace.keys())
        self._prelude_keys.add("__builtins__")
        self._prelude_ready = True


class AuditorKernel(_NetworkIsolationMixin, PersistentMathKernel):
    """Ephemeral, isolated execution kernel dedicated strictly to empirical proof auditing.

    Guarantees:
    - Never shares state or variables with the solver's kernel.
    - Strict execution timeout (default 15.0s).
    - Pure computational and symbolic validation.
    - Offline by default (allow_internet=False): research tools stubbed, network AST denylist,
      and runtime socket guard during exec — same policy as TCSCompliantKernel.
    """

    def __init__(self, default_timeout: float = 15.0, allow_internet: bool = False):
        self._prelude_ready = False
        self.allow_internet = allow_internet
        super().__init__(default_timeout=default_timeout)
        try:
            import networkx as _nx
            self.namespace["networkx"] = _nx
            self.namespace["nx"] = _nx
        except ImportError:
            pass

    def _load_prelude(self) -> None:
        super()._load_prelude()
        self._apply_network_policy()
        self._prelude_ready = True

    def execute(self, code: str, timeout_seconds: Optional[float] = None) -> ExecutionResult:
        blocked = self._forbidden_import(code.strip())
        if blocked:
            return ExecutionResult(
                success=False, stdout="", stderr="", result_value=None,
                result_repr=None, error=blocked, execution_time=0.0
            )
        return super().execute(code, timeout_seconds)


__all__ = [
    "TCSCompliantKernel",
    "AuditorKernel",
    "_NETWORK_MODULES",
    "_TCS_PROTOCOL_MSG",
]
