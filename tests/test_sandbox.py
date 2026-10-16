"""Sandbox soundness tests for PersistentMathKernel / TCSCompliantKernel / AuditorKernel.

Covers: BaseException timeouts, trace-tampering pre-check, removal of the unsound SymPy
Relational.__bool__ patch (holds_on_samples replacement), network isolation (AST denylist +
thread-local runtime socket guard + allow_internet property), and math_tools predicate errors.
All workloads are tiny (1s timeouts, loopback sockets, graphs with n <= 4).
"""

import socket
import sys
import threading

import pytest
import sympy as sp

from core_engine.kernel import (
    PersistentMathKernel,
    TimeoutException,
    _in_sandboxed_exec,
    apply_process_memory_cap,
    holds_on_samples,
)
from core_engine.math_tools import find_graph_counterexample
from core_engine.prover_kernel import AuditorKernel, TCSCompliantKernel


# ---------------------------------------------------------------------------
# 1. Timeout soundness
# ---------------------------------------------------------------------------

def test_timeout_exception_is_base_exception():
    assert issubclass(TimeoutException, BaseException)
    assert not issubclass(TimeoutException, Exception)


@pytest.mark.parametrize("handler", ["Exception"])
def test_timeout_not_swallowed_by_model_except(handler):
    k = PersistentMathKernel(default_timeout=1.0)
    code = f"i = 0\nwhile True:\n    try:\n        i += 1\n    except {handler}:\n        pass\n"
    res = k.execute(code)
    assert not res.success
    assert "timed out" in res.error.lower()
    assert not k._lock.locked()
    # Kernel remains usable after a timeout.
    assert k.execute("1 + 1").result_repr == "2"


def test_keyboard_interrupt_reraised_and_state_restored():
    k = PersistentMathKernel(default_timeout=5.0)
    trace_before = sys.gettrace()
    with pytest.raises(KeyboardInterrupt):
        k.execute("raise KeyboardInterrupt")
    assert not k._lock.locked()
    assert sys.gettrace() is trace_before
    assert not _in_sandboxed_exec()
    res = k.execute("print('still alive')")
    assert res.success and "still alive" in res.stdout


def test_model_sys_exit_does_not_kill_harness():
    k = PersistentMathKernel(default_timeout=5.0)
    res = k.execute("import sys\nsys.exit(1)")
    assert not res.success
    assert "SystemExit" in res.error
    assert k.execute("exit()").success
    assert not k._lock.locked()


# ---------------------------------------------------------------------------
# 2. Trace disabling is rejected
# ---------------------------------------------------------------------------

def test_sys_not_in_prelude_namespace():
    k = PersistentMathKernel()
    assert "sys" not in k.namespace


@pytest.mark.parametrize("code", [
    "import sys\nsys.settrace(None)",
    "import sys\nsys.setprofile(None)",
    "from sys import settrace\nsettrace(None)",
    "import threading\nthreading.settrace(None)",
    "import sys\ngetattr(sys, 'settrace')(None)",
    "import sys\nsys._getframe().f_trace = None",
])
def test_trace_tampering_rejected_with_error_result(code):
    k = PersistentMathKernel(default_timeout=2.0)
    trace_before = sys.gettrace()
    res = k.execute(code)
    assert not res.success
    assert "[SANDBOX]" in res.error
    assert sys.gettrace() is trace_before


# ---------------------------------------------------------------------------
# 3. z3 timeout
# ---------------------------------------------------------------------------

def test_z3_global_timeout_set_from_kernel_timeout():
    z3 = pytest.importorskip("z3")
    PersistentMathKernel(default_timeout=7.0)
    assert z3.get_param("timeout") == "7000"


# ---------------------------------------------------------------------------
# 4. Relational patch removed; holds_on_samples is explicit and honest
# ---------------------------------------------------------------------------

def test_relational_bool_not_monkeypatched():
    PersistentMathKernel()
    w = sp.Symbol("w")
    with pytest.raises(TypeError):
        bool(w >= 0)


def test_unsound_assert_no_longer_passes():
    # Old patch: only checked x in {2,3,5,10} -> all zero -> assert passed. False at x=7.
    k = PersistentMathKernel()
    res = k.execute("assert (x-2)*(x-3)*(x-5)*(x-10) >= 0")
    assert not res.success
    assert "TypeError" in res.error


def test_holds_on_samples_semantics():
    x = sp.Symbol("x")
    p = sp.Symbol("p", positive=True)
    assert holds_on_samples((x - 2) * (x - 3) * (x - 5) * (x - 10) >= 0) is False
    assert holds_on_samples(x**2 >= 0) is True
    # Assumption-aware sampling: false for negative x, true on positive samples.
    assert holds_on_samples(x**3 - x**2 + x >= 0) is False
    assert holds_on_samples(p**3 - p**2 + p >= 0) is True
    # Inconclusive: only non-real evaluations, or non-boolean input.
    assert holds_on_samples(sp.sqrt(x) > -1, samples=[-1, -2]) is None
    assert holds_on_samples(x + 1) is None
    assert holds_on_samples(sp.Integer(3) > 2) is True
    assert "NOT A PROOF" in holds_on_samples.__doc__


def test_holds_on_samples_available_in_namespace():
    k = PersistentMathKernel()
    assert k.execute("holds_on_samples(x**2 + 1 > 0)").result_repr == "True"


# ---------------------------------------------------------------------------
# 5. Network isolation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("code", [
    "__import__('socket')",
    "m = 'soc' + 'ket'\n__import__(m)",
    "import importlib",
    "import importlib.util",
    "import os\nos.system('true')",
    "import os\nos.popen('true')",
    "from os import system",
    "import subprocess",
    "from subprocess import run",
    "import sys\nsys.modules['httpx']",
    "exec('import socket')",
    "eval(\"__import__('os')\")",
    "import _socket",
    "import urllib.request",
])
def test_tcs_ast_denylist_blocks_escape_hatches(code):
    k = TCSCompliantKernel(allow_internet=False)
    res = k.execute(code)
    assert not res.success
    assert "[TCS-BENCH PROTOCOL]" in res.error


@pytest.mark.parametrize("code,expected", [
    ("__import__('math').sqrt(4)", "2.0"),
    ("import os\nos.path.join('a', 'b')", "'a/b'"),
    ("eval('1 + 2')", "3"),
])
def test_tcs_ast_denylist_allows_benign_code(code, expected):
    k = TCSCompliantKernel(allow_internet=False)
    res = k.execute(code)
    assert res.success, res.error
    assert res.result_repr == expected


@pytest.fixture
def loopback_server():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(16)
    try:
        yield srv.getsockname()
    finally:
        srv.close()


def _make_probe(addr):
    """Harness-side helper that opens a TCP connection; injected into the namespace so the
    connect happens *inside* exec without tripping the AST denylist."""
    def probe():
        s = socket.create_connection(addr, timeout=2)
        s.close()
        return "connected"

    def raw_probe():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.settimeout(2)
            s.connect(addr)
            return "connected"
        finally:
            s.close()
    return probe, raw_probe


def test_runtime_guard_blocks_inside_exec_but_not_harness(loopback_server):
    k = TCSCompliantKernel(allow_internet=False)
    probe, raw_probe = _make_probe(loopback_server)
    k.namespace["_probe"] = probe
    k.namespace["_raw_probe"] = raw_probe

    for name in ("_probe()", "_raw_probe()"):
        res = k.execute(name)
        assert not res.success
        assert "PermissionError" in res.error and "[SANDBOX]" in res.error

    # Same thread, outside exec (like the harness's own LLM HTTP calls): works.
    assert not _in_sandboxed_exec()
    assert probe() == "connected"
    assert raw_probe() == "connected"


def test_runtime_guard_is_thread_local(loopback_server):
    k = TCSCompliantKernel(allow_internet=False)
    probe, _ = _make_probe(loopback_server)
    entered, release = threading.Event(), threading.Event()

    def wait_inside_exec():
        entered.set()
        release.wait(5)
        return "done"

    k.namespace["_wait"] = wait_inside_exec
    results = {}
    worker = threading.Thread(target=lambda: results.setdefault("r", k.execute("_wait()", timeout_seconds=10)))
    worker.start()
    try:
        assert entered.wait(5)
        # Another thread (e.g. a concurrent worker's LLM call) keeps network access
        # while the sandboxed exec is in progress.
        assert probe() == "connected"
    finally:
        release.set()
        worker.join(10)
    assert results["r"].success and results["r"].result_repr == "'done'"


def test_allow_internet_true_does_not_block(loopback_server):
    k = TCSCompliantKernel(allow_internet=True)
    probe, _ = _make_probe(loopback_server)
    k.namespace["_probe"] = probe
    res = k.execute("_probe()")
    assert res.success, res.error
    assert k.execute("import socket").success


def test_allow_internet_property_flip_reinstalls_tools_and_guard(loopback_server):
    k = TCSCompliantKernel(allow_internet=False)
    probe, _ = _make_probe(loopback_server)
    stub_title = k.namespace["search_math"]("x")[0]["title"]
    assert "TCS-BENCH PROTOCOL" in stub_title

    k.allow_internet = True
    assert k.namespace["search_math"] is k._live_research_tools["search_math"]
    assert k.namespace["oeis_lookup"] is k._live_research_tools["oeis_lookup"]
    k.namespace["_probe"] = probe
    assert k.execute("_probe()").success

    k.allow_internet = False
    assert k.namespace["search_math"] is not k._live_research_tools["search_math"]
    assert "TCS-BENCH PROTOCOL" in k.namespace["read_paper"]("x")
    res = k.execute("_probe()")
    assert not res.success and "PermissionError" in res.error


def test_auditor_kernel_offline_by_default(loopback_server):
    a = AuditorKernel(default_timeout=5.0)
    assert a.allow_internet is False
    assert "TCS-BENCH PROTOCOL" in a.namespace["search_math"]("x")[0]["title"]
    assert "TCS-BENCH PROTOCOL" in a.namespace["oeis_lookup"]("1,2,3")
    res = a.execute("import requests")
    assert not res.success and "[TCS-BENCH PROTOCOL]" in res.error
    probe, _ = _make_probe(loopback_server)
    a.namespace["_probe"] = probe
    res = a.execute("_probe()")
    assert not res.success and "PermissionError" in res.error
    # Ordinary auditing code still works.
    assert a.execute("assert pow(2, 3, 13) == 8").success


def test_state_machine_exists_after_reset():
    for k in (TCSCompliantKernel(), AuditorKernel()):
        k.reset()
        assert k._state_machine is not None
        assert k.namespace["state"] is k._state_machine
        assert "StateInvariantViolation" in k.namespace


# ---------------------------------------------------------------------------
# 6. math_tools: predicate exceptions are never reported as "holds"
# ---------------------------------------------------------------------------

def test_graph_predicate_errors_are_inconclusive():
    res = find_graph_counterexample(lambda G: G.graph["missing_key"], max_n=4)
    assert res["counterexample_found"] is False
    assert res["verified"] is False
    assert res["inconclusive"] is True
    assert res["predicate_errors"] == res["graphs_tested"]
    assert res["status"].startswith("INCONCLUSIVE")
    assert "holds" not in res["status"].lower()


def test_graph_counterexample_still_found_after_errors():
    def pred(G):
        if G.number_of_nodes() == 2:
            raise ValueError("boom")
        return G.number_of_edges() <= G.number_of_nodes()  # fails on K_4

    res = find_graph_counterexample(pred, max_n=4)
    assert res["counterexample_found"] is True


def test_graph_predicate_valid_is_verified():
    res = find_graph_counterexample(lambda G: G.number_of_nodes() >= 1, max_n=4)
    assert res["counterexample_found"] is False
    assert res["verified"] is True and res["inconclusive"] is False


# ---------------------------------------------------------------------------
# 7. Memory cap helper (validation only -- never applied in-process by tests)
# ---------------------------------------------------------------------------

def test_memory_cap_rejects_nonpositive():
    with pytest.raises(ValueError):
        apply_process_memory_cap(0)
    with pytest.raises(ValueError):
        apply_process_memory_cap(-1.5)
