import pytest
from unittest.mock import MagicMock, patch

from core_engine.prover import MathProver
from core_engine.prover_kernel import TCSCompliantKernel
from core_engine.prover_types import AuditResult, ProofResult
from core_engine.proof_extractor import (
    extract_latex_proof,
    is_valid_latex_proof,
    sanitize_informal_proof,
)


def test_is_valid_latex_proof():
    valid = r"""\begin{proof}
Let $G = (V, E)$ be a graph. We proceed by induction on $|V|$.
In the base case $|V| = 1$, the claim is immediate.
For the inductive step, assume the claim holds for all graphs on $< n$ vertices.
Then $G - v$ satisfies the inductive hypothesis. Thus the result follows.
\end{proof}"""
    assert is_valid_latex_proof(valid)

    # Empty or unclosed proof should fail
    assert not is_valid_latex_proof(r"\begin{proof} missing end")
    assert not is_valid_latex_proof("")
    assert not is_valid_latex_proof(None)

    # Proof with conversational markers should fail
    conversational = r"""\begin{proof}
Wait, let me rethink this approach. Could this be a contradiction?
Actually, the claim is trivial.
\end{proof}"""
    assert not is_valid_latex_proof(conversational)


def test_extract_latex_proof_direct():
    text = r"""Here is the complete formal proof:

\begin{proof}
Let $G$ be a connected bipartite graph.
We partition $V(G) = A \cup B$ such that all edges lie between $A$ and $B$.
Since $G$ is bipartite, it contains no odd cycles.
Therefore, the two-coloring is valid and unique up to swapping colors.
\end{proof}

I hope this helps!"""
    extracted = extract_latex_proof(text)
    assert extracted is not None
    assert extracted.startswith(r"\begin{proof}")
    assert extracted.endswith(r"\end{proof}")
    assert "connected bipartite graph" in extracted
    assert "I hope this helps" not in extracted


def test_extract_latex_proof_markdown_wrapped():
    text = r"""```latex
\begin{proof}
Let $T$ be a tree on $n \ge 1$ vertices.
By induction on $n$, every tree on $n \ge 2$ vertices has at least two leaves.
In the base case $n = 2$, $T$ is $K_2$ which has 2 leaves.
For $n > 2$, remove a leaf $v$. Then $T - v$ is a tree with at least 2 leaves.
Adding $v$ back maintains at least two leaves in $T$.
\end{proof}
```"""
    extracted = extract_latex_proof(text)
    assert extracted is not None
    assert extracted.startswith(r"\begin{proof}")
    assert extracted.endswith(r"\end{proof}")
    assert "By induction on $n$" in extracted


def test_extract_latex_proof_nested_environments():
    text = r"""\begin{proof}
We establish the theorem in two steps.
\begin{claim}
Every component $C$ is bipartite.
\end{claim}
\begin{proof}
Since $C$ is a subgraph of bipartite $G$, it has no odd cycles.
\end{proof}
By the claim, all components are 2-colorable. This completes the proof.
\end{proof}"""
    extracted = extract_latex_proof(text)
    assert extracted is not None
    assert extracted.startswith(r"\begin{proof}")
    assert extracted.endswith(r"\end{proof}")
    assert "Every component $C$ is bipartite" in extracted


def test_sanitize_informal_proof():
    text = r"""\begin{proof}
Wait, let $x \in S$. Actually, $f(x) \ge 0$.
\end{proof}"""
    sanitized = sanitize_informal_proof(text)
    assert sanitized.startswith(r"\begin{proof}")
    assert sanitized.endswith(r"\end{proof}")
    assert "Wait," not in sanitized
    assert "Observe that" in sanitized
    assert "Note that" in sanitized


def test_tcs_compliant_kernel_forbidden_imports():
    kernel = TCSCompliantKernel()
    kernel.reset()

    # Math imports should succeed
    res = kernel.execute("import sympy as sp; x = sp.Symbol('x'); print(sp.factor(x**2 - 1))")
    assert res.success
    assert "(x - 1)*(x + 1)" in res.stdout

    # Network / socket imports must be blocked per TCS protocol
    res_sock = kernel.execute("import socket")
    assert not res_sock.success
    assert "[TCS-BENCH PROTOCOL]" in res_sock.error

    res_req = kernel.execute("import requests")
    assert not res_req.success
    assert "[TCS-BENCH PROTOCOL]" in res_req.error


def test_tcs_bench_solver_empty_prompt():
    mock_client = MagicMock()
    mock_kernel = TCSCompliantKernel()
    mock_memory = MagicMock()
    mock_memory.check_health.return_value = True

    solver = MathProver(
        client=mock_client,
        kernel=mock_kernel,
        memory=mock_memory,
        default_max_iterations=3,
    )

    result = solver.solve("test_001", "")
    assert not result.verified
    assert result.error == "Empty task prompt in dataset"
    assert result.iterations == 0
