"""Prime Math Judging Package.

Provides formal compiler judges and multi-turn LLM judges for mathematical proofs:
- Lean 4 Formal Compiler Judge (`lean_judge`): Machine-checked, programmatic verification with zero tolerance for `sorry`/`admit`.
- TCS-Bench Judges (`tcs_judge`): Peer-review and binary consensus LLM evaluators replicating the official arXiv:2608.09538 protocol.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from core_engine.llm_client import TunnelThinkingClient

from .lean_judge import (
    LeanJudgeRecord,
    extract_lean_code,
    judge_lean_proof,
)
from .tcs_judge import (
    A1_INSTRUCTIONS,
    B_PARSE_ERROR,
    B_PARSE_OK,
    B_PARSE_RECOVERED,
    TARGET_RE,
    TCSJudgeRecordA,
    TCSJudgeRecordB,
    aggregate_b_replicates,
    build_a_prompt,
    build_b_prompt,
    call_judge,
    clamp01,
    extract_target,
    judge_b_once,
    judge_tcs_proof,
    parse_a_vote,
    parse_b_json,
    parse_b_validity,
)


def judge_proof(
    candidate: str,
    problem_prompt: Optional[str] = None,
    gt_solution: Optional[str] = None,
    mode: str = "auto",
    client: Optional[TunnelThinkingClient] = None,
    timeout_seconds: float = 60.0,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Universal mathematical proof judge.

    Args:
        candidate: Candidate proof string (LaTeX, Lean 4 code, or mixed markdown).
        problem_prompt: Problem statement / context.
        gt_solution: Ground-truth reference proof (required for TCS judging).
        mode: "auto" (detects Lean code vs LaTeX), "lean", "tcs", or "all".
        client: TunnelThinkingClient instance (for TCS LLM calls).
        timeout_seconds: Execution timeout.
        **kwargs: Extra parameters passed to the underlying judges.

    Returns:
        Structured evaluation verdict dictionary.
    """
    if mode == "auto":
        lean_code = extract_lean_code(candidate)
        mode = "lean" if lean_code else "tcs"

    results: Dict[str, Any] = {"mode": mode, "status": "judged"}

    if mode in ("lean", "all"):
        lean_res = judge_lean_proof(
            candidate,
            problem_statement=problem_prompt,
            timeout=timeout_seconds,
            lean_bin=kwargs.get("lean_bin"),
        )
        results["lean"] = lean_res
        if mode == "lean":
            results["verdict"] = lean_res["verdict"]
            results["passed"] = lean_res["passed"]
            results["status"] = lean_res["status"]
            return results

    if mode in ("tcs", "all"):
        if not gt_solution:
            results["tcs"] = {"error": "Ground-truth reference proof required for TCS judging"}
            results["verdict"] = "0"
            results["passed"] = False
        else:
            tcs_res = judge_tcs_proof(
                candidate=candidate,
                problem_prompt=problem_prompt or "",
                gt_solution=gt_solution,
                client=client,
                config=kwargs.get("config", "b"),
                timeout_seconds=timeout_seconds,
                threshold=kwargs.get("threshold"),
                **{k: kwargs[k] for k in ("replicates", "adaptive", "thinking", "max_tokens") if k in kwargs},
            )
            results["tcs"] = tcs_res
            if tcs_res.get("status") == "judge_parse_error":
                results["status"] = "judge_parse_error"
            if mode == "tcs":
                results["verdict"] = tcs_res["verdict"]
                results["passed"] = (tcs_res["verdict"] == "1")
                return results

    if mode == "all":
        lean_pass = results.get("lean", {}).get("passed", False)
        tcs_pass = (results.get("tcs", {}).get("verdict") == "1")
        results["passed"] = (lean_pass or tcs_pass)
        results["verdict"] = "1" if results["passed"] else "0"

    return results


__all__ = [
    # Lean 4 Judge
    "LeanJudgeRecord",
    "extract_lean_code",
    "judge_lean_proof",
    # TCS Judges
    "A1_INSTRUCTIONS",
    "B_PARSE_ERROR",
    "B_PARSE_OK",
    "B_PARSE_RECOVERED",
    "TARGET_RE",
    "TCSJudgeRecordA",
    "TCSJudgeRecordB",
    "aggregate_b_replicates",
    "build_a_prompt",
    "build_b_prompt",
    "call_judge",
    "clamp01",
    "extract_target",
    "judge_b_once",
    "judge_tcs_proof",
    "parse_a_vote",
    "parse_b_json",
    "parse_b_validity",
    # Universal Entrypoint
    "judge_proof",
]
