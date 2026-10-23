"""Theoretical Computer Science (TCS-Bench) LLM Proof Judges.

Implements the official evaluation protocols for TCS-Bench (arXiv:2608.09538):
1. Config A (A.1 Verifier): Consensus majority voting across 4 independent zero-shot calls (>= 3/4 agreement required to pass).
2. Config B (Adversarial Critic): Line-by-line defect hunting with continuous scoring (rigor, completeness, boundary_cases, validity probability).
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from statistics import mean, median, pstdev
from typing import Any, Dict, List, Optional, Tuple

from core_engine.llm_client import LLMResponse, TunnelThinkingClient
from core_engine.tcs_config import TCS_BORDERLINE_BAND, TCS_JUDGE_REPLICATES

logger = logging.getLogger("core_engine.judging.tcs_judge")

# Canonical target extractor for TCS-Bench prompts (scripts import this; do not re-define it).
TARGET_RE = re.compile(r"Target Problem \(your task\):\s*(.+)$", re.S)

# ---------------------------------------------------------------------------
# Config A: TCS-Bench v1 Appendix A.1 verifier prompt, VERBATIM.
# ---------------------------------------------------------------------------
A1_INSTRUCTIONS = """# Instructions
Your task is to evaluate whether a provided 'student_answer' correctly and rigorously proves the 'Target Problem' using only the definitions and lemmas found in the 'solve_promp'. You must compare the student's logic, variable tracking, and quantitative derivations against the 'ground_truth_proof'.

## Evaluation Criteria
1. **Logical Rigor and External Assumption Check:**
    *   **No Unauthorized Constraints:** Award a 0 if the student introduces arbitrary constraints, "safety margins," or specific examples that are not explicitly provided in the 'solve_prompt'.
    *   **Strict Lemma Adherence:** The student must use only the lemmas provided in the 'solve_prompt'. Applying a theorem without verifying its preconditions is a failure.
    *   **Case Integrity:** The student must address the same case splits and boundary conditions as the ground-truth proof.
2. **Variable Alignment and Property Scope:**
    *   **Property Scope Integrity:** Award a 0 if the student generalizes a property established for a subset to a larger set without proof.
    *   **Index and Set Precision:** Ensure all variable indices and set memberships match the 'ground_truth_proof'.
3. **Semantic Integrity and Jargon Detection:**
    *   **No Hallucinated Logic:** Reject answers containing "word salad" or circular reasoning.
    *   **No "Standard" Hand-waving:** Award a 0 if the student claims a non-trivial step is "standard" or "obvious" without justification.
4. **Quantitative and Limit Accuracy:**
    *   **Derivation Accuracy:** Award a 0 if there is any error in arithmetic, algebra, or inequality direction.
    *   **Boundary and Floor/Ceiling Precision:** Check for off-by-one errors in any floor/ceiling or boundary decisions.

## Output Format
* If the 'student_answer' is a valid, rigorous proof of the 'Target Problem', output exactly: 1
* If the 'student_answer' contains gaps, jargon, unauthorized assumptions, scope errors, or calculation errors, output exactly: 0

Do not provide any explanation, feedback, or additional text. Your response must be only '1' or '0'."""


def extract_target(prompt: str) -> str:
    """Extracts the target statement out of the self-contained task prompt."""
    m = TARGET_RE.search(prompt)
    if m and m.group(1).strip():
        return m.group(1).strip()
    return prompt[-4000:].strip()


def build_a_prompt(solve_prompt: str, target: str, gt: str, student: str) -> str:
    """Builds the prompt for Config A (Appendix A.1 verifier)."""
    return (
        A1_INSTRUCTIONS
        + "\n\n# Inputs\n\n"
        + f"'solve_prompt':\n{solve_prompt}\n\n"
        + f"'Target Problem':\n{target}\n\n"
        + f"'ground_truth_proof':\n{gt}\n\n"
        + f"'student_answer':\n{student}\n"
    )


def build_b_prompt(problem: str, gt: str, candidate: str) -> str:
    """Builds the prompt for the TCS Judge (rigorous defect critic + continuous validity scoring)."""
    return f"""You are an expert mathematical referee performing a rigorous peer review of a proof candidate for the TCS-Bench benchmark.

Your job: assess whether the candidate proof establishes the Target Problem with full mathematical correctness and completeness, auditing it line by line:
1. RECOMPUTE every arithmetic and algebraic claim independently.
2. Verify the direction of every inequality and the correctness of every bound, parameter setting, floor/ceiling, and boundary check.
3. Flag any unproven "standard"/"obvious"/"well-known" step that is actually non-trivial, and any false or unjustified premise.
4. Check completeness: does the proof establish all claims required by the target statement?
5. Different notation, an alternative reduction, or a different proof path than the reference proof is FULLY VALID if mathematically sound. Never dock a proof simply because it differs in technique from the reference. The reference proof is a guide for what must be established, not the only acceptable form.
6. The candidate solver only has access to the problem statement and context provided in the Target Problem prompt. Do NOT penalize a candidate for omitting external lemmas from the paper that were not provided in the prompt context, as long as the proof's own derivations are mathematically self-contained and sound.

If every step of the candidate proof is mathematically sound and complete, say so -- do NOT invent defects to justify a low score.

Target Problem:
{problem}

Reference proof (for context and requirements):
{gt}

Candidate proof (under evaluation):
{candidate}

Respond with ONLY a JSON object (no markdown fences, no commentary) of the form:
{{"rigor": <0-1>, "completeness": <0-1>, "boundary_cases": <0-1>, "confidence": <0-1>, "validity": <0-1>, "issues": "<one sentence: the most serious defect found, or 'none'>"}}
where each score is a continuous value from 0 to 1, and "validity" is a calibrated referee score on [0.0, 1.0] (with 0.50 as the pass threshold) assigned according to this scale:
- 0.80 to 1.00 (Strong Pass): Complete, rigorous proof establishing all required claims.
- 0.50 to 0.79 (Pass): Mathematically valid proof whose core mechanism establishes the target theorem (may have minor exposition brevity, routine algebraic steps omitted, standard asymptotic small-n absorption, or invoke context-stated guarantees without re-proving them from scratch, with no false claims or unclosed logical gaps).
- 0.35 to 0.49 (Near-Miss / Salvageable): Core inductive step or main reduction is mathematically sound across the primary parameter regime, but has an isolated boundary/degenerate base-case omission or a localized constant-factor slip.
- 0.15 to 0.34 (Partial Progress / Fail): Useful partial derivations or special cases, but a central technical step is incomplete.
- 0.00 to 0.14 (Fail): Fundamentally flawed, circular, vacuous, or leaves the main technical obstacle unresolved."""


def parse_a_vote(content: str) -> Optional[str]:
    """Finds the first standalone 0/1 token in the A response ('1' or '0'), else None."""
    m = re.search(r"\b([01])\b", content)
    return m.group(1) if m else None


B_PARSE_OK = "ok"
B_PARSE_RECOVERED = "recovered"  # validity recovered by regex from malformed / truncated JSON
B_PARSE_ERROR = "parse_error"    # no numeric validity could be recovered (NOT the same as validity 0)

_NUM = r"(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
# A number must be followed by a delimiter; a number at the very end of a truncated reply
# (e.g. '"validity": 0.' cut from '0.85') is NOT trusted.
_NUM_END = r"(?=[\s,}\]\"])"


def _regex_field(text: str, name: str) -> Optional[float]:
    """Last numeric value of `"name": <num>` (optionally quoted) in text, else None."""
    matches = re.findall(rf'"{name}"\s*:\s*"?\s*{_NUM}{_NUM_END}', text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def parse_b_json(content: str) -> Optional[Dict[str, Any]]:
    """Extracts and JSON-decodes the Config B response object.

    Returns a dict (with an added ``"parse_status"`` key) or None when nothing usable exists:
      - ``"ok"``: a JSON object was decoded and it carries a numeric ``validity``.
      - ``"recovered"``: JSON was malformed/truncated, but ``"validity": <num>`` was recovered by regex.
      - ``"parse_error"``: a JSON object was decoded but ``validity`` is missing/null/non-numeric.
    Use :func:`parse_b_validity` to get ``(validity, status)`` directly.
    """
    text = (content or "").strip()
    fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    decoded: Optional[Dict[str, Any]] = None
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                decoded = obj
        except (json.JSONDecodeError, ValueError):
            decoded = None

    if decoded is not None and clamp01(decoded.get("validity")) is not None:
        decoded["parse_status"] = B_PARSE_OK
        return decoded

    # Fallback (LaTeX backslash escapes inside JSON strings, or a reply truncated mid-object):
    # recover the numeric fields by regex, searching the raw content too in case a fence was unclosed.
    for source in (text, (content or "")):
        validity = _regex_field(source, "validity")
        if validity is None:
            continue
        fallback_dict: Dict[str, Any] = {"validity": validity}
        for fld in ("rigor", "completeness", "boundary_cases", "confidence"):
            val = _regex_field(source, fld)
            if val is not None:
                fallback_dict[fld] = val
        m_iss = re.search(r'"issues"\s*:\s*"(.*?)"\s*\}', source, flags=re.DOTALL)
        if not m_iss:
            m_iss = re.search(r'"issues"\s*:\s*"(.*)', source, flags=re.DOTALL)  # truncated string
        fallback_dict["issues"] = m_iss.group(1) if m_iss else ""
        fallback_dict["parse_status"] = B_PARSE_RECOVERED
        return fallback_dict

    if decoded is not None:
        # Valid JSON but validity missing / null / non-numeric: a parse failure, not a score of 0.
        decoded["parse_status"] = B_PARSE_ERROR
        return decoded
    return None


def parse_b_validity(content: str) -> Tuple[Optional[float], str, Optional[Dict[str, Any]]]:
    """Returns ``(validity_or_None, status, parsed_dict_or_None)``.

    ``status`` is ``"ok"``, ``"recovered"`` or ``"parse_error"``; validity is None iff status is
    ``"parse_error"``. Never raises on ``"validity": null``.
    """
    parsed = parse_b_json(content)
    if not isinstance(parsed, dict):
        return None, B_PARSE_ERROR, None
    validity = clamp01(parsed.get("validity"))
    if validity is None:
        return None, B_PARSE_ERROR, parsed
    return validity, parsed.get("parse_status", B_PARSE_OK), parsed


def clamp01(x: Any) -> Optional[float]:
    """Clamps a numeric value between 0.0 and 1.0, or None if invalid."""
    if isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    return max(0.0, min(1.0, v))


def call_judge(
    client: TunnelThinkingClient,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout_seconds: float,
    thinking: Optional[bool] = None,
    thinking_budget_tokens: Optional[int] = None,
) -> LLMResponse:
    """Helper executing an LLM completion call for judging."""
    if thinking and thinking_budget_tokens is None and max_tokens > 4096:
        from core_engine.tcs_config import load_tcs_config
        tcs_cfg = load_tcs_config()
        cfg_budget = int(getattr(tcs_cfg, "judge_thinking_budget", 12288) or 12288)
        thinking_budget_tokens = min(cfg_budget, max_tokens - 4096)
    return client.complete(
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        enable_thinking=thinking,
        thinking_budget_tokens=thinking_budget_tokens,
    )


@dataclass
class TCSJudgeRecordA:
    """Results for Config A (Appendix A.1 4-call consensus)."""
    raw: List[str] = field(default_factory=list)
    votes: List[Optional[str]] = field(default_factory=list)
    times: List[float] = field(default_factory=list)
    parseable: int = 0
    ones: int = 0
    verdict: str = "INCONCLUSIVE"  # "1", "0", "INCONCLUSIVE", or "NO_PROOF"


@dataclass
class TCSJudgeRecordB:
    """Results for Config B (continuous peer-review audit)."""
    raw: str = ""
    validity: Optional[float] = None
    verdict: Optional[str] = None  # "1" or "0"
    scores: Dict[str, Any] = field(default_factory=dict)
    time_s: float = 0.0


def _resolve_threshold(threshold: Optional[float]) -> float:
    if threshold is not None:
        return float(threshold)
    from core_engine import tcs_config
    return float(tcs_config.TCS_PASS_THRESHOLD)


def judge_b_once(
    client: TunnelThinkingClient,
    prompt_b: str,
    temperature: float,
    max_tokens: int,
    timeout_seconds: float,
    thinking: Optional[bool],
    retry_max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """One Config-B referee replicate, with a single repair retry.

    If the reply cannot be parsed, or the server reports ``finish_reason == "length"``, the call
    is retried once with a larger ``max_tokens`` and thinking DISABLED (the JSON verdict is short;
    truncation almost always comes from the reasoning trace eating the budget).

    Returns a dict with ``validity`` (None on parse failure), ``status`` ("ok" / "recovered" /
    "parse_error"), ``finish_reason``, ``retried``, ``raw``, ``parsed`` and ``time_s``.
    """
    t0 = time.perf_counter()
    resp = call_judge(
        client,
        [{"role": "user", "content": prompt_b}],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        thinking=thinking,
    )
    content = getattr(resp, "content", "") or ""
    finish_reason = getattr(resp, "finish_reason", None)
    finish_reason = finish_reason if isinstance(finish_reason, str) else None
    validity, status, parsed = parse_b_validity(content)
    rec: Dict[str, Any] = {
        "raw": content,
        "validity": validity,
        "status": status,
        "finish_reason": finish_reason,
        "retried": False,
        "parsed": parsed,
    }

    if status == B_PARSE_ERROR or finish_reason == "length":
        retry_tokens = retry_max_tokens if retry_max_tokens is not None else max_tokens + 4096
        logger.warning(
            f"TCS Judge replicate {'unparseable' if status == B_PARSE_ERROR else 'truncated (length)'}; "
            f"retrying once with max_tokens={retry_tokens}, thinking disabled."
        )
        rec["retried"] = True
        rec["first_attempt"] = {"raw": content, "status": status, "finish_reason": finish_reason}
        try:
            resp2 = call_judge(
                client,
                [{"role": "user", "content": prompt_b}],
                temperature=temperature,
                max_tokens=retry_tokens,
                timeout_seconds=timeout_seconds,
                thinking=False,
            )
            content2 = getattr(resp2, "content", "") or ""
            fr2 = getattr(resp2, "finish_reason", None)
            fr2 = fr2 if isinstance(fr2, str) else None
            v2, s2, p2 = parse_b_validity(content2)
            if s2 != B_PARSE_ERROR or status == B_PARSE_ERROR:
                rec.update({"raw": content2, "validity": v2, "status": s2, "finish_reason": fr2, "parsed": p2})
        except Exception as exc:  # keep the first attempt's result; record why the repair failed
            rec["retry_error"] = f"{type(exc).__name__}: {exc}"

    rec["time_s"] = round(time.perf_counter() - t0, 2)
    return rec


def aggregate_b_replicates(
    replicates: List[Dict[str, Any]], threshold: float
) -> Dict[str, Any]:
    """Aggregates Config-B replicates: median validity, mean, spread, pass probability.

    Parse failures are EXCLUDED from the statistics (they are not scores of 0). If no replicate
    produced a validity, ``validity`` is None and ``status`` is ``"judge_parse_error"``.
    """
    vals = [r["validity"] for r in replicates if r.get("validity") is not None]
    n_err = sum(1 for r in replicates if r.get("validity") is None)
    agg: Dict[str, Any] = {
        "replicate_validities": vals,
        "replicate_statuses": [r.get("status") for r in replicates],
        "replicate_finish_reasons": [r.get("finish_reason") for r in replicates],
        "n_replicates": len(replicates),
        "n_parse_errors": n_err,
        "threshold": threshold,
    }
    if not vals:
        agg.update({
            "validity": None,
            "validity_mean": None,
            "validity_spread": None,
            "validity_stdev": None,
            "pass_probability": None,
            "verdict": None,
            "status": "judge_parse_error",
        })
        return agg
    med = round(float(median(vals)), 4)
    agg.update({
        "validity": med,
        "validity_mean": round(float(mean(vals)), 4),
        "validity_spread": round(float(max(vals) - min(vals)), 4),
        "validity_stdev": round(float(pstdev(vals)), 4) if len(vals) > 1 else 0.0,
        "pass_probability": round(sum(1 for v in vals if v >= threshold) / len(vals), 4),
        "verdict": "1" if med >= threshold else "0",
        "status": "judged",
    })
    return agg


def judge_tcs_proof(
    candidate: str,
    problem_prompt: str,
    gt_solution: str,
    client: Optional[TunnelThinkingClient] = None,
    config: str = "tcs",
    temp_a: float = 0.3,
    temp_b: float = 0.1,
    threshold: Optional[float] = None,
    timeout_seconds: float = 120.0,
    thinking: bool = True,
    max_tokens: int = 16384,
    replicates: int = TCS_JUDGE_REPLICATES,
    borderline_replicates: bool = True,
    adaptive: bool = False,
    borderline_band: float = TCS_BORDERLINE_BAND,
    retry_max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Primary TCS-Bench Proof Judge.

    Evaluates candidate proofs against academic journal referee standards with deep reasoning,
    line-by-line defect hunting, and continuous scoring (rigor, completeness, boundary_cases, validity).

    Args:
        candidate: The student or solver candidate proof text.
        problem_prompt: The task problem prompt (including context).
        gt_solution: Ground truth reference proof.
        client: TunnelThinkingClient instance (created if None).
        config: "tcs" / "b" (default primary judge), "ab" (dual), or "a" (legacy Appendix A.1 consensus).
        temp_a: Sampling temperature for legacy Config A calls (default 0.3).
        temp_b: Sampling temperature for main referee call (default 0.1).
        threshold: Pass threshold on validity (default: ``tcs_config.TCS_PASS_THRESHOLD``).
        timeout_seconds: Timeout per LLM call.
        thinking: Whether thinking is enabled (default True for rigorous verification).
        max_tokens: Max output tokens.
        replicates: Number k of independent referee calls (default 3). The reported validity is
            the median of the parseable replicates; parse failures are excluded, not scored 0.
        borderline_replicates: Legacy switch, only used when ``replicates == 1``: if the single call is
            unparseable or lands in the borderline band, escalate to 3 replicates.
        adaptive: If True, run one call first and stop if it parsed and lies OUTSIDE the borderline band
            ``threshold +/- borderline_band``; otherwise continue up to ``replicates`` calls.
        borderline_band: Half-width of the borderline band centred on the threshold (default 0.20).
        retry_max_tokens: max_tokens for the one repair retry on parse failure / length truncation
            (default ``max_tokens + 4096``; that retry always runs with thinking disabled).

    Returns:
        Structured evaluation dict with "status" ("judged" / "no_proof" / "judge_parse_error"),
        "verdict", "validity" (median, or None on parse failure), "pass_probability",
        "replicate_validities", "scores", and component results under "a" / "b".
    """
    threshold = _resolve_threshold(threshold)
    if client is None:
        client = TunnelThinkingClient()
    from core_engine.tcs_config import load_tcs_config
    tcs_cfg = load_tcs_config()
    if max_tokens == 16384 and getattr(tcs_cfg, "judge_max_tokens", 16384) != 16384:
        max_tokens = int(tcs_cfg.judge_max_tokens)
    if timeout_seconds == 120.0 and getattr(tcs_cfg, "judge_timeout_seconds", 360.0) > 120.0:
        timeout_seconds = float(tcs_cfg.judge_timeout_seconds)

    out: Dict[str, Any] = {
        "candidate_chars": len(candidate),
        "status": "judged",
        "a": None,
        "b": None,
        "verdict": "0",
        "threshold": threshold,
    }

    if not candidate.strip():
        out["status"] = "no_proof"
        out["a"] = {"verdict": "NO_PROOF"}
        out["verdict"] = "0"
        return out

    target = extract_target(problem_prompt)

    if config in ("a", "ab"):
        prompt_a = build_a_prompt(problem_prompt, target, gt_solution, candidate)
        raw_a: List[str] = []
        votes_a: List[Optional[str]] = []
        times_a: List[float] = []
        for _ in range(4):
            t0 = time.perf_counter()
            resp = call_judge(
                client,
                [{"role": "user", "content": prompt_a}],
                temperature=temp_a,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                thinking=thinking,
            )
            times_a.append(round(time.perf_counter() - t0, 2))
            raw_a.append(resp.content)
            votes_a.append(parse_a_vote(resp.content))

        parseable = sum(1 for v in votes_a if v is not None)
        ones = sum(1 for v in votes_a if v == "1")
        verdict_a = "INCONCLUSIVE" if parseable < 3 else ("1" if ones >= 3 else "0")
        out["a"] = {
            "raw": raw_a,
            "votes": votes_a,
            "times": times_a,
            "parseable": parseable,
            "ones": ones,
            "verdict": verdict_a,
        }

    if config in ("tcs", "b", "ab"):
        prompt_b = build_b_prompt(problem_prompt, gt_solution, candidate)
        t0 = time.perf_counter()

        k = max(1, int(replicates))
        is_adaptive = bool(adaptive)
        if k == 1 and borderline_replicates:
            is_adaptive, k = True, 3  # legacy: escalate a borderline/unparseable single call to 3
        lo, hi = threshold - borderline_band, threshold + borderline_band

        reps: List[Dict[str, Any]] = []
        for i in range(k):
            rep = judge_b_once(client, prompt_b, temp_b, max_tokens, timeout_seconds, thinking,
                               retry_max_tokens=retry_max_tokens)
            reps.append(rep)
            if is_adaptive and i == 0:
                v0 = rep.get("validity")
                if v0 is not None and not (lo <= v0 <= hi):
                    break  # decisive first call: outside the borderline band

        agg = aggregate_b_replicates(reps, threshold)

        # Representative replicate (closest to the median) supplies raw text, sub-scores and issues.
        rep_idx = 0
        if agg["validity"] is not None:
            rep_idx = min(
                (i for i, r in enumerate(reps) if r.get("validity") is not None),
                key=lambda i: abs(reps[i]["validity"] - agg["validity"]),
            )
        rep_parsed = reps[rep_idx].get("parsed")
        scores_b: Dict[str, Any] = {}
        if isinstance(rep_parsed, dict):
            for key in ("rigor", "completeness", "boundary_cases", "confidence"):
                scores_b[key] = clamp01(rep_parsed.get(key))
            scores_b["issues"] = str(rep_parsed.get("issues", ""))[:500]

        out["b"] = {
            "raw": reps[rep_idx]["raw"],
            "raw_replicates": [r["raw"] for r in reps],
            "validity": agg["validity"],
            "replicate_validities": agg["replicate_validities"],
            "replicate_statuses": agg["replicate_statuses"],
            "replicate_finish_reasons": agg["replicate_finish_reasons"],
            "replicate_retried": [r.get("retried", False) for r in reps],
            "n_replicates": agg["n_replicates"],
            "n_parse_errors": agg["n_parse_errors"],
            "validity_mean": agg["validity_mean"],
            "validity_spread": agg["validity_spread"],
            "validity_stdev": agg["validity_stdev"],
            "pass_probability": agg["pass_probability"],
            "adaptive": is_adaptive,
            "borderline_band": [round(lo, 4), round(hi, 4)],
            "verdict": agg["verdict"],
            "status": agg["status"],
            "scores": scores_b,
            "time_s": round(time.perf_counter() - t0, 2),
        }
        for key in ("validity", "pass_probability", "replicate_validities", "validity_mean", "validity_spread"):
            out[key] = out["b"][key]
        out["scores"] = scores_b
        out["issues"] = scores_b.get("issues", "")
        if agg["validity"] is None:
            out["status"] = "judge_parse_error"

    # Derive overall verdict based on configured modes
    if config in ("tcs", "b"):
        out["verdict"] = (out["b"]["verdict"] or "0") if out["b"] else "0"
    elif config == "a":
        out["verdict"] = out["a"]["verdict"] if out["a"] else "0"
    else:  # "ab"
        # Secondary metric only: passes if either Config A passes (>=3/4) or Config B passes.
        passed_a = (out["a"]["verdict"] == "1") if out["a"] else False
        passed_b = (out["b"]["verdict"] == "1") if out["b"] else False
        out["verdict"] = "1" if (passed_a or passed_b) else "0"

    return out
