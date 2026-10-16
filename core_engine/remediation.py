"""Automated 3-Tier Post-Evaluation Remediation Engine for Prime Math.

Diagnoses evaluation failures across three distinct architectural tiers:
- Tier 1 (Harness & Code Level): Python kernel timeouts, stdout capture, regex extraction, memory leaks.
- Tier 2 (Orchestration Prompt & Invariants): Universal cognitive invariants in MATH_SYSTEM_PROMPT,
  governed by a strict 4-gate Universality Rubric (rejects non-universal or domain-specific rules to Tier 3).
- Tier 3 (Procedural Memory / TurboVec): Domain-specific mathematical heuristics, algorithmic recipes,
  asymptotic extrapolation tricks, and boundary-condition patterns committed to TurboVec.
"""

from __future__ import annotations

import ast
import datetime
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Re-exported for backward compatibility: the validator now lives in heuristic_validation
# (stdlib-only) to break the memory <-> remediation import cycle.
from .heuristic_validation import (  # noqa: F401
    PROVENANCE_VERSION,
    find_ground_truth_leak,
    is_valid_universal_heuristic,
    redact_ground_truth,
)
from .llm_client import TunnelThinkingClient
from .memory import CoALAMemoryManager, TurboVecUnavailableError

logger = logging.getLogger("core_engine.remediation")
console = Console()

CORE_ENGINE_PKG_DIR = Path(__file__).parent.resolve()
REPO_ROOT_DIR = CORE_ENGINE_PKG_DIR.parent.resolve()
REMEDIATION_DIR = REPO_ROOT_DIR / "data" / "remediations"
DEFAULT_EVAL_DIR = REPO_ROOT_DIR / "data" / "eval_results"
PROBLEMS_LOG_DIR = REPO_ROOT_DIR / "logs" / "problems"
DATA_DIR = REPO_ROOT_DIR / "data"
PARKED_ISSUES_FILE = REMEDIATION_DIR / "parked_tier1_issues.jsonl"

DEFAULT_MAX_TIER1_LINES_TARGET: int = 10
DEFAULT_MAX_TIER1_LINES_REPLACEMENT: int = 15


class RemediationTier(str, Enum):
    TIER_1_HARNESS = "TIER_1_HARNESS"
    TIER_2_ORCHESTRATION = "TIER_2_ORCHESTRATION"
    TIER_3_PROCEDURAL = "TIER_3_PROCEDURAL"


@dataclass
class Tier2Rubric:
    """Strict 4-Gate Universality Rubric for proposing Tier 2 Orchestration Invariants.

    All four gates must evaluate to True for a rule to be admitted to MATH_SYSTEM_PROMPT.
    If ANY gate fails, the rule is definitively routed to Tier 3 Procedural Memory.
    """
    is_universal: bool  # 1. 100% domain-agnostic across all math fields (not domain-specific)
    is_behavioral_posture: bool  # 2. Governs cognitive posture/lifecycle, not an algorithm/trick
    no_degradation_risk: bool  # 3. Will not bias, anchor, or degrade unrelated problem types
    procedural_insufficient: bool  # 4. Dynamic just-in-time retrieval is strictly insufficient
    rationale: str
    verdict: str  # "PASS" or "REJECT_TO_TIER3"


@dataclass
class ProblemDiagnostic:
    problem_id: int
    title: str
    root_cause: str
    target_tier: str
    tier2_rubric: Optional[Tier2Rubric] = None
    tier1_patch: Optional[Dict[str, str]] = None  # file, issue, suggested_fix
    tier2_rule: Optional[str] = None
    tier3_heuristic: Optional[str] = None
    rerouted_to_tier3: bool = False
    persisted_to_turbovec: bool = False
    tier1_patch_applied: bool = False
    persisted_status: Optional[str] = None
    is_parked: bool = False
    parked_reason: Optional[str] = None
    tier3_rejected_reason: Optional[str] = None


@dataclass
class RemediationReport:
    timestamp: str
    total_analyzed: int
    tier1_count: int
    tier2_count: int
    tier3_count: int
    tier2_rerouted_count: int
    heuristics_persisted: int
    tier1_patched_count: int = 0
    tier1_parked_count: int = 0
    tier3_rejected_count: int = 0
    diagnostics: List[ProblemDiagnostic] = field(default_factory=list)


def extract_json_payload(text: str) -> Optional[Dict[str, Any]]:
    """Extracts valid JSON payload from model response, handling code fences and think tags."""
    if not text:
        return None

    # Strip thinking tags
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Look for ```json ... ``` blocks
    m_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if m_block:
        try:
            return json.loads(m_block.group(1))
        except json.JSONDecodeError:
            pass

    # Look for outermost { ... }
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(cleaned[first_brace : last_brace + 1])
        except json.JSONDecodeError:
            pass

    return None


class RemediationEngine:
    """3-Tier Post-Evaluation Remediation Engine."""

    def __init__(
        self,
        client: Optional[TunnelThinkingClient] = None,
        memory: Optional[CoALAMemoryManager] = None,
        max_tier1_lines_target: int = DEFAULT_MAX_TIER1_LINES_TARGET,
        max_tier1_lines_replacement: int = DEFAULT_MAX_TIER1_LINES_REPLACEMENT,
        auto_apply_tier1: Optional[bool] = None,
    ):
        self.client = client or TunnelThinkingClient()
        self.memory = memory or CoALAMemoryManager()
        self.max_tier1_lines_target = max_tier1_lines_target
        self.max_tier1_lines_replacement = max_tier1_lines_replacement
        if auto_apply_tier1 is None:
            self.auto_apply_tier1 = os.getenv("PRIME_MATH_AUTO_APPLY_TIER1", "false").lower() in ("true", "1")
        else:
            self.auto_apply_tier1 = bool(auto_apply_tier1)
        self.euler_dataset, self.frontier_dataset = self._load_all_datasets()
        self.problems_dataset = {**self.frontier_dataset, **self.euler_dataset}
        REMEDIATION_DIR.mkdir(parents=True, exist_ok=True)

    def _log_parked_issue(
        self,
        diagnostic: ProblemDiagnostic,
        telemetry: Dict[str, Any],
        target_filename: str,
        issue: str,
        suggested_fix: str,
        reason: str,
        target_content: Optional[str] = None,
        replacement_content: Optional[str] = None,
        target_lines: int = 0,
        replacement_lines: int = 0,
    ) -> None:
        """Logs a complex Tier 1 issue to data/remediations/parked_tier1_issues.jsonl for paired AGY review."""
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "problem_id": diagnostic.problem_id,
            "title": diagnostic.title,
            "target_file": f"core_engine/{target_filename}",
            "issue": issue,
            "suggested_fix": suggested_fix,
            "parked_reason": reason,
            "complexity": {
                "target_lines": target_lines,
                "replacement_lines": replacement_lines,
                "max_allowed_target": self.max_tier1_lines_target,
                "max_allowed_replacement": self.max_tier1_lines_replacement,
            },
            "proposed_patch": {
                "target_content": target_content,
                "replacement_content": replacement_content,
            } if target_content and replacement_content else None,
            "last_error": telemetry.get("last_error", "None"),
        }
        try:
            with open(PARKED_ISSUES_FILE, "a", encoding="utf-8") as fp:
                fp.write(json.dumps(entry) + "\n")
            logger.info(f"Logged parked Tier 1 issue for Problem {diagnostic.problem_id} to {PARKED_ISSUES_FILE}")
        except Exception as exc:
            logger.warning(f"Failed to log parked issue: {exc}")

    def _load_all_datasets(self) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
        """Loads available problem datasets from data/ separated by domain."""
        euler_probs: Dict[int, Dict[str, Any]] = {}
        frontier_probs: Dict[int, Dict[str, Any]] = {}

        frontier_path = DATA_DIR / "frontiermath_samples.json"
        if frontier_path.exists():
            try:
                with open(frontier_path, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                for item in data:
                    pid = item.get("id")
                    if pid is not None:
                        frontier_probs[pid] = item
            except Exception as e:
                logger.debug(f"Could not load frontier dataset: {e}")

        euler_paths = [
            DATA_DIR / "euler" / "euler_800_1000.json",
            DATA_DIR / "euler" / "euler_200.json",
        ]
        for fpath in euler_paths:
            if fpath.exists():
                try:
                    with open(fpath, "r", encoding="utf-8") as fp:
                        data = json.load(fp)
                    for item in data:
                        pid = item.get("id")
                        if pid is not None and pid not in euler_probs:
                            euler_probs[pid] = item
                except Exception as e:
                    logger.debug(f"Could not load euler dataset {fpath}: {e}")

        return euler_probs, frontier_probs

    def find_failed_problems(
        self,
        latest_n_evals: int = 1,
        problem_ids: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Discovers failed problems from evaluation JSON logs or problem logs."""
        failed_map: Dict[Any, Dict[str, Any]] = {}

        # 1. If explicit problem IDs provided, gather them
        if problem_ids:
            for pid in problem_ids:
                item = self.problems_dataset.get(pid, {})
                failed_map[pid] = {
                    "problem_id": pid,
                    "title": item.get("title", f"Problem {pid}"),
                    "expected_answer": str(item.get("answer", "")).strip(),
                    "predicted_answer": None,
                    "failure_category": "MANUAL_INSPECTION",
                    "turns": 0,
                    "error": None,
                }
            return list(failed_map.values())

        # 2. Inspect latest evaluation files
        eval_files = sorted(
            DEFAULT_EVAL_DIR.glob("eval_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for eval_file in eval_files[:latest_n_evals]:
            try:
                with open(eval_file, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                for rec in data.get("records", []):
                    if not rec.get("is_correct"):
                        pid = rec.get("problem_id")
                        tier = rec.get("tier")
                        key = f"frontier_{tier}_{pid}" if tier is not None else pid
                        if pid and key not in failed_map:
                            failed_map[key] = {
                                "problem_id": pid,
                                "tier": tier,
                                "dataset": "frontier" if tier is not None else "euler",
                                "title": rec.get("title", f"Problem {pid}"),
                                "expected_answer": str(rec.get("expected_answer", "")).strip(),
                                "predicted_answer": rec.get("predicted_answer"),
                                "failure_category": rec.get("failure_category", "FAILED"),
                                "turns": rec.get("turns", 0),
                                "error": rec.get("error_message"),
                            }
            except Exception as e:
                logger.warning(f"Error reading eval file {eval_file}: {e}")

        # 3. If no eval files or no failures found in eval files, inspect problem logs
        if not failed_map and PROBLEMS_LOG_DIR.exists():
            for log_file in sorted(PROBLEMS_LOG_DIR.glob("problem_*.log")):
                try:
                    m = re.match(r"problem_(\d+)\.log", log_file.name)
                    if not m:
                        continue
                    pid = int(m.group(1))
                    item = self.problems_dataset.get(pid, {})
                    expected = str(item.get("answer", "")).strip()

                    txt = log_file.read_text(encoding="utf-8", errors="ignore")
                    m_ans = re.search(r"\[(?:SESSION|TIER 1) COMPLETE\] (?:Final |Verified )?Boxed Answer: (\S+)", txt)
                    cand = m_ans.group(1).replace("{", "").replace("}", "").strip() if m_ans else None

                    # If not verified or answers don't match
                    if not cand or (expected and cand.lower() != expected.lower()):
                        failed_map[pid] = {
                            "problem_id": pid,
                            "title": item.get("title", f"Problem {pid}"),
                            "expected_answer": expected,
                            "predicted_answer": cand,
                            "failure_category": "UNVERIFIED_OR_WRONG",
                            "turns": txt.count("--- [TURN "),
                            "error": None,
                        }
                except Exception:
                    continue

        return list(failed_map.values())

    def extract_failure_telemetry(self, fail_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts dense failure telemetry and log excerpts for diagnosis."""
        pid = fail_info["problem_id"]
        tier = fail_info.get("tier")
        is_frontier = tier is not None or fail_info.get("dataset") == "frontier"

        log_file = None
        if is_frontier:
            frontier_dir = Path(__file__).parent.parent / "logs" / "frontier"
            if tier is not None:
                cand = frontier_dir / f"tier{tier}_prob_{pid:02d}.log"
                if cand.exists():
                    log_file = cand
            if log_file is None and frontier_dir.exists():
                for f in frontier_dir.glob(f"*prob_{pid:02d}.log"):
                    log_file = f
                    break

        if log_file is None:
            log_file = PROBLEMS_LOG_DIR / f"problem_{pid:03d}.log"
            if not log_file.exists():
                log_file = PROBLEMS_LOG_DIR / f"problem_{pid}.log"

        log_excerpt = ""
        last_error = ""
        turns_count = fail_info.get("turns", 0)

        if log_file and log_file.exists():
            try:
                content = log_file.read_text(encoding="utf-8", errors="ignore")
                lines = content.splitlines()

                # Find last error or traceback
                tb_lines = []
                in_tb = False
                for line in lines[-120:]:
                    if "Traceback (most recent call last)" in line or "EXECUTION ERROR" in line:
                        in_tb = True
                    if in_tb:
                        tb_lines.append(line)
                        if len(tb_lines) > 30:
                            break
                if tb_lines:
                    last_error = "\n".join(tb_lines)

                # Keep last 60 lines as contextual execution excerpt
                log_excerpt = "\n".join(lines[-60:])
                turns_count = max(turns_count, content.count("--- [TURN "))
            except Exception as e:
                log_excerpt = f"[Failed to read log: {e}]"

        if is_frontier and pid in self.frontier_dataset:
            item = self.frontier_dataset[pid]
        else:
            item = self.euler_dataset.get(pid, self.problems_dataset.get(pid, {}))

        statement = (
            item.get("problem")
            or item.get("statement")
            or item.get("description")
            or fail_info.get("title", f"Problem {pid}")
        )

        return {
            "problem_id": pid,
            "tier": tier,
            "title": fail_info.get("title", item.get("title", f"Problem {pid}")),
            "statement": statement[:1500],
            "expected_answer": fail_info.get("expected_answer") or str(item.get("answer", "")),
            "predicted_answer": fail_info.get("predicted_answer"),
            "failure_category": fail_info.get("failure_category", "UNKNOWN"),
            "turns_count": turns_count,
            "last_error": last_error or fail_info.get("error") or "None",
            "log_excerpt": log_excerpt,
        }

    def diagnose_failure(self, telemetry: Dict[str, Any]) -> ProblemDiagnostic:
        """Invokes the thinking model to diagnose failure and apply the Tier 2 Rubric Gate."""
        pid = telemetry["problem_id"]
        exp_ans = telemetry.get("expected_answer")
        redacted_log = redact_ground_truth(telemetry.get("log_excerpt", ""), exp_ans)
        redacted_err = redact_ground_truth(telemetry.get("last_error", ""), exp_ans)

        prompt = f"""You are the Lead Remediation Architect for Prime Math, an autonomous mathematical problem-solving system.
A benchmark problem has FAILED during evaluation. Your mission is to analyze the failure, diagnose the root cause, and classify the remediation into the appropriate architectural tier.

PROBLEM DETAILS:
- ID: {pid} ({telemetry['title']})
- Evaluation Status: FAILED / INCORRECT ANSWER
- Predicted Answer: {telemetry['predicted_answer']}
- Failure Category: {telemetry['failure_category']}
- Turns Executed: {telemetry['turns_count']}

PROBLEM STATEMENT:
{telemetry['statement']}

EXECUTION LOG & TRACE EXCERPT:
{redacted_log}

LAST DETECTED ERROR:
{redacted_err}

================================================================================
REMEDIATION TIERS:
- TIER 1: Harness & Execution Engine Code
  Scope: Kernel timeouts, stdout/stderr capture, regex extraction, memory leaks, dependency/import bugs in core_engine/ (e.g. kernel.py, evaluator.py, solver.py).
  Use when: The code harness crashed, threw an unexpected Python exception, or failed to parse execution output.

- TIER 2: Orchestration Prompt & Universal Invariants (MATH_SYSTEM_PROMPT)
  Scope: Universal cognitive posture, reasoning-computation lifecycle, anti-anchor bias, turn budget discipline.
  MANDATORY UNIVERSALITY EVALUATION RUBRIC (ALL 4 GATES REQUIRED FOR PASS):
  1. Universality: Is this rule 100% domain-agnostic across ALL mathematics (number theory, algebra, geometry, combinatorics, analysis)?
  2. Behavioral Posture: Does this govern cognitive posture or lifecycle rather than a specific algorithm, formula, or trick?
  3. No Degradation Risk: Is there zero risk of biasing or confusing unrelated mathematical problems?
  4. Procedural Insufficient: Would dynamic just-in-time procedural retrieval from TurboVec be insufficient?
  NOTE: If ANY of these 4 gates evaluates to false, Tier 2 MUST BE REJECTED and the heuristic MUST be routed to Tier 3!

- TIER 3: Universal Procedural Heuristics (TurboVec 'prime_math_procedural')
  Scope: Genuine universal mathematical heuristics, theoretical reduction patterns, algebraic/analytic invariants, and proven structural simplification strategies.
  MANDATORY UNIVERSALITY & SOUNDNESS REQUIREMENTS:
  1. ONLY GENUINE UNIVERSAL MATHEMATICAL HEURISTICS ARE PERMITTED.
  2. STRICTLY FORBIDDEN IN TIER 3: Empirical sieving recipes, Richardson or Aitken extrapolation hacks, sample checkpointing, brute-force search advice, problem-specific variable bindings (e.g., ord_p(2) > ord_p(3)), and benchmark/problem number identifiers.
  3. If the failure was due to lack of domain insight or failed proof attempt and no genuine universal mathematical heuristic can be formulated, DO NOT generate a heuristic (set "tier3_heuristic": null).
================================================================================

MANDATORY OUTPUT FORMAT & CONSTRAINTS:
- Be concise and decisive. Limit internal reasoning to under 150 words and immediately output the final valid JSON.
- Do NOT generate markdown outside the JSON block.

RESPOND STRICTLY IN VALID JSON WITH THIS EXACT SCHEMA:
{{
  "root_cause": "Detailed analysis of why the problem failed",
  "target_tier": "TIER_1_HARNESS" | "TIER_2_ORCHESTRATION" | "TIER_3_PROCEDURAL",
  "tier1_patch": {{
    "file": "path/to/file.py",
    "issue": "Description of code bug",
    "suggested_fix": "Concrete Python fix or code snippet"
  }} or null,
  "tier2_rubric": {{
    "is_universal": true/false,
    "is_behavioral_posture": true/false,
    "no_degradation_risk": true/false,
    "procedural_insufficient": true/false,
    "rationale": "Evaluation against the 4 gates",
    "verdict": "PASS" | "REJECT_TO_TIER3"
  }} or null,
  "tier2_rule": "Proposed universal invariant text" or null,
  "tier3_heuristic": "Concise universal mathematical heuristic (<250 chars) for TurboVec, or null if no universal heuristic applies" or null
}}
"""

        try:
            resp = self.client.complete(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
                temperature=0.1,
                enable_thinking=False,
            )
            # Try parsing from content first, then fallback to reasoning if content is empty
            data = extract_json_payload(resp.content)
            if not data and resp.reasoning:
                data = extract_json_payload(resp.reasoning)

            if not data:
                logger.error(f"Failed to parse JSON diagnosis for Problem {pid}. Content: {resp.content[:200]}")
                return ProblemDiagnostic(
                    problem_id=pid,
                    title=telemetry["title"],
                    root_cause="Model diagnosis produced unparseable response",
                    target_tier=RemediationTier.TIER_3_PROCEDURAL.value,
                    tier3_heuristic=None,
                )

            root_cause = data.get("root_cause", "No root cause provided")
            target_tier = data.get("target_tier", RemediationTier.TIER_3_PROCEDURAL.value)
            t1_patch = data.get("tier1_patch")
            t2_data = data.get("tier2_rubric")
            t2_rule = data.get("tier2_rule")
            t3_heuristic = data.get("tier3_heuristic")

            rubric = None
            if t2_data:
                rubric = Tier2Rubric(
                    is_universal=bool(t2_data.get("is_universal")),
                    is_behavioral_posture=bool(t2_data.get("is_behavioral_posture")),
                    no_degradation_risk=bool(t2_data.get("no_degradation_risk")),
                    procedural_insufficient=bool(t2_data.get("procedural_insufficient")),
                    rationale=str(t2_data.get("rationale", "")),
                    verdict=str(t2_data.get("verdict", "REJECT_TO_TIER3")),
                )

            # --- PROGRAMMATIC TIER 2 RUBRIC GATE ENFORCEMENT ---
            rerouted = False
            if target_tier == RemediationTier.TIER_2_ORCHESTRATION.value:
                # Check if all 4 gates pass
                passes_rubric = (
                    rubric is not None
                    and rubric.is_universal
                    and rubric.is_behavioral_posture
                    and rubric.no_degradation_risk
                    and rubric.procedural_insufficient
                    and rubric.verdict == "PASS"
                )

                if not passes_rubric:
                    logger.info(
                        f"[Problem {pid}] Tier 2 proposed rule failed Universality Rubric. "
                        "Programmatically re-routing to Tier 3 Procedural Memory."
                    )
                    target_tier = RemediationTier.TIER_3_PROCEDURAL.value
                    rerouted = True
                    if not t3_heuristic and t2_rule:
                        t3_heuristic = t2_rule
                    t2_rule = None

            diagnostic = ProblemDiagnostic(
                problem_id=pid,
                title=telemetry["title"],
                root_cause=root_cause,
                target_tier=target_tier,
                tier2_rubric=rubric,
                tier1_patch=t1_patch,
                tier2_rule=t2_rule,
                tier3_heuristic=t3_heuristic,
                rerouted_to_tier3=rerouted,
            )

            # Preemptive parking if Tier 1 issue is explicitly diagnosed as complex/architectural
            if target_tier == RemediationTier.TIER_1_HARNESS.value and t1_patch:
                if str(t1_patch.get("complexity", "")).upper() == "COMPLEX":
                    diagnostic.is_parked = True
                    park_reason = f"Diagnosed as complex architectural issue ({t1_patch.get('issue', '')}). Parked for paired AGY review."
                    diagnostic.parked_reason = park_reason
                    diagnostic.persisted_status = f"TIER_1_PARKED_COMPLEX: {park_reason}"
                    self._log_parked_issue(
                        diagnostic=diagnostic,
                        telemetry=telemetry,
                        target_filename=Path(t1_patch.get("file", "unknown.py")).name,
                        issue=t1_patch.get("issue", ""),
                        suggested_fix=t1_patch.get("suggested_fix", ""),
                        reason=park_reason,
                    )

            return diagnostic

        except Exception as e:
            logger.error(f"Error during failure diagnosis for Problem {pid}: {e}")
            return ProblemDiagnostic(
                problem_id=pid,
                title=telemetry["title"],
                root_cause=f"Diagnosis exception: {e}",
                target_tier=RemediationTier.TIER_3_PROCEDURAL.value,
                tier3_heuristic=None,
            )

    def apply_tier1_patch(
        self,
        diagnostic: ProblemDiagnostic,
        telemetry: Dict[str, Any],
        dry_run: bool = False,
    ) -> Tuple[bool, str]:
        """Synthesizes, bounds-checks, applies, and tests a code patch strictly inside prime_math/."""
        if diagnostic.is_parked:
            return False, diagnostic.parked_reason or "Issue is parked for AGY operator review"

        if not diagnostic.tier1_patch:
            return False, "No tier1_patch metadata provided"

        raw_file = diagnostic.tier1_patch.get("file", "")
        issue = diagnostic.tier1_patch.get("issue", "")
        suggested_fix = diagnostic.tier1_patch.get("suggested_fix", "")

        # 1. Strict Boundary Check: MUST resolve strictly within core_engine/
        target_filename = Path(raw_file).name
        if target_filename == "remediation.py":
            return False, "Remediation engine code is locked and cannot modify itself"

        target_path = (CORE_ENGINE_PKG_DIR / target_filename).resolve()

        try:
            target_path.relative_to(CORE_ENGINE_PKG_DIR)
        except ValueError:
            err = f"Security Violation: Target file {target_path} is outside core_engine/"
            logger.error(err)
            return False, err

        if not target_path.exists() or not target_path.is_file():
            err = f"Target file {target_path} does not exist in core_engine/"
            logger.error(err)
            return False, err

        if dry_run:
            return True, f"DRY_RUN: Patch verified for core_engine/{target_filename} (skipped write)"

        # 2. Read original file and store in-memory backup
        original_content = target_path.read_text(encoding="utf-8")

        # 3. Prompt thinking model for exact contiguous search-and-replace block
        patch_prompt = f"""You are an expert Python engineer maintaining the 'prime-math' mathematical reasoning harness.
You need to fix a harness code bug in the file `core_engine/{target_filename}`.

DIAGNOSED ISSUE:
{issue}

SUGGESTED FIX:
{suggested_fix}

FAILURE CONTEXT:
Problem ID: {diagnostic.problem_id} ({diagnostic.title})
Last Error / Traceback:
{telemetry.get('last_error', 'None')}

CURRENT FILE CONTENT (`core_engine/{target_filename}`):
```python
{original_content}
```

MANDATORY INSTRUCTIONS:
1. Provide a single contiguous search-and-replace block to fix the bug.
2. `target_content` MUST BE an EXACT verbatim substring from the current file above (including exact indentation, whitespace, and line breaks) that occurs EXACTLY ONCE.
3. `replacement_content` is the drop-in replacement code.
4. Keep the patch MINIMAL and targeted (aim for a few lines of change).
5. Output STRICTLY a JSON object with keys "target_content" and "replacement_content".
6. Do NOT include markdown outside the JSON block.
"""
        try:
            resp = self.client.complete(
                messages=[{"role": "user", "content": patch_prompt}],
                max_tokens=4096,
                temperature=0.1,
            )
            data = extract_json_payload(resp.content)
            if not data and resp.reasoning:
                data = extract_json_payload(resp.reasoning)

            if not data or "target_content" not in data or "replacement_content" not in data:
                return False, "Failed to parse target_content and replacement_content from model response"

            target_content = data["target_content"]
            replacement_content = data["replacement_content"]

            if not target_content:
                return False, "Empty target_content in patch proposal"

            target_lines = len(target_content.strip().splitlines())
            replacement_lines = len(replacement_content.strip().splitlines())

            # Complexity Gate: Simple fixes (a few lines of code) are permitted.
            # Complex patches exceeding thresholds are parked for paired AGY review.
            if target_lines > self.max_tier1_lines_target or replacement_lines > self.max_tier1_lines_replacement:
                park_reason = (
                    f"Exceeds automated change threshold: {target_lines} target lines "
                    f"(max {self.max_tier1_lines_target}) and {replacement_lines} replacement lines "
                    f"(max {self.max_tier1_lines_replacement}). Parked for paired AGY review."
                )
                diagnostic.is_parked = True
                diagnostic.parked_reason = park_reason
                diagnostic.persisted_status = f"TIER_1_PARKED_COMPLEX: {park_reason}"
                self._log_parked_issue(
                    diagnostic=diagnostic,
                    telemetry=telemetry,
                    target_filename=target_filename,
                    issue=issue,
                    suggested_fix=suggested_fix,
                    reason=park_reason,
                    target_content=target_content,
                    replacement_content=replacement_content,
                    target_lines=target_lines,
                    replacement_lines=replacement_lines,
                )
                logger.warning(f"Problem {diagnostic.problem_id}: Tier 1 patch parked ({park_reason})")
                return False, park_reason

            occurrences = original_content.count(target_content)
            if occurrences == 0:
                target_content_stripped = target_content.strip()
                if original_content.count(target_content_stripped) == 1:
                    target_content = target_content_stripped
                    occurrences = 1
                else:
                    return False, f"target_content does not match any verbatim chunk in core_engine/{target_filename}"

            if occurrences > 1:
                return False, f"target_content is ambiguous (occurs {occurrences} times in core_engine/{target_filename})"

            new_content = original_content.replace(target_content, replacement_content, 1)

            # 4. AST syntax validation check
            try:
                ast.parse(new_content)
            except SyntaxError as syn_err:
                return False, f"SyntaxError in synthesized patch: {syn_err}"

            # 5. Write new content to disk
            target_path.write_text(new_content, encoding="utf-8")

            # 6. Run acceptance regression suite (pytest)
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/"],
                cwd=str(REPO_ROOT_DIR),
                capture_output=True,
                text=True,
                timeout=120,
            )

            if proc.returncode == 0:
                msg = f"SUCCESS: Patched core_engine/{target_filename} and verified with pytest (all tests passed)"
                logger.info(msg)
                return True, msg
            else:
                # Immediate Rollback on regression failure
                target_path.write_text(original_content, encoding="utf-8")
                tail_out = proc.stdout[-400:] if proc.stdout else (proc.stderr[-400:] if proc.stderr else "")
                err_msg = f"ROLLBACK: pytest regression failed (code {proc.returncode}). Reverted core_engine/{target_filename}. Output:\n{tail_out}"
                logger.warning(err_msg)
                return False, err_msg

        except Exception as exc:
            # Guarantee rollback on unexpected error
            target_path.write_text(original_content, encoding="utf-8")
            return False, f"Exception during patch execution: {exc}"

    def apply_remediation(
        self,
        diagnostic: ProblemDiagnostic,
        telemetry: Optional[Dict[str, Any]] = None,
        auto_apply_tier3: bool = True,
        auto_apply_tier1: Optional[bool] = None,
        dry_run: bool = False,
    ) -> ProblemDiagnostic:
        """Applies vetted remediations (persists Tier 3 heuristics into TurboVec, or applies verified Tier 1 patches inside core_engine/)."""
        if dry_run:
            diagnostic.persisted_status = "DRY_RUN (Skipped persistence/patching)"
            return diagnostic

        effective_auto_t1 = self.auto_apply_tier1 if auto_apply_tier1 is None else auto_apply_tier1

        if diagnostic.target_tier == RemediationTier.TIER_3_PROCEDURAL.value:
            if not diagnostic.tier3_heuristic:
                diagnostic.persisted_to_turbovec = False
                diagnostic.persisted_status = "TIER_3_SKIPPED (No universal heuristic proposed)"
                return diagnostic

            statement = telemetry.get("statement") if telemetry else None
            expected = telemetry.get("expected_answer") if telemetry else None
            is_valid, reason = is_valid_universal_heuristic(
                diagnostic.tier3_heuristic,
                problem_title=diagnostic.title,
                problem_id=diagnostic.problem_id,
                expected_answer=expected,
                problem_statement=statement,
            )
            if not is_valid:
                diagnostic.persisted_to_turbovec = False
                diagnostic.tier3_rejected_reason = reason
                diagnostic.persisted_status = f"TIER_3_REJECTED_NON_UNIVERSAL: {reason}"
                logger.warning(
                    f"[Problem {diagnostic.problem_id}] Rejected Tier 3 heuristic: {reason} "
                    f"(heuristic: '{diagnostic.tier3_heuristic}')"
                )
                return diagnostic

            if auto_apply_tier3:
                try:
                    added = self.memory.add_procedural_heuristic(
                        diagnostic.tier3_heuristic,
                        metadata={
                            "problem_id": diagnostic.problem_id,
                            "source": "remediation_engine",
                            "title": diagnostic.title,
                            "provenance_version": PROVENANCE_VERSION,
                        },
                    )
                    diagnostic.persisted_to_turbovec = added
                    diagnostic.persisted_status = (
                        "PERSISTED to TurboVec (prime_math_procedural)"
                        if added
                        else "SKIPPED (Near-duplicate already in TurboVec)"
                    )
                except TurboVecUnavailableError as exc:
                    diagnostic.persisted_to_turbovec = False
                    diagnostic.persisted_status = f"ERROR: TurboVec offline ({exc})"
                except Exception as exc:
                    diagnostic.persisted_to_turbovec = False
                    diagnostic.persisted_status = f"ERROR: {exc}"
            else:
                diagnostic.persisted_status = "PENDING_OPERATOR_APPROVAL"

        elif diagnostic.target_tier == RemediationTier.TIER_1_HARNESS.value:
            if diagnostic.is_parked:
                diagnostic.persisted_status = f"TIER_1_PARKED_COMPLEX: {diagnostic.parked_reason}"
            elif effective_auto_t1 and telemetry and diagnostic.tier1_patch:
                success, msg = self.apply_tier1_patch(diagnostic, telemetry, dry_run=dry_run)
                diagnostic.tier1_patch_applied = success
                if diagnostic.is_parked:
                    diagnostic.persisted_status = f"TIER_1_PARKED_COMPLEX: {diagnostic.parked_reason}"
                else:
                    diagnostic.persisted_status = f"TIER_1_APPLIED: {msg}" if success else f"TIER_1_FAILED: {msg}"
            else:
                diagnostic.persisted_status = "TIER_1_CODE_PATCH_STAGED (Pending Operator Review)"

        elif diagnostic.target_tier == RemediationTier.TIER_2_ORCHESTRATION.value:
            diagnostic.persisted_status = "TIER_2_INVARIANT_APPROVED"

        return diagnostic

    def run_remediation(
        self,
        problem_ids: Optional[List[int]] = None,
        latest_n_evals: int = 1,
        auto_apply_tier3: bool = True,
        auto_apply_tier1: bool = True,
        dry_run: bool = False,
        limit: Optional[int] = None,
    ) -> RemediationReport:
        """Executes full end-to-end 3-tier remediation cycle."""
        console.print(
            Panel(
                "[bold cyan]Prime Math 3-Tier Post-Evaluation Remediation Engine[/bold cyan]\n"
                "[dim]Diagnoses failures, evaluates Tier 2 Rubric Gate, and commits vetted heuristics to TurboVec.[/dim]",
                border_style="cyan",
            )
        )

        failures = self.find_failed_problems(latest_n_evals=latest_n_evals, problem_ids=problem_ids)
        if limit and len(failures) > limit:
            failures = failures[:limit]

        if not failures:
            console.print("[green]✓ No failed problems detected requiring remediation.[/green]")
            return RemediationReport(
                timestamp=datetime.datetime.now().isoformat(),
                total_analyzed=0,
                tier1_count=0,
                tier2_count=0,
                tier3_count=0,
                tier2_rerouted_count=0,
                heuristics_persisted=0,
                tier1_patched_count=0,
                tier1_parked_count=0,
                tier3_rejected_count=0,
                diagnostics=[],
            )

        console.print(f"[bold blue]Found {len(failures)} problems to diagnose & remediate...[/bold blue]")
        diagnostics: List[ProblemDiagnostic] = []

        tier1_cnt = 0
        tier2_cnt = 0
        tier3_cnt = 0
        rerouted_cnt = 0
        persisted_cnt = 0
        tier1_patched_cnt = 0
        tier1_parked_cnt = 0
        tier3_rejected_cnt = 0

        for i, fail_info in enumerate(failures, start=1):
            pid = fail_info["problem_id"]
            title = fail_info.get("title", f"Problem {pid}")
            console.print(f"\n[bold yellow][{i}/{len(failures)}] Diagnosing Problem {pid}: {title}...[/bold yellow]")

            telemetry = self.extract_failure_telemetry(fail_info)
            diagnostic = self.diagnose_failure(telemetry)
            diagnostic = self.apply_remediation(
                diagnostic,
                telemetry=telemetry,
                auto_apply_tier3=auto_apply_tier3,
                auto_apply_tier1=auto_apply_tier1,
                dry_run=dry_run,
            )

            if diagnostic.target_tier == RemediationTier.TIER_1_HARNESS.value:
                tier1_cnt += 1
                if diagnostic.tier1_patch_applied:
                    tier1_patched_cnt += 1
                elif diagnostic.is_parked:
                    tier1_parked_cnt += 1
            elif diagnostic.target_tier == RemediationTier.TIER_2_ORCHESTRATION.value:
                tier2_cnt += 1
            else:
                tier3_cnt += 1

            if diagnostic.tier3_rejected_reason:
                tier3_rejected_cnt += 1
            if diagnostic.rerouted_to_tier3:
                rerouted_cnt += 1
            if diagnostic.persisted_to_turbovec:
                persisted_cnt += 1

            diagnostics.append(diagnostic)

        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report = RemediationReport(
            timestamp=datetime.datetime.now().isoformat(),
            total_analyzed=len(diagnostics),
            tier1_count=tier1_cnt,
            tier2_count=tier2_cnt,
            tier3_count=tier3_cnt,
            tier2_rerouted_count=rerouted_cnt,
            heuristics_persisted=persisted_cnt,
            tier1_patched_count=tier1_patched_cnt,
            tier1_parked_count=tier1_parked_cnt,
            tier3_rejected_count=tier3_rejected_cnt,
            diagnostics=diagnostics,
        )

        # Save structured report
        report_path = REMEDIATION_DIR / f"remediation_{now_str}.json"
        try:
            with open(report_path, "w", encoding="utf-8") as fp:
                json.dump(asdict(report), fp, indent=2)
            logger.info(f"Saved remediation report to {report_path}")
        except Exception as e:
            logger.error(f"Failed to save remediation report: {e}")

        self.render_report(report)
        return report

    def render_report(self, report: RemediationReport) -> None:
        """Renders rich tables summarizing 3-tier remediation outcomes."""
        # Summary Overview Table
        summary_tbl = Table(title="3-Tier Remediation Summary Overview")
        summary_tbl.add_column("Metric", style="cyan")
        summary_tbl.add_column("Value", style="magenta")
        summary_tbl.add_row("Total Failures Diagnosed", str(report.total_analyzed))
        summary_tbl.add_row("Tier 1: Harness & Engine Patches", str(report.tier1_count))
        summary_tbl.add_row("Tier 1 Patches Applied & Verified", f"[bold green]{report.tier1_patched_count}[/bold green]" if report.tier1_patched_count else "0")
        summary_tbl.add_row("Tier 1 Issues Parked (AGY Review)", f"[bold yellow]{report.tier1_parked_count}[/bold yellow]" if report.tier1_parked_count else "0")
        summary_tbl.add_row("Tier 2: Universal Invariants (Approved)", str(report.tier2_count))
        summary_tbl.add_row("Tier 3: Procedural Heuristics (Total)", str(report.tier3_count))
        summary_tbl.add_row("Tier 3: Rejected (Non-Universal / Empirical)", f"[bold red]{report.tier3_rejected_count}[/bold red]" if report.tier3_rejected_count else "0")
        summary_tbl.add_row("Tier 2 -> Tier 3 Rerouted (Rubric Gate)", f"[yellow]{report.tier2_rerouted_count}[/yellow]")
        summary_tbl.add_row("Heuristics Persisted to TurboVec", f"[bold green]{report.heuristics_persisted}[/bold green]")
        console.print("\n", summary_tbl)

        # Problem Diagnostic Table
        diag_tbl = Table(title="Detailed Problem Remediation Breakdown")
        diag_tbl.add_column("Problem", style="cyan", width=12)
        diag_tbl.add_column("Target Tier", style="bold", width=16)
        diag_tbl.add_column("Rubric Gate", style="yellow", width=14)
        diag_tbl.add_column("Action / Status", style="green")
        diag_tbl.add_column("Root Cause / Heuristic", style="dim")

        for d in report.diagnostics:
            tier_style = (
                "[red]Tier 1 (Harness)[/red]"
                if d.target_tier == RemediationTier.TIER_1_HARNESS.value
                else (
                    "[cyan]Tier 2 (Prompt)[/cyan]"
                    if d.target_tier == RemediationTier.TIER_2_ORCHESTRATION.value
                    else "[blue]Tier 3 (Procedural)[/blue]"
                )
            )

            gate_str = "N/A"
            if d.tier2_rubric:
                gate_str = "[bold green]PASS[/bold green]" if d.tier2_rubric.verdict == "PASS" else "[bold yellow]REJECT -> T3[/bold yellow]"
            elif d.rerouted_to_tier3:
                gate_str = "[bold yellow]REJECT -> T3[/bold yellow]"

            if d.is_parked:
                status_str = "[bold yellow]PARKED (AGY Review)[/bold yellow]"
            elif d.tier3_rejected_reason:
                status_str = "[bold red]REJECTED (Non-Universal)[/bold red]"
            else:
                status_str = d.persisted_status or ("Applied" if d.persisted_to_turbovec else "Staged")

            summary_text = d.tier3_heuristic or (d.tier1_patch.get("issue") if d.tier1_patch else d.root_cause)
            summary_text = summary_text[:100] + "..." if len(summary_text) > 100 else summary_text

            diag_tbl.add_row(f"P{d.problem_id}", tier_style, gate_str, status_str, summary_text)

        console.print("\n", diag_tbl)

        # If there are Tier 1 patches, render code diagnosis
        t1_diagnostics = [d for d in report.diagnostics if d.target_tier == RemediationTier.TIER_1_HARNESS.value and d.tier1_patch]
        if t1_diagnostics:
            t1_tbl = Table(title="Tier 1: Harness & Engine Code Patches Staged")
            t1_tbl.add_column("Problem", style="cyan")
            t1_tbl.add_column("File", style="yellow")
            t1_tbl.add_column("Issue", style="red")
            t1_tbl.add_column("Suggested Fix", style="green")
            for d in t1_diagnostics:
                patch = d.tier1_patch or {}
                t1_tbl.add_row(f"P{d.problem_id}", patch.get("file", ""), patch.get("issue", ""), patch.get("suggested_fix", ""))
            console.print("\n", t1_tbl)

        # Dedicated Table for Parked Issues
        parked_diagnostics = [d for d in report.diagnostics if d.is_parked]
        if parked_diagnostics:
            parked_tbl = Table(title="[bold yellow]Tier 1 Issues Parked for AGY Operator Review[/bold yellow]")
            parked_tbl.add_column("Problem", style="cyan", width=10)
            parked_tbl.add_column("File", style="magenta", width=22)
            parked_tbl.add_column("Issue", style="red")
            parked_tbl.add_column("Parked Reason", style="yellow")
            for d in parked_diagnostics:
                patch = d.tier1_patch or {}
                parked_tbl.add_row(
                    f"P{d.problem_id}",
                    patch.get("file", "unknown"),
                    patch.get("issue", d.root_cause)[:70],
                    (d.parked_reason or "Complexity exceeded")[:90],
                )
            console.print("\n", parked_tbl)
            console.print(f"[dim yellow]> Logged full context to: {PARKED_ISSUES_FILE} (Ready for paired AGY session)[/dim yellow]\n")
