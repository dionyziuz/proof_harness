"""lean_harness_guards.py — Non-Destructive Core-Engine Guardrails, Indexing & Advisory Triage for Lean 4 Proving.

Implements the safe-mode harness improvements derived from MMAT (arXiv:2607.04394) and
AI Co-Mathematician (arXiv:2605.06651) at the `core_engine` level:

1. Theorem & Library Signature `Guarding` Gate (`extract_declaration_signatures`, `check_signature_drift`,
   `snapshot_and_verify_library_signatures`):
   - Protects `theorem target` and already-banked `Verified_Library.lean` declarations from silent
     signature drift or proposition weakening.
2. Witness-Gated `Obstruction` Registry (` register_obstruction`, `load_obstructions`,
   `format_obstructions_briefing`):
   - Stores machine-readable mathematical dead routes in `<work_dir>/obstructions.json`.
   - Enforces the witness-only safety rule: only concrete mathematical refutations (counterexamples,
     LP/numerical feasibility witnesses, or vacuous-hypothesis proofs) are recorded—never mere tactic
     or syntax failures.
3. Non-Destructive Library `Golfer` / Compact Signature Indexer (`extract_library_index`,
   `search_library_signatures`, `write_library_index_markdown`, `format_compact_library_toc`):
   - Never deletes any declaration from `Verified_Library.lean`.
   - Builds a searchable signature Table of Contents and exposes `search_verified_library(query)`
     inside `LeanCompliantKernel`.
4. Advisory-Only `Regulator` Failure Triage (`classify_lean_failure_triage`):
   - Annotates compiler repair and strategic synthesis prompts with failure classification
     (`TACTIC_OR_TYPECLASS_GAP`, `PREMISE_OR_DUPLICATE_LOOKUP`, `UNSOLVED_SUBGOAL_SANITY_CHECK`)
     without blocking or skipping repair attempts.
5. Advisory-Only Stall Detector & Cached arXiv Literature Enrichment (`record_cycle_and_check_stall`,
   `retrieve_cached_arxiv_briefing`):
   - Tracks cycle progress in `archive/logs/stall_monitor.json` and emits non-blocking advisory
     diagnostics + cached `core_engine.research.search_math(..., prioritize_arxiv=True)` literature
     briefings when consecutive zero-declaration cycles occur.
6. Persistent Computational Workspace (`save_computational_witness`, `load_computational_witnesses`,
   `format_computational_witness_briefing`):
   - Persists Python REPL empirical findings, LP bounds, and finite-case witnesses across turns and
     cycles in `<work_dir>/archive/computational_workspace/witness_cache.json`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import os
import re

from .research import search_math

_DECL_HEADER_RE = re.compile(
    r"^(?:private\s+)?(?:noncomputable\s+)?(theorem|lemma|def|abbrev)\s+([A-Za-z_][A-Za-z0-9_']*)\b(.*?)(?::=|\bwhere\b)",
    re.MULTILINE | re.DOTALL,
)


def _strip_comments(code: str) -> str:
    """Strips Lean 4 block and line comments for deterministic signature comparison."""
    no_block = re.sub(r"/\-.*?\-/", " ", code, flags=re.DOTALL)
    no_line = re.sub(r"--[^\n]*", " ", no_block)
    return no_line


def normalize_signature(sig_text: str) -> str:
    """Collapses whitespace in a Lean 4 declaration header for canonical comparison."""
    cleaned = _strip_comments(sig_text)
    return re.sub(r"\s+", " ", cleaned).strip()


def extract_declaration_signatures(code: str) -> Dict[str, Dict[str, str]]:
    """Extracts `{name: {'kind': kind, 'signature': normalized_header, 'sha256': hash}}` from Lean 4 code."""
    cleaned = _strip_comments(code)
    result: Dict[str, Dict[str, str]] = {}
    for m in _DECL_HEADER_RE.finditer(cleaned):
        kind = m.group(1).strip()
        name = m.group(2).strip()
        params_and_type = re.sub(r"\s+", " ", m.group(3)).strip()
        norm_sig = f"{kind} {name} {params_and_type}".strip()
        sig_hash = hashlib.sha256(norm_sig.encode("utf-8")).hexdigest()[:16]
        result[name] = {
            "kind": kind,
            "name": name,
            "signature": norm_sig,
            "params_and_type": params_and_type,
            "sha256": sig_hash,
        }
    return result


def check_signature_drift(
    candidate_code: str,
    baseline_signatures: Dict[str, Dict[str, str]],
    protected_names: Optional[List[str]] = None,
) -> Tuple[bool, List[str]]:
    """Checks whether `candidate_code` mutates or weakens any protected/baseline signature.

    Returns `(ok, drift_warnings)`. Purely protective: only flags declarations that exist in both
    `candidate_code` and `baseline_signatures` with a *different* normalized type signature.
    """
    cand_sigs = extract_declaration_signatures(candidate_code)
    scope = set(protected_names) if protected_names is not None else set(baseline_signatures.keys())
    drifts: List[str] = []
    for name in sorted(scope):
        if name in cand_sigs and name in baseline_signatures:
            expected = baseline_signatures[name].get("params_and_type", "")
            actual = cand_sigs[name].get("params_and_type", "")
            if expected and actual and expected != actual:
                drifts.append(
                    f"Signature drift detected on `{name}`: expected `{expected[:160]}`, got `{actual[:160]}`"
                )
    return (len(drifts) == 0), drifts


def snapshot_and_verify_library_signatures(
    work_dir: Path,
    candidate_decls: Dict[str, str],
    prev_verified_decls: Dict[str, str],
) -> Tuple[Dict[str, str], List[str]]:
    """Enforces monotonic signature guarding when aggregating declarations into `Verified_Library.lean`.

    If a candidate declaration has the same name as an already-verified declaration in `prev_verified_decls`
    but alters its type signature, restores the verified declaration verbatim and logs the drift warning.
    Also writes a signature snapshot to `archive/logs/signature_guard_snapshot.json`.
    """
    guarded_decls = dict(candidate_decls)
    warnings: List[str] = []

    prev_sigs: Dict[str, Dict[str, str]] = {}
    for name, body in prev_verified_decls.items():
        extracted = extract_declaration_signatures(body)
        if name in extracted:
            prev_sigs[name] = extracted[name]

    for name, prev_body in prev_verified_decls.items():
        if name in guarded_decls and guarded_decls[name] != prev_body:
            cand_extracted = extract_declaration_signatures(guarded_decls[name])
            if name in prev_sigs and name in cand_extracted:
                if prev_sigs[name]["params_and_type"] != cand_extracted[name]["params_and_type"]:
                    warnings.append(
                        f"Guarded `{name}` against signature mutation "
                        f"(kept verified signature `{prev_sigs[name]['params_and_type'][:120]}`)."
                    )
                    guarded_decls[name] = prev_body

    # Persist snapshot of all guarded signatures
    try:
        logs_dir = work_dir / "archive" / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        snap_file = logs_dir / "signature_guard_snapshot.json"
        all_sigs: Dict[str, Dict[str, str]] = {}
        for name, body in guarded_decls.items():
            ext = extract_declaration_signatures(body)
            if name in ext:
                all_sigs[name] = ext[name]
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "total_guarded_declarations": len(all_sigs),
            "warnings": warnings,
            "signatures": all_sigs,
        }
        snap_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass

    return guarded_decls, warnings


# ==============================================================================
# 2. Witness-Gated Obstruction Registry (`obstructions.json`)
# ==============================================================================

_TACTIC_ONLY_REJECT_RE = re.compile(
    r"^(?:omega|ring|linarith|simp|rw|rewrite|typeclass|coercion|timeout|syntax)\s+(?:failed|error)\b",
    re.IGNORECASE,
)


def _obstructions_file(work_dir: Path) -> Path:
    return work_dir.resolve() / "obstructions.json"


def load_obstructions(work_dir: Optional[Path]) -> List[Dict[str, Any]]:
    """Loads witness-verified mathematical obstructions from `<work_dir>/obstructions.json`."""
    if work_dir is None:
        return []
    obs_path = _obstructions_file(work_dir)
    if not obs_path.is_file():
        return []
    try:
        data = json.loads(obs_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("obstructions"), list):
            return data["obstructions"]
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def register_obstruction(
    work_dir: Path,
    dead_route: str,
    witness_or_no_go: str,
    banned_pattern: str = "",
    verified_by: str = "empirical_or_kernel_witness",
) -> Dict[str, Any]:
    """Registers a witness-backed mathematical obstruction in `<work_dir>/obstructions.json`.

    SAFETY GATE: Rejects vague or tactic-only failures so valid proof strategies are never
    prematurely banned due to temporary Lean syntax/tactic friction.
    """
    route_clean = (dead_route or "").strip()
    witness_clean = (witness_or_no_go or "").strip()
    if len(route_clean) < 10 or len(witness_clean) < 12:
        return {
            "registered": False,
            "reason": "Rejected: obstruction requires both a specific `dead_route` and a concrete mathematical `witness_or_no_go`.",
        }
    if _TACTIC_ONLY_REJECT_RE.match(witness_clean):
        return {
            "registered": False,
            "reason": "Rejected by Witness-Only Safety Gate: tactic/syntax failures are not mathematical obstructions.",
        }

    existing = load_obstructions(work_dir)
    for item in existing:
        if item.get("dead_route", "").strip().lower() == route_clean.lower():
            item["witness_or_no_go"] = witness_clean
            item["banned_pattern"] = banned_pattern.strip() or item.get("banned_pattern", "")
            item["updated_at"] = datetime.now(timezone.utc).isoformat()
            _save_obstructions(work_dir, existing)
            return {"registered": True, "updated": True, "id": item.get("id"), "total": len(existing)}

    new_id = f"OBS-{len(existing) + 1:03d}"
    entry = {
        "id": new_id,
        "dead_route": route_clean,
        "witness_or_no_go": witness_clean,
        "banned_pattern": banned_pattern.strip(),
        "verified_by": verified_by,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    existing.append(entry)
    _save_obstructions(work_dir, existing)
    return {"registered": True, "updated": False, "id": new_id, "total": len(existing)}


def _save_obstructions(work_dir: Path, obstructions: List[Dict[str, Any]]) -> None:
    obs_path = _obstructions_file(work_dir)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "policy": "WITNESS_GATED_ONLY (concrete counterexamples, LP/numerical witnesses, or vacuous-hypothesis proofs; never tactic/syntax errors)",
        "obstructions": obstructions,
    }
    tmp_path = obs_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, obs_path)


def format_obstructions_briefing(work_dir: Optional[Path]) -> str:
    """Formats the witness-verified obstruction registry for prompt injection."""
    items = load_obstructions(work_dir)
    if not items:
        return ""
    lines = [
        "\n\n--- VERIFIED MATHEMATICAL OBSTRUCTION REGISTRY (WITNESS-GATED — DO NOT RE-ATTEMPT) ---",
        "The following routes have been mathematically refuted with concrete witnesses/counterexamples:",
    ]
    for it in items[-12:]:
        oid = it.get("id", "OBS")
        route = it.get("dead_route", "")
        witness = it.get("witness_or_no_go", "")
        banned = it.get("banned_pattern", "")
        ban_str = f" | Banned Pattern: `{banned}`" if banned else ""
        lines.append(f"  • [{oid}] Dead Route: {route}\n    Witness / No-Go: {witness}{ban_str}")
    return "\n".join(lines) + "\n"


# ==============================================================================
# 3. Non-Destructive Library Signature Indexer (`search_verified_library`)
# ==============================================================================


def extract_library_index(library_path: Optional[Path]) -> List[Dict[str, str]]:
    """Extracts an ordered list of `{name, kind, signature, params_and_type}` from `Verified_Library.lean`."""
    if library_path is None or not library_path.is_file():
        return []
    try:
        code = library_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    sigs = extract_declaration_signatures(code)
    return list(sigs.values())


def search_library_signatures(
    library_path: Optional[Path],
    query: str = "",
    limit: int = 25,
) -> str:
    """Searches `Verified_Library.lean` declaration names and type signatures without truncating the library."""
    entries = extract_library_index(library_path)
    if not entries:
        return "No verified declarations found in preloaded library."
    q = (query or "").strip().lower()
    if not q:
        selected = entries[-limit:]
        header = f"Showing latest {len(selected)}/{len(entries)} verified declarations in {library_path.name}:"
    else:
        tokens = [t for t in re.split(r"\s+", q) if t]
        scored: List[Tuple[int, int, Dict[str, str]]] = []
        for idx, item in enumerate(entries):
            hay = f"{item['name']} {item['params_and_type']}".lower()
            score = sum(3 if t in item["name"].lower() else (1 if t in hay else 0) for t in tokens)
            if score > 0:
                scored.append((score, idx, item))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        selected = [x[2] for x in scored[:limit]]
        header = f"Found {len(selected)} matching verified declarations (out of {len(entries)} total) for query '{query}':"

    if not selected:
        return f"No verified declarations matched '{query}' (total declarations: {len(entries)}). Try a broader substring."
    lines = [header]
    for it in selected:
        lines.append(f"  - {it['signature']}")
    return "\n".join(lines)


def write_library_index_markdown(work_dir: Path, library_path: Path) -> Optional[Path]:
    """Generates `<work_dir>/Verified_Library_Index.md` as a compact, non-destructive signature index."""
    entries = extract_library_index(library_path)
    if not entries:
        return None
    out_md = work_dir / "Verified_Library_Index.md"
    lines = [
        f"# Verified Library Signature Index (`{library_path.name}`)",
        f"- **Updated:** `{datetime.now(timezone.utc).isoformat()}`",
        f"- **Total Kernel-Verified Declarations:** `{len(entries)}`",
        "",
        "## Recent Frontier Declarations (Last 40)",
    ]
    for it in entries[-40:]:
        lines.append(f"- `{it['signature']}`")
    lines.append("")
    lines.append("## Full Declaration Table of Contents")
    for it in entries:
        lines.append(f"- **`{it['name']}`** ({it['kind']}): `{it['params_and_type'][:180]}`")
    try:
        out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out_md
    except Exception:
        return None


def format_compact_library_toc(library_path: Optional[Path], max_recent: int = 25) -> str:
    """Formats a compact signature index of recent declarations for prompt context efficiency."""
    entries = extract_library_index(library_path)
    if not entries:
        return ""
    recent = entries[-max_recent:]
    lines = [
        f"\n\n--- PRELOADED VERIFIED LIBRARY SIGNATURE INDEX ({len(entries)} total in `{library_path.name}`) ---",
        "Tip: In the Python REPL, call `search_verified_library('keyword')` to inspect exact type signatures of any of the "
        f"{len(entries)} preloaded declarations. Recent {len(recent)} signatures:",
    ]
    for it in recent:
        lines.append(f"  • {it['signature'][:200]}")
    return "\n".join(lines) + "\n"


# ==============================================================================
# 4. Advisory-Only Regulator Failure Triage
# ==============================================================================


def classify_lean_failure_triage(
    compiler_output: str,
    status: str = "",
    verified_theorems: Optional[List[str]] = None,
) -> str:
    """Deterministic, non-blocking failure classifier (MMAT Regulator pattern).

    Categorizes Lean 4 compiler diagnostics to focus repair/next-cycle reasoning without ever
    aborting a valid proof attempt.
    """
    out = compiler_output or ""
    if not out.strip() and status == " PASSED":
        return ""

    categories: List[str] = []
    advice: List[str] = []

    if "has already been declared" in out:
        categories.append("DUPLICATE_PRELOADED_DECLARATION")
        advice.append(
            "Delete the duplicate lemma/def from your candidate block and call the preloaded `Verified_Library.lean` "
            "symbol directly (use `search_verified_library('<name>')` in Python REPL to check its exact signature)."
        )
    if "unknown identifier" in out or "unknown constant" in out:
        unk = re.findall(r"unknown (?:identifier|constant) '([^']+)'", out)
        unk_preview = ", ".join(dict.fromkeys(unk[:5]))
        categories.append("MISSING_PREMISE_OR_IDENTIFIER")
        advice.append(
            f"Unknown identifier(s) ({unk_preview or 'see compiler output'}): check `search_verified_library(...)` "
            "or test exact Mathlib lemma names with a 5-line `#check` inside `run_lean(...)` in the Python REPL."
        )
    if any(
        k in out
        for k in (
            "type mismatch",
            "failed to synthesize",
            "application type mismatch",
            "coe",
            "Finset.sup",
        )
    ):
        categories.append("TYPECLASS_OR_COERCION_MISMATCH")
        advice.append(
            "Type/coercion mismatch: isolate the failing `have` step into a standalone 10-line REPL check and verify "
            "`ℕ`/`ℤ`/`ℝ` casts (`exact_mod_cast`, `push_cast`, `omega`, or explicit type annotations)."
        )
    if any(
        k in out
        for k in (
            "unsolved goals",
            "omega could not prove",
            "linarith failed",
            "ring failed",
            "Tactic `decide` failed",
        )
    ):
        categories.append("UNSOLVED_SUBGOAL_OR_ARITHMETIC_GAP")
        advice.append(
            "Unsolved goal / arithmetic gap: (1) check if an explicit positivity or disjointness hypothesis is missing "
            "from the local context, and (2) optionally test the intermediate inequality on small finite cases in the "
            "Python REPL (`save_computational_witness(...)`) to confirm no edge-case hypothesis was omitted."
        )
    if "(deterministic) timeout" in out or "maximum recursion depth" in out or "whnf" in out:
        categories.append("ELABORATION_OR_WHNF_TIMEOUT")
        advice.append(
            "Elaboration/whnf timeout: break the monolithic proof into 2-3 smaller helper lemmas with explicit type "
            "signatures and avoid heavy `simp` or `decide` over symbolic expressions."
        )

    if not categories:
        return ""

    v_count = len(verified_theorems or [])
    lines = [
        f"--- ADVISORY REGULATOR FAILURE TRIAGE ({', '.join(categories)}) ---",
        f"• Verified helper declarations preserved so far: {v_count}",
    ]
    for item in advice:
        lines.append(f"• {item}")
    return "\n".join(lines)


# ==============================================================================
# 5. Advisory-Only Stall Monitor & Cached arXiv Literature Lookup
# ==============================================================================


def _extract_error_fingerprint(validation_summary: str) -> str:
    """Computes a short normalized fingerprint of the primary compiler error class."""
    s = (validation_summary or "").strip()
    m = re.search(r"error:\s*([^\n;]{1,100})", s)
    if m:
        norm = re.sub(r"/tmp/[^\s:]+:\d+:\d+:", "", m.group(1)).strip()
        norm = re.sub(r"\b\d+\b", "N", norm)
        return norm[:80]
    return s[:80]


def record_cycle_and_check_stall(
    work_dir: Path,
    cycle: int,
    total_decls: int,
    delta_decls: int,
    validation_summary: str,
) -> Dict[str, Any]:
    """Records cycle outcome in `archive/logs/stall_monitor.json` and returns advisory stall status.

    NEVER halts or aborts the proving loop; purely produces diagnostic context for strategic synthesis.
    """
    logs_dir = work_dir.resolve() / "archive" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    mon_file = logs_dir / "stall_monitor.json"

    history: List[Dict[str, Any]] = []
    if mon_file.is_file():
        try:
            data = json.loads(mon_file.read_text(encoding="utf-8"))
            if isinstance(data.get("history"), list):
                history = data["history"]
        except Exception:
            history = []

    err_fp = _extract_error_fingerprint(validation_summary)
    entry = {
        "cycle": cycle,
        "total_decls": total_decls,
        "delta_decls": delta_decls,
        "error_fingerprint": err_fp,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    # Replace if re-running same cycle number
    history = [h for h in history if h.get("cycle") != cycle]
    history.append(entry)
    history.sort(key=lambda x: int(x.get("cycle", 0)))

    consecutive_zero_decls = 0
    for h in reversed(history):
        if int(h.get("delta_decls", 0)) <= 0:
            consecutive_zero_decls += 1
        else:
            break

    repeated_error_cycles = 0
    if err_fp:
        for h in reversed(history):
            if h.get("error_fingerprint") == err_fp:
                repeated_error_cycles += 1
            else:
                break

    is_advisory_stall = (consecutive_zero_decls >= 2) or (repeated_error_cycles >= 2)
    advisory_message = ""
    if is_advisory_stall:
        advisory_message = (
            f"\n\n--- ADVISORY STALL DIAGNOSTIC (NON-BLOCKING) ---\n"
            f"• Consecutive cycles with 0 newly banked library declarations: {consecutive_zero_decls}\n"
            f"• Consecutive cycles with same error fingerprint (`{err_fp}`): {repeated_error_cycles}\n"
            "• Advisory Guidance (Do NOT abandon a nearly-compiling lemma if only a minor coercion remains):\n"
            "  1. If the current lemma is failing solely on Lean 4 tactic/typeclass friction, decompose it into "
            "2 smaller atomic helper lemmas so the working half banks immediately into `Verified_Library.lean`.\n"
            "  2. If the subgoal itself may have a missing hypothesis or mathematical gap, run a quick finite Python "
            "verification in the REPL and check `obstructions.json` before spending another full cycle on the same form.\n"
        )

    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "consecutive_zero_decls": consecutive_zero_decls,
        "repeated_error_cycles": repeated_error_cycles,
        "is_advisory_stall": is_advisory_stall,
        "history": history[-50:],
    }
    try:
        mon_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass

    return {
        "is_advisory_stall": is_advisory_stall,
        "consecutive_zero_decls": consecutive_zero_decls,
        "repeated_error_cycles": repeated_error_cycles,
        "advisory_message": advisory_message,
    }


def retrieve_cached_arxiv_briefing(
    work_dir: Optional[Path],
    query_seed: str,
    count: int = 3,
    force_refresh: bool = False,
) -> str:
    """Retrieves and caches arXiv-prioritized literature snippets via `core_engine.research.search_math`."""
    if not query_seed or not query_seed.strip():
        return ""
    cache_file: Optional[Path] = None
    if work_dir is not None:
        logs_dir = work_dir.resolve() / "archive" / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        cache_file = logs_dir / "arxiv_literature_cache.json"
        if cache_file.is_file() and not force_refresh:
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                if cached.get("briefing"):
                    return str(cached["briefing"])
            except Exception:
                pass

    # Prefer the human-readable problem title in `(Subnet 66: ... (<Title>))` or Section 4 if present
    title_m = re.search(r"\(Subnet\s*66:[^(]*\(([^)]+)\)\)", query_seed)
    seed_text = title_m.group(1) if title_m else query_seed

    # Extract clean mathematical keywords from seed_text, skipping harness/Lean boilerplate
    clean_line = re.sub(r"```.*?```", " ", seed_text, flags=re.DOTALL)
    clean_line = re.sub(r"\bfc-v2-[A-Za-z0-9\-]+\b", " ", clean_line)
    clean_line = re.sub(r"[^A-Za-z0-9_\-\s]", " ", clean_line)
    boilerplate = {
        "theorem",
        "lemma",
        "target",
        "import",
        "mathlib",
        "formalconjectures",
        "prove",
        "section",
        "mandatory",
        "validator",
        "wrapper",
        "formal",
        "formalizing",
        "proof",
        "bittensor",
        "subnet",
        "conjectures",
        "task",
        "solving",
        "lean",
        "exact",
        "solution",
        "header",
        "footer",
        "bounty",
        "inserted",
        "here",
        "math30",
        "catalog",
        "main",
    }
    words = [
        w
        for w in clean_line.split()
        if len(w) > 2 and w.lower() not in boilerplate and not w.isdigit()
    ]
    short_query = " ".join(dict.fromkeys(words[:10]))
    if not short_query:
        return ""

    try:
        results = search_math(short_query, count=count, timeout=20.0, prioritize_arxiv=True)
    except Exception:
        return ""

    if not results:
        return ""

    lines = ["\n\n--- CACHED ARXIV & MATHEMATICAL LITERATURE BRIEFING ---"]
    for idx, r in enumerate(results[:count], 1):
        title = r.get("title", "arXiv Preprint")
        url = r.get("url", "")
        snippet = re.sub(r"\s+", " ", (r.get("content") or r.get("snippet") or "")).strip()[:450]
        lines.append(f"  [{idx}] {title} ({url})\n      {snippet}")
    briefing = "\n".join(lines) + "\n"

    if cache_file is not None:
        try:
            cache_file.write_text(
                json.dumps(
                    {
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "query": short_query,
                        "results": results[:count],
                        "briefing": briefing,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass
    return briefing


# ==============================================================================
# 6. Persistent Computational Workspace (`witness_cache.json`)
# ==============================================================================


def _witness_cache_file(work_dir: Path) -> Path:
    ws_dir = work_dir.resolve() / "archive" / "computational_workspace"
    ws_dir.mkdir(parents=True, exist_ok=True)
    return ws_dir / "witness_cache.json"


def load_computational_witnesses(work_dir: Optional[Path]) -> Dict[str, Any]:
    """Loads persisted Python REPL empirical checks/witnesses for the problem working directory."""
    if work_dir is None:
        return {}
    cfile = _witness_cache_file(work_dir)
    if not cfile.is_file():
        return {}
    try:
        data = json.loads(cfile.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("witnesses"), dict):
            return data["witnesses"]
    except Exception:
        pass
    return {}


def save_computational_witness(
    work_dir: Path,
    key: str,
    summary: str,
    data: Any = None,
) -> Dict[str, Any]:
    """Persists an empirical finding, finite-case verification, or LP bound across turns and cycles."""
    k = (key or "").strip()
    if not k:
        return {"saved": False, "reason": "Empty witness key"}
    witnesses = load_computational_witnesses(work_dir)
    witnesses[k] = {
        "summary": (summary or "").strip()[:1000],
        "data": data,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    cfile = _witness_cache_file(work_dir)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "witnesses": witnesses,
    }
    tmp = cfile.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, cfile)
    return {"saved": True, "key": k, "total_witnesses": len(witnesses)}


def format_computational_witness_briefing(work_dir: Optional[Path]) -> str:
    """Formats persisted computational witnesses for prompt injection."""
    witnesses = load_computational_witnesses(work_dir)
    if not witnesses:
        return ""
    lines = [
        "\n\n--- PERSISTENT COMPUTATIONAL WORKSPACE (EMPIRICAL WITNESS CACHE) ---",
        "Tip: Use `save_computational_witness(key, summary, data)` and `load_computational_witnesses()` "
        "in the Python REPL to persist finite checks across turns/cycles:",
    ]
    for k, val in list(witnesses.items())[-10:]:
        lines.append(f"  • [{k}]: {val.get('summary', '')}")
    return "\n".join(lines) + "\n"
