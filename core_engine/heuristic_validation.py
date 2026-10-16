"""Validation and leak-prevention gates for procedural-memory heuristics.

This module is intentionally dependency-free (stdlib only) so that both
``core_engine.memory`` and ``core_engine.remediation`` can import it without
creating an import cycle (memory <-> remediation).

It provides three groups of helpers:

1. ``is_valid_universal_heuristic``: the universality / anti-hack gate applied to
   every heuristic before it is committed to TurboVec procedural memory.
2. ``find_ground_truth_leak``: a post-filter that rejects heuristics carrying the
   benchmark's expected answer, distinctive numeric constants (>= 4 digits) from
   the problem statement, or the problem id / title.
3. ``redact_ground_truth``: scrubs reference answers from free text (log excerpts,
   failure messages) before that text is placed into any LLM prompt whose output
   can be written to memory.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional, Set, Tuple

# Bumped whenever the provenance/leak-safety contract of persisted heuristics changes.
# Records carrying this key in their payload were written by leak-safe code paths.
PROVENANCE_VERSION: int = 2

REDACTED_TOKEN = "[REDACTED]"

# Thousands-separated numbers (1,234,567 / 1_234_567) or plain digit runs.
_NUMBER_TOKEN_RE = re.compile(r"\d{1,3}(?:[,_]\d{3})+(?!\d)|\d+")

_GROUND_TRUTH_LABEL_RE = re.compile(
    r"(?i)\b(expected(?:\s+answer)?|ground[\s_-]*truth|reference\s+answer|correct\s+answer|true\s+answer)"
    r"(\s*(?:[:=]|\bis\b|\bwas\b)?\s*)"
    r"([^\s,;)\]]+)"
)


def _strip_separators(token: str) -> str:
    return token.replace(",", "").replace("_", "")


def extract_number_tokens(text: Optional[str]) -> Set[str]:
    """Returns the set of integer digit-runs in ``text`` (thousands separators removed)."""
    if not text:
        return set()
    return {_strip_separators(t) for t in _NUMBER_TOKEN_RE.findall(str(text))}


def extract_distinctive_constants(statement: Optional[str], min_digits: int = 4) -> Set[str]:
    """Numeric constants in a problem statement distinctive enough to fingerprint it (>= ``min_digits`` digits)."""
    return {t for t in extract_number_tokens(statement) if len(t) >= min_digits}


def _normalize_answer(ans: Any) -> str:
    s = str(ans).strip()
    s = re.sub(r"\\boxed\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    return s.replace("$", "").replace(",", "").replace(" ", "").replace("_", "").strip()


def find_ground_truth_leak(
    text: Optional[str],
    expected_answer: Optional[Any] = None,
    problem_statement: Optional[str] = None,
    problem_id: Optional[Any] = None,
    problem_title: Optional[str] = None,
) -> Optional[str]:
    """Returns a human-readable reason if ``text`` leaks benchmark ground truth, else ``None``.

    Checks (any hit rejects):
    - the expected answer string (integers of >= 2 digits matched as whole numbers, decimals and
      symbolic answers matched as substrings);
    - any numeric constant with >= 4 digits that appears in the problem statement;
    - the problem id (prefixed form like "problem 850"/"#850", or a bare >= 3-digit token);
    - the problem title (> 6 chars) or any mention of "Project Euler".
    """
    if not text:
        return None
    raw = str(text)
    lowered = raw.lower()
    tokens = extract_number_tokens(raw)

    if expected_answer is not None:
        norm = _normalize_answer(expected_answer)
        if norm:
            if re.fullmatch(r"[-+]?\d+", norm):
                digits = norm.lstrip("+-").lstrip("0") or "0"
                if len(digits) >= 2 and digits in {t.lstrip("0") or "0" for t in tokens}:
                    return f"Heuristic contains the expected answer ({digits})"
            elif re.fullmatch(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", norm):
                if norm.lstrip("+-") in raw.replace(",", "").replace(" ", ""):
                    return f"Heuristic contains the expected answer ({norm})"
            elif len(norm) >= 3 and norm.lower() in lowered.replace(" ", ""):
                return "Heuristic contains the expected answer string"

    constants = extract_distinctive_constants(problem_statement)
    hits = sorted(constants & tokens)
    if hits:
        return f"Heuristic contains distinctive problem-statement constant(s): {', '.join(hits[:3])}"

    if problem_id is not None:
        pid = str(problem_id).strip()
        if pid:
            if re.search(
                rf"(?:\b(?:problem|pid|p|euler|task|id)\s*#?\s*|#\s*){re.escape(pid)}\b", raw, re.IGNORECASE
            ):
                return f"Heuristic references problem id {pid}"
            if pid.isdigit() and len(pid) >= 3 and pid in tokens:
                return f"Heuristic contains bare problem id token {pid}"

    if problem_title:
        clean_title = " ".join(str(problem_title).split()).lower()
        if len(clean_title) > 6 and clean_title in " ".join(lowered.split()):
            return f"Heuristic references problem title '{problem_title}'"

    if re.search(r"\bproject\s*euler\b", lowered):
        return "Heuristic references Project Euler"

    return None


def redact_ground_truth(text: Optional[str], expected_answer: Optional[Any] = None) -> str:
    """Scrubs reference-answer values from free text before it enters an LLM prompt.

    - Any ``Expected[ Answer]: X`` / ``ground truth: X`` / ``correct answer is X`` value is replaced.
    - If ``expected_answer`` has >= 2 significant characters, every whole-token occurrence of it
      (with or without thousands separators) is replaced.
    """
    if not text:
        return text or ""
    out = _GROUND_TRUTH_LABEL_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED_TOKEN}", str(text))

    if expected_answer is not None:
        norm = _normalize_answer(expected_answer)
        if len(norm.lstrip("+-")) >= 2:
            if re.fullmatch(r"[-+]?\d+", norm):
                digits = norm.lstrip("+-")
                # Match the integer as a whole token, tolerating thousands separators.
                sep_pattern = r"[,_]?".join(re.escape(ch) for ch in digits)
                out = re.sub(rf"(?<![\d.]){sep_pattern}(?![\d])", REDACTED_TOKEN, out)
            else:
                out = re.sub(re.escape(str(expected_answer).strip()), REDACTED_TOKEN, out)
                if norm != str(expected_answer).strip():
                    out = re.sub(re.escape(norm), REDACTED_TOKEN, out)
    return out


def is_valid_universal_heuristic(
    text: str,
    problem_title: Optional[str] = None,
    problem_id: Optional[Any] = None,
    expected_answer: Optional[Any] = None,
    problem_statement: Optional[str] = None,
) -> Tuple[bool, str]:
    """Validates that a proposed heuristic is a genuine, universal mathematical strategy.

    Enforces strict quality and universality gates before any heuristic can be committed
    to TurboVec procedural memory:
    1. Substantive length and non-triviality (>= 25 characters, non-placeholder).
    2. Zero empirical sieving or extrapolation hacks (Richardson, Aitken, sampling checkpoints,
       finite density approximations, brute-force search).
    3. Zero problem-specific bindings (problem numbers/IDs, benchmark titles, instance variables).
    4. Zero ground-truth leakage (expected answer, >= 4-digit statement constants, problem id/title)
       when ``expected_answer`` / ``problem_statement`` context is supplied.
    """
    if not text or not isinstance(text, str):
        return False, "Heuristic is empty or not a string"

    cleaned = text.strip()
    if len(cleaned) < 25:
        return False, f"Heuristic is too short ({len(cleaned)} chars < 25 chars)"

    # 1. Trivial or placeholder responses
    placeholder_patterns = [
        r"^(?:none|n/?a|null|no\s+(?:heuristic|reflection|learning)|nothing\s+to\s+add)\b",
        r"^review\s+(?:constraints|numerical|code|precision)\b",
        r"^no\s+generalizable\s+heuristic\b",
    ]
    for pat in placeholder_patterns:
        if re.search(pat, cleaned, re.IGNORECASE):
            return False, f"Heuristic matches trivial/placeholder pattern: '{pat}'"

    # 2. Empirical sieving, Richardson/Aitken extrapolation, or brute-force search
    empirical_patterns = [
        (r"\brichardson(?:\s+extrapolation)?\b", "Richardson extrapolation"),
        (r"\baitken(?:'s)?(?:\s+delta|\s+extrapolation)?\b", "Aitken extrapolation"),
        (r"\bshanks(?:\s+transformation)?\b", "Shanks transformation"),
        (r"\b(?:sample|sampling)\s+checkpoints\b", "Sampling checkpoints"),
        (r"\bextrapolat(?:e|ing|ion)\s+(?:the\s+limit|to\s+infinity|asymptotic|from\s+checkpoints)\b", "Asymptotic extrapolation hack"),
        (r"\bsieve\s+(?:primes\s+)?up\s+to\b", "Empirical prime sieving limit"),
        (r"\bbrute\s*-?\s*force\s+sieve\b", "Brute-force sieve"),
        (r"\bfast_spf\b", "Empirical fast_spf sieve"),
        (r"\bapproximate\s+(?:infinite\s+)?density\s+by\s+(?:sieving|sampling)\b", "Empirical density approximation"),
        (r"\bempirical\s+(?:sampling|extrapolation|density\s+estimation)\b", "Empirical estimation"),
    ]
    for pat, desc in empirical_patterns:
        m = re.search(pat, cleaned, re.IGNORECASE)
        if m:
            return False, f"Prohibited empirical hack ({desc}): '{m.group(0)}'"

    # 3. Problem-specific benchmark / problem number bindings
    problem_id_patterns = [
        r"\bproblem\s*#?\s*\d+\b",
        r"\bfrontiermath(?:\s*#?\s*\d*)?\b",
        r"\bmini_bench\b",
        r"\bpid\s*#?\s*\d+\b",
        r"\bproject\s*euler\b",
        r"\beuler\s*#\s*\d+\b",
    ]
    for pat in problem_id_patterns:
        m = re.search(pat, cleaned, re.IGNORECASE)
        if m:
            return False, f"Problem-specific benchmark identifier: '{m.group(0)}'"

    if problem_id is not None:
        pid_str = str(problem_id).strip()
        if pid_str and re.search(rf"\b(?:problem|pid|p)\s*#?\s*{re.escape(pid_str)}\b", cleaned, re.IGNORECASE):
            return False, f"Explicit reference to problem ID {pid_str}"

    # 4. Problem-specific title binding
    if problem_title:
        clean_title = problem_title.strip()
        if len(clean_title) > 6 and clean_title.lower() in cleaned.lower():
            return False, f"Explicit reference to problem title: '{clean_title}'"

    # 5. Problem-specific variable bindings (e.g. ord_p(2) > ord_p(3))
    var_binding_patterns = [
        r"\bord(?:_[a-zA-Z0-9]+)?\s*\(\s*\d+\s*\)\s*(?:>|<|==|!=|>=|<=)\s*ord(?:_[a-zA-Z0-9]+)?\s*\(\s*\d+\s*\)",
        r"\b2\s*\^\s*k\s*(?:==|=)\s*3\s*\^\s*m\b",
        r"\bfor\s+this\s+specific\s+problem\b",
        r"\bin\s+this\s+problem\b",
    ]
    for pat in var_binding_patterns:
        m = re.search(pat, cleaned, re.IGNORECASE)
        if m:
            return False, f"Problem-specific variable binding: '{m.group(0)}'"

    # 6. Ground-truth leakage post-filter (answer, statement constants, id/title)
    leak = find_ground_truth_leak(
        cleaned,
        expected_answer=expected_answer,
        problem_statement=problem_statement,
        problem_id=problem_id,
        problem_title=problem_title,
    )
    if leak:
        return False, f"Ground-truth leakage: {leak}"

    return True, "Valid universal mathematical heuristic"


def parse_benchmark_header(problem: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Parses ``'Project Euler Problem {id} ({title}):'`` style headers -> (problem_id, title)."""
    if not problem:
        return None, None
    first = problem.strip().splitlines()[0] if problem.strip() else ""
    m = re.match(r"^\s*(?:project\s+euler\s+)?problem\s*#?\s*(\d+)\s*(?:\(([^)]*)\))?", first, re.IGNORECASE)
    if not m:
        return None, None
    return m.group(1), (m.group(2).strip() if m.group(2) else None)


_BENCHMARK_PREFIX_RE = re.compile(
    r"^\s*(?:project\s+euler\s+|frontiermath\s+(?:tier\s*\d+\s+)?)?problem\s*#?\s*\d+\s*(?:\([^)]*\))?\s*[:\-\u2013]?\s*",
    re.IGNORECASE,
)


def strip_benchmark_prefix(problem: Optional[str]) -> str:
    """Removes a leading ``'Project Euler Problem {id} ({title}):'`` header (id and title) from text."""
    if not problem:
        return ""
    return _BENCHMARK_PREFIX_RE.sub("", problem, count=1).lstrip()


def normalize_id(value: Optional[Any]) -> str:
    """Canonical string form of a problem/task id for metadata storage and comparison."""
    if value is None:
        return ""
    return str(value).strip()


def iter_payload_tags(payload: Optional[dict], keys: Iterable[str]) -> Iterable[str]:
    """Yields normalized values for ``keys`` from a payload dict (and a nested 'metadata' dict)."""
    if not isinstance(payload, dict):
        return
    nested = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    for key in keys:
        for src in (payload, nested):
            if key in src and src[key] not in (None, ""):
                yield normalize_id(src[key])
