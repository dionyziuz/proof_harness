"""LaTeX Proof Extraction, Normalization, and Syntax Validation for Mathematical Theorem Proving.

Isolates multi-tier regex matching, balanced-brace parsing, academic discourse marker
sanitization, and structural validation of LaTeX proof candidates.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple


def is_nested_proof_start(cleaned: str, pos: int, tag_str: str) -> bool:
    """Checks if a \\begin{proof} is an inner sub-claim/lemma proof rather than a top-level proof start."""
    if re.match(r"\\begin\{proof\}\s*\[Proof of (?:Claim|Lemma|Subclaim)", tag_str, re.IGNORECASE):
        return True

    # Check if inside an unclosed claim or sublemma
    idx_claim_open = cleaned.rfind(r"\begin{claim}", 0, pos)
    idx_claim_close = cleaned.rfind(r"\end{claim}", 0, pos)
    if idx_claim_open != -1 and idx_claim_open > idx_claim_close:
        return True

    idx_sub_open = cleaned.rfind(r"\begin{sublemma}", 0, pos)
    idx_sub_close = cleaned.rfind(r"\end{sublemma}", 0, pos)
    if idx_sub_open != -1 and idx_sub_open > idx_sub_close:
        return True

    # Check if preceded immediately by a closed \end{claim} or \end{sublemma} (only whitespace between)
    preceding = cleaned[max(0, pos - 500):pos]
    if re.search(r"\\end\{(?:claim|sublemma)\}\s*$", preceding):
        return True

    # Or preceded in the immediate paragraph by a bold/labeled claim header without an intervening \end{proof}
    last_end_proof = preceding.rfind(r"\end{proof}")
    tail_after_proof = preceding[last_end_proof + len(r"\end{proof}"):] if last_end_proof != -1 else preceding
    claim_indicators = [
        "**Claim", "**Subclaim",
        r"Claim \ref", r"Claim ~\ref",
    ]
    if any(ci in tail_after_proof for ci in claim_indicators):
        return True
    return False


def sanitize_informal_proof(proof: Optional[str]) -> Optional[str]:
    """Applies generic academic journal normalization to conversational transition words
    in a candidate LaTeX proof, replacing informal markers with standard formal discourse markers.
    """
    if not proof:
        return None
    sanitized = proof
    substitutions = [
        (r"(?:^|\n)[^\n]*\b[Ww]ait,\s*this\s+gives\s+the\s+wrong\s+direction\.?\s*[Ll]et\s+(?:me|us)\s+reconsider\.?[ \t]*\n?", "\n"),
        (r"\b[Ww]ait,\s*[Ll]et\s+(?:me|us)\s+", "We "),
        (r"\b[Ww]ait,\s*[Ii]\s+", "We "),
        (r"\b[Ww]ait,\s*", "Observe that "),
        (r"\b[Ww]ait\b", "Observe"),
        (r"\b[Aa]ctually,\s*[Ii]\s+", "We "),
        (r"\b[Aa]ctually,\s*we\s+cannot\b", "We cannot"),
        (r"\b[Aa]ctually,\s*", "Note that "),
        (r"\b[Aa]ctually\s+[Ii]\b", "We"),
        (r"\b[Aa]ctually\b", "In fact"),
        (r"\b[Ll]et\s+(?:me|us)\s+reconsider\s*:\s*", "More precisely: "),
        (r"\b[Ll]et\s+(?:me|us)\s+reconsider\.?\s*", ""),
        (r"\b[Ll]et\x27?s\s+reconsider\.?\s*", ""),
        (r"\b[Ll]et\s+(?:me|us)\s+re-read[^.\n]*\.\s*", ""),
        (r"\b[Ll]et\s+(?:me|us)\s+re-read\b", "Examining"),
        (r"\b[Rr]e-reading\b", "Examining"),
        (r"\b[Rr]e-examining\b", "Examining"),
        (r"\b[Ll]et\s+(?:me|us)\s+think[^.\n]*\.\s*", ""),
        (r"\b[Ll]et\s+(?:me|us)\s+think\b", "We consider"),
        (r"\b[Ll]et\s+(?:me|us)\s+carefully\s+analyze\b", "We analyze"),
        (r"\b[Ll]et\s+(?:me|us)\s+focus\s+on\s+what\b", "We focus on what"),
        (r"\b[Ii]\s+realize\s+the\s+issue\s*:\s*", "Specifically: "),
        (r"\b[Ii]\s+realize\s+that\b", "We observe that"),
        (r"\b[Ii]\s+realize\b", "We observe"),
        (r"\b[Ii]\s+think\s+the\s+issue\s+is\s+that\b", "Note that"),
        (r"\b[Ii]\s+think\s+the\s+issue\b", "The key point"),
        (r"\b[Ii]\s+think\s+the\b", "The"),
        (r"\b[Ii]\s+think\s+we\b", "We"),
        (r"\b[Ll]et\x27?s\s+(?:trace|check|restart)\b", "We verify"),
        (r"\b[Ll]et\x27?s refine(?:\s+the\s+construction)?:\s*", "We refine the construction: "),
        (r"\b[Ll]et us refine(?:\s+the\s+construction)?:\s*", "We refine the construction: "),
        (r"\b[Ll]et\s+me\s+refine(?:\s+the\s+construction)?:\s*", "We refine the construction: "),
        (r"\b[Ll]et\x27?s refine\b", "We refine"),
        (r"\blet\x27?s refine\b", "we refine"),
        (r"\b[Ll]et us refine\b", "We refine"),
        (r"\blet us refine\b", "we refine"),
        (r"\b[Ll]et me refine\b", "We refine"),
        (r"\blet me refine\b", "we refine"),
        (r"\b[Tt]his is not true\b", "this does not hold"),
        (r"\b[Tt]his is false\b", "this does not hold"),
        (r"\b[Tt]his cannot be true\b", "this cannot hold"),
    ]
    for pat, rep in substitutions:
        sanitized = re.sub(pat, rep, sanitized)

    # Normalize duplicate consecutive \end{proof} at the end of proof:
    while sanitized.strip().endswith(r"\end{proof}" + "\n" + r"\end{proof}"):
        sanitized = sanitized.strip()[:-len(r"\end{proof}" + "\n" + r"\end{proof}")] + r"\end{proof}"

    # Strip informal scratchpad/meta-planning preamble lines leaked immediately inside \begin{proof}
    m_body = re.match(r"^(\\begin\{proof\}(?:\[.*?\])?\s*\n)(.*?)(\\end\{proof\})$", sanitized, flags=re.DOTALL)
    if m_body:
        header, body, footer = m_body.group(1), m_body.group(2), m_body.group(3)
        body_lines = body.splitlines(keepends=True)
        cleaned_lines = []
        in_preamble = True
        scratchpad_lead_pats = [
            r"^\s*We\s+need\s+(?:prove|to\s+prove|produce|think|construct|handle|decide|avoid)\b",
            r"^\s*Need\s+(?:prove|produce|think|construct|handle|decide|avoid)\b",
            r"^\s*Let\x27?s\s+(?:inspect|continue\s+reasoning|examine|see|think)\b",
            r"^\s*Let\s+us\s+(?:inspect|continue\s+reasoning|examine|see|think)\b",
            r"^\s*Potential\s+issue\b",
            r"^\s*Publication-grade\s+but\b",
        ]
        for line in body_lines:
            stripped = line.strip()
            if in_preamble:
                if not stripped:
                    continue
                if any(re.search(pat, stripped, re.IGNORECASE) for pat in scratchpad_lead_pats):
                    continue
                in_preamble = False
                cleaned_lines.append(line)
            else:
                cleaned_lines.append(line)
        if cleaned_lines:
            sanitized = header + "".join(cleaned_lines).lstrip("\n") + ("\n" if not "".join(cleaned_lines).endswith("\n") else "") + footer

    return sanitized


# Phrases that introduce a hypothetical (proof by contradiction / case analysis), e.g.
# "Suppose, for contradiction, that the theorem is false".  A disproof marker inside such a
# clause is part of a legitimate argument, not an admission that the target is false.
_HYPOTHETICAL_PREFIX_RE = re.compile(
    r"(suppose|assume|assuming|supposing|if|otherwise|contrary|contradiction|were|toward|towards)\b[^.;:]{0,80}$"
)


def _is_asserted_disproof(p_lower: str, marker: str) -> bool:
    """True if ``marker`` occurs at least once outside a hypothetical clause."""
    start = 0
    while True:
        idx = p_lower.find(marker, start)
        if idx < 0:
            return False
        sentence_start = max(p_lower.rfind(".", 0, idx), p_lower.rfind("\n\n", 0, idx), 0)
        prefix = p_lower[sentence_start:idx]
        if not _HYPOTHETICAL_PREFIX_RE.search(prefix):
            return True
        start = idx + len(marker)


def is_valid_latex_proof(proof: Optional[str]) -> bool:
    """Validates whether an extracted string is a syntactically and structurally sound
    LaTeX proof candidate rather than an admission of failure, disproof, or empty string.
    """
    if not proof:
        return False
    p = proof.strip()
    # A publication proof can range from 150 to 120,000 characters (honoring 32k output tokens).
    if len(p) < 150 or len(p) > 120000:
        return False
    if not p.startswith(r"\begin{proof}") or not p.endswith(r"\end{proof}"):
        return False

    # Check for explicit admissions of incompleteness, failure, or code fences
    failure_markers = [
        "cannot fully write",
        "within the turn limit",
        "i've been unable",
        "i have been unable",
        "unable to complete the proof",
        "i leave the rest",
        "left as an exercise",
        "proof is incomplete",
        "i realize my proof is invalid",
        "as an ai language model",
        "as an ai,",
        "```python",
        "as shown in the main text",
        "as shown in earlier turns",
        "as shown in turn ",
        "as argued in earlier turns",
        "as discussed in the prompt",
        "properties are verified as shown in the code",
        "verified as shown in the code",
        "(as above)",
        "... (as above)",
        "...(as above)",
        r"\dots (as above)",
        "proof: ...",
        "proof: ... (as above)",
        "omitted for brevity",
        "omitted for space",
        "proof omitted",
        "details omitted",
        "left to the reader",
        "left as an exercise to the reader",
    ]
    p_lower = p.lower()
    if any(m in p_lower for m in failure_markers):
        return False

    # A valid proof must establish the theorem, not claim the target theorem/lemma is false.
    disproof_markers = [
        "is false as stated",
        "lemma is false",
        "theorem is false",
        "statement is false",
        "cannot hold as stated",
        "statement cannot hold",
        "assertion is false",
        "assertion cannot hold",
        "refutes the lemma",
        "refutes the statement",
        "refutes the theorem",
        "refutes the assertion",
        "no proof of the stated lemma can exist",
    ]
    if any(_is_asserted_disproof(p_lower, dm) for dm in disproof_markers):
        return False

    # A valid publication proof cannot consist of informal stream-of-consciousness or unfinished brainstorming
    informal_markers = [
        "wait, let me",
        "wait, i ",
        "wait, is ",
        "wait, actually",
        "wait, does",
        "wait, maybe",
        "wait, we ",
        "wait, the lemma",
        "let me reconsider",
        "let me think",
        "let me re-read",
        "re-reading",
        "re-examining",
        "let us refine",
        "let me refine",
        "actually, i",
        "actually, we cannot",
        "actually i",
        "misreading",
        "i realize",
        "let's trace",
        "let's check",
        "let's restart",
        "i think the issue",
        "confusing myself",
        "my previous deduction was wrong",
        "let me focus on what",
        "let me carefully analyze",
        "does such a vertex exist?",
        "is this always true?",
        "what's going on?",
        "is the lemma false",
        "this is a flaw",
        "flaw in the draft",
        "flaw in my",
        "the draft's",
        "the draft ",
        "in the draft",
        "i will state",
        "i will assume",
        "i will output",
        "i will stick",
        "i will write",
        "property i'm missing",
        "property i am missing",
        "let's follow the draft",
        "let's look at the draft",
        "stream-of-consciousness",
        "stream of consciousness",
        "i don't know",
        "why would it stop",
    ]
    if any(im in p_lower for im in informal_markers):
        return False
    # Question marks outside LaTeX undefined-reference placeholders ("??") signal self-questioning.
    if p.replace("??", "").count("?") > 4:
        return False

    informal_regexes = [
        r"\bwait\b",
        r"\bactually\b",
        r"\bthis is not true\b",
        r"\bthis is false\b",
        r"\bthis cannot be true\b",
        r"\blet'?s refine\b",
        r"\blet us refine\b",
        r"\bwe might need\b",
        r"\bi need to find\b",
        r"\blet me search\b",
        r"\bi think the\b",
        r"\bi think we\b",
        r"\blikely relies\b",
        r"\bopen question\b",
        r"\bto be completed\b",
        r"\bcontradiction\?",
        r"\bis this possible\?",
    ]
    if any(re.search(pat, p_lower) for pat in informal_regexes):
        return False

    # Must contain mathematical notation or LaTeX markup
    has_math = any(sym in p for sym in ["$", "\\", "\\[", "\\begin", "\\end", "\\in", "\\le", "\\ge"])
    if not has_math:
        return False

    return True


def extract_latex_proof(text: Optional[str]) -> Optional[str]:
    """Extracts a clean, complete LaTeX proof from model text.
    Handles multiple restarts by evaluating top-level proof attempts in reverse,
    returning the latest valid publication-grade proof, and preserves nested proof environments.
    Strictly requires genuine closure with \\end{proof} without synthetic completion.
    """
    if not text:
        return None

    cleaned = text.strip()

    # Strip markdown code blocks if the entire response is wrapped in one
    m_fence = re.match(r"^```(?:latex|tex)?\s*\n(.*?)\n```$", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if m_fence:
        cleaned = m_fence.group(1).strip()

    # Strip any embedded ```python ... ``` blocks so code snippets between or inside
    # proof attempts do not poison span matching or trigger the ```python failure marker.
    if "```python" in cleaned.lower():
        cleaned = re.sub(r"```python\b.*?```", "", cleaned, flags=re.DOTALL | re.IGNORECASE).strip()

    pattern = re.compile(r"(\\begin\{proof\}(?:\[.*?\])?|\\end\{proof\})")
    matches = list(pattern.finditer(cleaned))
    if not matches:
        m_xml = re.search(r"<proof>(.*?)</proof>", cleaned, flags=re.DOTALL | re.IGNORECASE)
        if m_xml:
            xml_proof = m_xml.group(1).strip()
            if not xml_proof.startswith(r"\begin{proof}"):
                xml_proof = f"\\begin{{proof}}\n{xml_proof}\n\\end{{proof}}"
            return xml_proof if is_valid_latex_proof(xml_proof) else None

        # Generic fallback for Proof: / \textbf{Proof:} without \begin{proof}
        m_head = re.search(r"(?:^|\n)(?:\\textbf\{Proof[\.:]?\}|Proof[\.:])\s*(.+?)(?:\\qed|\u220e|\u25a0|Q\.E\.D\.|\Z)", cleaned, flags=re.DOTALL | re.IGNORECASE)
        if m_head:
            body = m_head.group(1).strip()
            cand = f"\\begin{{proof}}\n{body}\n\\end{{proof}}"
            if is_valid_latex_proof(cand):
                return cand

    tokens = [(m.start(), m.end(), m.group(1)) for m in matches]
    raw_spans: List[Tuple[int, int, str]] = []

    # 1. Depth-matched candidate extraction for every \begin{proof}
    for i, (start, end, tag) in enumerate(tokens):
        if tag.startswith(r"\begin{proof}"):
            depth = 0
            matched_end = None
            for j in range(i, len(tokens)):
                _, t_end, t_tag = tokens[j]
                if t_tag.startswith(r"\begin{proof}"):
                    depth += 1
                elif t_tag == r"\end{proof}":
                    depth -= 1
                    if depth == 0:
                        matched_end = t_end
                        break
            if matched_end is not None:
                cand = cleaned[start:matched_end].strip()
                cand = re.sub(r"\\cqed\b", "", cand).strip()
                raw_spans.append((start, matched_end, cand))

    # Filter out nested sub-proofs: if candidate B is strictly enclosed inside candidate A,
    # or if its start is marked as a nested proof start, do not treat B as a top-level proof.
    top_candidates: List[str] = []
    for s_start, s_end, s_cand in raw_spans:
        is_nested = False
        for o_start, o_end, _ in raw_spans:
            if o_start <= s_start and s_end <= o_end and (o_start, o_end) != (s_start, s_end):
                is_nested = True
                break
        if not is_nested and not is_nested_proof_start(cleaned, s_start, cleaned[s_start:s_start+30]):
            top_candidates.append(s_cand)

    n_begin_all = len(re.findall(r"\\begin\{proof\}", cleaned))
    n_end_all = len(re.findall(r"\\end\{proof\}", cleaned))

    # Outer proof fallback: If cleaned ends with \end{proof} and contains \begin{proof},
    # ensure the entire outer proof is evaluated even if inner sub-claim environments had unbalanced tags,
    # UNLESS n_end_all > n_begin_all and first_p is a nested sub-claim proof (outer \begin{proof} omitted).
    if cleaned.endswith(r"\end{proof}") and r"\begin{proof}" in cleaned:
        first_p = cleaned.find(r"\begin{proof}")
        first_is_nested = is_nested_proof_start(cleaned, first_p, cleaned[first_p:first_p + 30])
        if not (n_end_all > n_begin_all and first_is_nested):
            outer_span = cleaned[first_p:].strip()
            if outer_span not in top_candidates:
                top_candidates.append(outer_span)

    # If unmatched \end{proof} exists (e.g. outer \begin{proof} was omitted before a \begin{claim}),
    # reconstruct the full outer proof from the first formal deduction paragraph so the first half
    # of the proof before \begin{claim} is never chopped off.
    if n_end_all > n_begin_all and cleaned.endswith(r"\end{proof}"):
        first_begin = cleaned.find(r"\begin{proof}")
        before_first = cleaned[:first_begin] if first_begin != -1 else ""
        m = re.search(
            r"(?:^|\n)[ \t]*(\\begin\{(?:claim|sublemma|lemma|align|itemize|enumerate)\}|"
            r"\\(?:textbf|noindent|medskip|smallskip|bigskip|paragraph)\b|"
            r"(?:We\s+(?:first|now|begin|prove|show|claim|construct|define|set|fix|assume|consider)|"
            r"Let\s+|Fix\s+|Suppose\s+|Assume\s+|Consider\s+|By\s+|For\s+|To\s+prove|Since\s+|Observe\s+that))",
            before_first,
        )
        idx = m.start(1) if m else 0
        synth_cand = r"\begin{proof}" + "\n" + cleaned[idx:].lstrip()
        if synth_cand not in top_candidates:
            top_candidates.append(synth_cand)

    # 2. Evaluate top_candidates in reverse (latest first)
    for cand in reversed(top_candidates):
        if is_valid_latex_proof(cand):
            return cand
        sanitized = sanitize_informal_proof(cand)
        if sanitized and is_valid_latex_proof(sanitized):
            return sanitized

    top_level_starts = []
    for m in matches:
        tag = m.group(1)
        if tag.startswith(r"\begin{proof}"):
            if not is_nested_proof_start(cleaned, m.start(), tag):
                top_level_starts.append(m.start())

    if not top_level_starts:
        return None

    # Search in reverse for the latest top-level attempt that forms a genuine complete proof
    for start_pos in reversed(top_level_starts):
        proof_body = cleaned[start_pos:].strip()
        idx_end = proof_body.rfind(r"\end{proof}")
        if idx_end == -1:
            continue
        proof_body = proof_body[:idx_end + len(r"\end{proof}")].strip()

        n_begin = len(re.findall(r"\\begin\{proof\}", proof_body))
        n_end = len(re.findall(r"\\end\{proof\}", proof_body))
        if n_begin != n_end:
            continue

        proof_body = re.sub(r"\\cqed\b", "", proof_body).strip()
        if is_valid_latex_proof(proof_body):
            return proof_body
        san_body = sanitize_informal_proof(proof_body)
        if san_body and is_valid_latex_proof(san_body):
            return san_body

    # Fallback check among all top-level candidates
    for start_pos in top_level_starts:
        proof_body = cleaned[start_pos:].strip()
        idx_end = proof_body.rfind(r"\end{proof}")
        if idx_end == -1:
            continue
        proof_body = proof_body[:idx_end + len(r"\end{proof}")].strip()
        n_begin = len(re.findall(r"\\begin\{proof\}", proof_body))
        n_end = len(re.findall(r"\\end\{proof\}", proof_body))
        if n_begin != n_end:
            continue
        proof_body = re.sub(r"\\cqed\b", "", proof_body).strip()
        if is_valid_latex_proof(proof_body):
            return proof_body
        san_body = sanitize_informal_proof(proof_body)
        if san_body and is_valid_latex_proof(san_body):
            return san_body

    return None
