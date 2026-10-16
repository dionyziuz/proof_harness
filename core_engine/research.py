"""Mathematical Research & Literature Retrieval Suite for Prime Math.

Empowers Prime Math with external research capabilities:
1. OEIS Sequence & Formula Lookup (`oeis_lookup`): Queries the On-Line Encyclopedia of
   Integer Sequences for closed-form recurrences, generating functions, and Python code.
2. Jina Mathematical Web Search (`search_math`): Searches mathematical literature, MathWorld,
   Wikipedia, and lecture notes via Jina AI Search.
3. arXiv Paper & Literature Reader (`read_paper`): Retrieves and renders arXiv papers and
   academic articles directly into clean Markdown via Jina Reader.
"""

from __future__ import annotations

import logging
import re
import urllib.parse
from typing import Any, Dict, List, Sequence, Union

import httpx

from .config import load_config

logger = logging.getLogger("prime_math.research")


def _get_jina_key() -> str:
    cfg = load_config()
    return cfg.jina_api_key or ""


def oeis_lookup(
    query: Union[str, Sequence[int]],
    max_results: int = 3,
    timeout: float = 20.0,
) -> List[Dict[str, Any]]:
    """Looks up sequences and formulas in the On-Line Encyclopedia of Integer Sequences (OEIS).

    Args:
        query: Sequence of integers (e.g. [1, 2, 4, 8, 16] or (1, 17, 98, 354)) or a search string.
        max_results: Maximum number of candidate sequences to return.
        timeout: HTTP request timeout in seconds.

    Returns:
        List of structured dictionaries with keys:
        - id: Sequence identifier (e.g. 'A000045')
        - name: Description / title of the sequence
        - terms: First terms of the sequence
        - formulas: List of algebraic formulas, recurrences, and generating functions
        - comments: Combinatorial interpretations and properties
        - programs: Implementations in Python, Mathematica, or PARI/GP
    """
    if isinstance(query, (list, tuple)):
        query_str = ",".join(str(int(x)) for x in query)
    else:
        query_str = str(query).strip()

    if not query_str:
        return []

    url = f"https://oeis.org/search?q={urllib.parse.quote(query_str)}&fmt=json"
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) PrimeMath/1.0"}

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and data:
                    results = []
                    for item in data[:max_results]:
                        seq_num = item.get("number")
                        seq_id = f"A{str(seq_num).zfill(6)}" if seq_num is not None else "A?"
                        results.append({
                            "id": seq_id,
                            "name": item.get("name", ""),
                            "terms": item.get("data", ""),
                            "formulas": item.get("formula", [])[:6],
                            "comments": item.get("comment", [])[:3],
                            "programs": item.get("program", [])[:3],
                            "url": f"https://oeis.org/{seq_id}",
                        })
                    return results
    except Exception as e:
        logger.warning(f"Direct OEIS API query failed: {e}; falling back to Jina search...")

    # Fallback to Jina Search scoped to oeis.org
    try:
        fallback_query = f"site:oeis.org {query_str}"
        jina_results = search_math(fallback_query, count=max_results, timeout=timeout)
        parsed_results = []
        for res in jina_results:
            m = re.search(r"A\d{6}", res.get("url", "") + " " + res.get("title", ""))
            seq_id = m.group(0) if m else "OEIS Result"
            parsed_results.append({
                "id": seq_id,
                "name": res.get("title", ""),
                "terms": "",
                "formulas": [res.get("snippet", "")],
                "comments": [],
                "programs": [],
                "url": res.get("url", ""),
            })
        return parsed_results
    except Exception as e:
        logger.warning(f"OEIS fallback search failed: {e}")
        return []


def clean_search_query(query: str, max_chars: int = 80) -> str:
    """Sanitizes and bounds a mathematical search query to prevent HTTP 422 errors and noise."""
    if not query:
        return ""

    # Strip XML/HTML tags
    text = re.sub(r"<[^>]+>", " ", query)

    # Normalize common LaTeX math terminology to standard English keywords
    replacements = [
        (r"(?i)\\operatorname\{ord\}|\\mathrm\{ord\}|ord_p|ord", "order"),
        (r"(?i)\\pi\s*\([^\)]*\)", "prime counting"),
        (r"(?i)\\gcd", "gcd"),
        (r"(?i)\\lcm", "lcm"),
        (r"(?i)\\zeta", "zeta"),
        (r"(?i)v_p|v_q", "valuation"),
        (r"(?i)\\pmod|\\mod", "modulo"),
        (r"(?i)\\le|\\ge|<=|>=", " "),
        (r"[<>]", " "),
    ]
    for pat, rep in replacements:
        text = re.sub(pat, rep, text)

    # Strip conversational filler prefixes
    filler_patterns = [
        r"^(we need|i need|user asks|user request|problem statement|problem|query|search query|here are|let's|calculate|construct|find)[\:\-\s]+",
        r"^to solve (this|the) problem[\:\-\s]+",
    ]
    text_clean = text.strip()
    for pat in filler_patterns:
        text_clean = re.sub(pat, "", text_clean, flags=re.IGNORECASE).strip()

    # Strip characters that break URL paths or query formats
    text_clean = re.sub(r"[{}\[\]\\\/\|\?\<\>\;\:\@\#\$\%\^\&\*\=\~`\"']+", " ", text_clean)

    # Collapse whitespace
    text_clean = " ".join(text_clean.split()).strip()

    # Bound length cleanly at word boundary
    if len(text_clean) > max_chars:
        text_clean = text_clean[:max_chars].rsplit(" ", 1)[0].strip()

    return text_clean


def search_wikipedia_math(
    query: str,
    count: int = 3,
    timeout: float = 10.0,
) -> List[Dict[str, str]]:
    """Authoritative zero-auth fallback search for mathematical concepts via Wikipedia API."""
    clean_q = clean_search_query(query)
    if not clean_q or len(clean_q) < 3:
        return []

    try:
        url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={urllib.parse.quote(clean_q)}&limit={count}&namespace=0&format=json"
        headers = {"User-Agent": "PrimeMath/1.0 (academic research)"}
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            # data format: [search_term, [titles], [descriptions], [urls]]
            if len(data) >= 4:
                titles = data[1]
                urls = data[3]
                snippets = []
                for title, page_url in zip(titles[:count], urls[:count]):
                    try:
                        summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
                        s_resp = client.get(summary_url, headers=headers, timeout=5.0)
                        if s_resp.status_code == 200:
                            s_data = s_resp.json()
                            extract = s_data.get("extract", "")
                            if extract and len(extract.strip()) > 30:
                                snippets.append({
                                    "title": f"Wikipedia: {title}",
                                    "url": page_url,
                                    "snippet": extract[:800],
                                })
                    except Exception:
                        continue
                return snippets
    except Exception as e:
        logger.debug(f"Wikipedia math fallback lookup hit exception: {e}")

    return []


def search_math(
    query: str,
    count: int = 3,
    timeout: float = 35.0,
    prioritize_arxiv: bool = True,
) -> List[Dict[str, str]]:
    """Performs mathematical research search with arXiv prioritization and Wikipedia fallback.

    Args:
        query: Mathematical search query (e.g. 'cuboid layer formula', 'Artin symbol Chebotarev density').
        count: Maximum number of search results to return.
        timeout: HTTP request timeout in seconds.
        prioritize_arxiv: Whether to prioritize academic preprints on arXiv before general web search.

    Returns:
        List of dicts with 'title', 'url', and 'snippet'.
    """
    clean_q = clean_search_query(query)
    if not clean_q or len(clean_q) < 3:
        return []

    jina_key = _get_jina_key()
    headers = {
        "Accept": "application/json",
        "X-Retain-Images": "none",
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) PrimeMath/1.0",
    }
    if jina_key:
        headers["Authorization"] = f"Bearer {jina_key}"

    # Step 1: Prioritize arXiv academic preprints (clean, zero captchas, state-of-the-art research)
    if prioritize_arxiv and not clean_q.startswith("site:"):
        arxiv_query = f"site:arxiv.org {clean_q}"
        arxiv_url = f"https://s.jina.ai/{urllib.parse.quote(arxiv_query)}"
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.get(arxiv_url, headers=headers)
                if resp.status_code == 200:
                    raw_results = resp.json().get("data", [])
                    arxiv_snippets = []
                    for r in raw_results:
                        title = r.get("title", "arXiv Paper")
                        url_str = r.get("url", "")
                        content_raw = r.get("content", "") or ""
                        desc_raw = r.get("description", "") or ""
                        combined = f"{desc_raw} {content_raw}".lower()

                        if any(k in combined for k in [
                            "verify you are human", "enable javascript", "access denied", "cloudflare"
                        ]):
                            continue

                        text_src = content_raw if len(content_raw.strip()) > len(desc_raw.strip()) else desc_raw
                        text_clean = " ".join(text_src.split())
                        if not text_clean or len(text_clean) < 30:
                            continue

                        arxiv_snippets.append({
                            "title": f"arXiv: {title}",
                            "url": url_str,
                            "snippet": text_clean[:800],
                        })
                        if len(arxiv_snippets) >= count:
                            break

                    if arxiv_snippets:
                        return arxiv_snippets
        except Exception as e:
            logger.debug(f"arXiv prioritized search failed: {e}")

    # Step 2: Open Jina Web Search
    url = f"https://s.jina.ai/{urllib.parse.quote(clean_q)}"
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url, headers=headers)
            if resp.status_code == 200:
                raw_results = resp.json().get("data", [])
                snippets = []
                for r in raw_results:
                    title = r.get("title", "Math Result")
                    url_str = r.get("url", "")
                    content_raw = r.get("content", "") or ""
                    desc_raw = r.get("description", "") or ""
                    combined = f"{desc_raw} {content_raw}".lower()

                    # Filter web noise & security challenges
                    if any(k in combined for k in [
                        "verify you are human", "enable javascript", "access denied", "cloudflare"
                    ]):
                        continue

                    text_src = content_raw if len(content_raw.strip()) > len(desc_raw.strip()) else desc_raw
                    text_clean = " ".join(text_src.split())
                    if not text_clean or len(text_clean) < 30:
                        continue

                    snippets.append({
                        "title": title,
                        "url": url_str,
                        "snippet": text_clean[:800],
                    })
                    if len(snippets) >= count:
                        break

                if snippets:
                    return snippets
    except httpx.HTTPStatusError as e:
        logger.warning(f"Mathematical web search HTTP status {e.response.status_code} for '{clean_q}': {e}")
    except httpx.TimeoutException:
        logger.warning(f"Mathematical web search timed out ({timeout}s) for '{clean_q}'")
    except Exception as e:
        logger.warning(f"Mathematical web search encountered error for '{clean_q}': {e}")

    # Step 3: Fallback to authoritative Wikipedia mathematical reference API
    logger.info(f"Falling back to Wikipedia mathematical lookup for '{clean_q}'...")
    return search_wikipedia_math(clean_q, count=min(count, 3), timeout=10.0)


def read_paper(
    url_or_id: str,
    max_chars: int = 8000,
    timeout: float = 25.0,
) -> str:
    """Reads research papers, arXiv articles, and mathematical pages via Jina Reader (r.jina.ai).

    Args:
        url_or_id: arXiv identifier (e.g. '2305.07394', 'math.NT/0501234') or direct HTTP/HTTPS URL.
        max_chars: Maximum characters of markdown content to return.
        timeout: HTTP request timeout in seconds.

    Returns:
        Clean markdown text extracted from the document.
    """
    target = url_or_id.strip()
    if not target:
        return "No URL or arXiv ID provided."

    # Normalize arXiv identifiers
    if re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", target):
        target = f"https://arxiv.org/abs/{target}"
    elif target.lower().startswith("arxiv:"):
        target = f"https://arxiv.org/abs/{target[6:]}"
    elif not target.startswith("http://") and not target.startswith("https://"):
        target = f"https://{target}"

    jina_key = _get_jina_key()
    reader_url = f"https://r.jina.ai/{target}"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) PrimeMath/1.0",
    }
    if jina_key:
        headers["Authorization"] = f"Bearer {jina_key}"

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(reader_url, headers=headers)
            resp.raise_for_status()
            text = resp.text.strip()
            if len(text) > max_chars:
                return text[:max_chars] + f"\n\n[... content truncated at {max_chars} chars ...]"
            return text
    except Exception as e:
        logger.warning(f"Failed to read paper from {target}: {e}")
        return f"Error reading paper: {e}"
