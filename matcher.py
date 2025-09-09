# matcher.py
import json
import os
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

try:
    from bs4 import BeautifulSoup
    HAVE_BS4 = True
except Exception:
    HAVE_BS4 = False

SUMMARY_CHAR_LIMIT = 400
RANK_RATIONALE_WORDS = 35
MODEL = os.getenv("MATCHER_MODEL", "gpt-4o-mini")

def _get_openai_client():
    from openai import OpenAI  # modern SDK
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    return OpenAI(api_key=api_key)

def _fetch_text(url: str) -> str:
    if not url or not url.startswith("http"):
        return ""
    try:
        r = requests.get(url, timeout=12, headers={"User-Agent": "RolodexMatcher/1.0"})
        r.raise_for_status()
        html = r.text
        if HAVE_BS4:
            soup = BeautifulSoup(html, "html.parser")
            for t in soup(["script", "style", "noscript", "iframe"]):
                try: t.decompose()
                except Exception: pass
            return soup.get_text("\n", strip=True)[:5000]
        return html[:5000]
    except Exception:
        return ""

def _program_long_text(row: Dict[str, Any]) -> str:
    desc = (row.get("scraped_description") or "").strip()
    if desc:
        return desc[:5000]
    fetched = _fetch_text((row.get("website") or "").strip())
    if fetched:
        return fetched
    parts = []
    for k in ("services", "verticals", "product_type"):
        v = (row.get(k) or "").strip()
        if v: parts.append(f"{k}: {v}")
    return " | ".join(parts)[:1500] if parts else "General support program."

def summarize_and_score_from_payload(payload_json: str, founder_needs_text: str,
                                     per_source_limit: Optional[int] = None,
                                     sort_by_distance: bool = True) -> pd.DataFrame:
    """
    payload_json: JSON list of providers; each has 'programs' and 'source'
    Returns a DataFrame with per-program scores.
    """
    providers: List[Dict[str, Any]] = json.loads(payload_json)

    # pick nearest N per source if asked
    def _dist(p):
        try: return float(p.get("distance"))
        except Exception: return float("inf")

    buckets = {"providers": [], "rolodex": []}
    for p in providers:
        buckets.setdefault(p.get("source","providers"), []).append(p)

    if sort_by_distance:
        for k in buckets:
            buckets[k].sort(key=_dist)

    selected: List[Dict[str, Any]] = []
    for k, items in buckets.items():
        selected.extend(items[:per_source_limit] if per_source_limit else items)

    rows: List[Dict[str, Any]] = []
    for p in selected:
        for g in p.get("programs", []):
            rows.append({
                "provider_name": p.get("provider_name",""),
                "Program Name":  g.get("program_name") or (p.get("provider_name","") + " — Program"),
                "website": g.get("website",""),
                "scraped_description": g.get("scraped_description",""),
                "services": g.get("services",""),
                "verticals": g.get("verticals",""),
                "product_type": g.get("product_type",""),
                "source": p.get("source","providers"),
                "distance": p.get("distance"),
            })
    if not rows:
        return pd.DataFrame()

    client = _get_openai_client()

    summaries, scores, rationales, dist_scores = [], [], [], []
    for r in rows:
        long_text = _program_long_text(r)
        prompt_sum = f"""Founder Needs:\n{founder_needs_text}\n\nProgram:\n{r['Program Name']}\n\nText:\n{long_text}\n\nWrite a 1–2 sentence summary (≤{SUMMARY_CHAR_LIMIT} chars)."""
        s = client.chat.completions.create(
            model=MODEL, temperature=0.3,
            messages=[{"role":"system","content":"You summarize startup support programs."},
                      {"role":"user","content":prompt_sum}]
        ).choices[0].message.content.strip()[:SUMMARY_CHAR_LIMIT]
        summaries.append(s)

        prompt_score = f"""Founder Needs:\n{founder_needs_text}\n\nProgram: {r['Program Name']}\nSummary: {s}\n\nReturn ONLY JSON: {{"relevance":0-100,"stage_fit":0-100,"overall":float,"rationale":"≤{RANK_RATIONALE_WORDS} words"}}"""
        try:
            sc = client.chat.completions.create(
                model=MODEL, temperature=0.15,
                messages=[{"role":"system","content":"You score startup support programs."},
                          {"role":"user","content":prompt_score}]
            ).choices[0].message.content.strip()
            data = json.loads(sc)
            overall = float(data.get("overall", 0.0))
            rationale = str(data.get("rationale", "")).strip()[:200]
        except Exception:
            overall, rationale = 0.0, "Scoring failed."

        scores.append(overall)
        rationales.append(rationale)

        d = r.get("distance")
        try:
            d = float(d) if d is not None else None
        except Exception:
            d = None
        dist_scores.append(50.0 if d is None else max(0.0, 100.0 - min(100.0, d)))

        time.sleep(0.5)

    df = pd.DataFrame(rows)
    df["GPT Summary"] = summaries
    df["Score"] = scores
    df["Rationale"] = rationales
    df["distance_score"] = dist_scores
    return df.sort_values(["Score","distance_score"], ascending=[False, False]).reset_index(drop=True)
