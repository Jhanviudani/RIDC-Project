#!/usr/bin/env python3
"""
Reads TXT files from Supabase Storage bucket 'scraped-sites',
parses them with an LLM, and upserts rows into the `program_parsed` table.

Requires:
  pip install supabase openai python-dotenv tiktoken

.env:
  SUPABASE_URL=...
  SUPABASE_SERVICE_ROLE_KEY=...
  OPENAI_API_KEY=...
  SUPABASE_BUCKET=scraped-sites  (optional; defaults to scraped-sites)
  OPENAI_EXTRACT_MODEL=gpt-4o-mini   (optional; default below)
  OPENAI_EMBED_MODEL=text-embedding-3-small
"""

import os
import re
import sys
import json
import hashlib
import time
import traceback
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI

# -------------------------
# Env & clients
# -------------------------
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
BUCKET = os.getenv("SUPABASE_BUCKET", "scraped-sites")

OPENAI_EXTRACT_MODEL = os.getenv("OPENAI_EXTRACT_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    print("❌ Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in .env", file=sys.stderr)
    sys.exit(1)
if not os.getenv("OPENAI_API_KEY"):
    print("❌ Missing OPENAI_API_KEY in .env", file=sys.stderr)
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
from openai import OpenAI
client = OpenAI()

MODEL_CANDIDATES = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"]

def chat_with_fallback(messages):
    last_err = None
    for m in MODEL_CANDIDATES:
        try:
            return client.chat.completions.create(model=m, messages=messages)
        except openai.BadRequestError as e:
            if "invalid model" in str(e).lower():
                last_err = e
                continue
            raise
    raise last_err or RuntimeError("No usable chat model found.")



# -------------------------
# Helpers
# -------------------------
def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()

URL_RE = re.compile(r'https?://[^\s)>\]"}]+', re.I)

def first_url(text: str) -> Optional[str]:
    m = URL_RE.search(text or "")
    return m.group(0) if m else None

def clean_basename(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").strip()

def parse_org_and_program_from_key(key: str) -> Tuple[str, str]:
    """
    Heuristic: 'Organization/Program Name.txt' or just 'Program.txt'
    """
    parts = key.split("/")
    fname = parts[-1]
    base = os.path.splitext(fname)[0]
    program = clean_basename(base)
    org = clean_basename(parts[-2]) if len(parts) >= 2 else ""
    # If org is empty, we'll let the LLM fill it.
    return org, program

def list_all_txt_keys(bucket: str) -> List[str]:
    """
    Walk the bucket recursively by listing each 'folder' level.
    Supabase storage list() is per 'path'. We simulate recursion.
    """
    keys: List[str] = []

    def walk(prefix: str = ""):
        # NOTE: storage.list returns both files and "folders" (as name ending with '/')
        # supabase-py returns dicts with keys: name, id, updated_at, created_at, last_accessed_at, metadata
        items = supabase.storage.from_(bucket).list(path=prefix)  # type: ignore
        for it in items:
            name = it.get("name")
            if not name:
                continue
            full = f"{prefix}{name}" if prefix == "" else f"{prefix}/{name}"
            # folder?
            if name.endswith("/"):
                walk(full.rstrip("/"))
            else:
                if full.lower().endswith(".txt"):
                    keys.append(full)

    walk("")
    return keys

def download_text(bucket: str, key: str) -> str:
    res = supabase.storage.from_(bucket).download(key)  # bytes
    return res.decode("utf-8", errors="ignore")



def llm_extract_fields(raw_text: str, org_hint: str, program_hint: str) -> Dict:
    """
    Ask LLM for normalized fields. Keep prompt short. Provide hints but allow correction.
    """
    snippet = raw_text[:6000]  # cap token use
    system = (
        "You are a strict information extractor for an entrepreneurship support directory. "
        "Return compact JSON only. If unknown, use null or empty list."
    )
    user = f"""


HINTS:
- organization_hint: {org_hint or "null"}
- program_name_hint: {program_hint or "null"}

TEXT (snippet):
\"\"\"{snippet}\"\"\"

Extract:
{{
  "program_name": string|null,
  "organization": string|null,
  "website": string|null,
  "address": string|null,
  "summary_gpt": string,               // one or two crisp sentences, max ~300 chars
  "topics": [string],                   // 5-8 lowercase tags (e.g., ["accelerator","robotics","funding"])
  "stages": [string]                    // subset of ["idea","early","growth","mature"]
}}

Rules:
- Prefer canonical/correct names over hints if the text clearly indicates otherwise.
- If multiple URLs appear, pick the primary program/org site (not social links).
- Keep tags simple, lowercase, no duplicates.
- If stage is implied (e.g., accelerator for early), include it.
Return ONLY JSON.
"""
    resp = client.chat.completions.create(
        model=OPENAI_EXTRACT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    txt = resp.choices[0].message.content.strip()

    # find json start (be resilient)
    start = min([i for i in [txt.find("{"), txt.find("[")] if i >= 0] or [0])
    j = txt[start:]
    data = json.loads(j)
    # schema guardrails
    for k in ["program_name", "organization", "website", "address", "summary_gpt"]:
        data.setdefault(k, None)
    for k in ["topics", "stages"]:
        v = data.get(k) or []
        if not isinstance(v, list): v = []
        # normalize to lower + dedupe
        seen = set()
        norm = []
        for t in v:
            if not isinstance(t, str): continue
            s = t.strip().lower()
            if s and s not in seen:
                seen.add(s)
                norm.append(s)
        data[k] = norm
    # Fallbacks if LLM omitted key facts
    if not data.get("website"):
        data["website"] = first_url(snippet)
    if not data.get("program_name"):
        data["program_name"] = program_hint or None
    if not data.get("organization"):
        data["organization"] = org_hint or None
    return data

def get_embedding(text: str) -> List[float]:
    emb = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=text)
    return emb.data[0].embedding

def upsert_program(row: Dict) -> None:
    # Supabase expects Postgres arrays for topics/stages; supabase-py will handle Python lists fine.
    supabase.table("program_parsed").upsert(
        row,
        on_conflict="program_name,organization"
    ).execute()

# -------------------------
# Main pipeline
# -------------------------
def process_key(key: str) -> Tuple[bool, str]:
    try:
        raw = download_text(BUCKET, key)
        if not raw.strip():
            return False, "empty file"

        org_hint, program_hint = parse_org_and_program_from_key(key)
        content_hash = sha256_text(raw)
        raw_excerpt = raw[:4000]

        # Extract fields via LLM
        info = llm_extract_fields(raw_excerpt, org_hint, program_hint)

        program_name = (info.get("program_name") or "").strip()
        organization = (info.get("organization") or "").strip()
        if not (program_name and organization):
            return False, "missing program_name or organization after extraction"
        
        
        # Summary for embedding (fallback to excerpt if summary missing)
        summary = (info.get("summary_gpt") or "").strip()
        embed_text = summary if summary else f"{program_name} {organization} {info.get('website') or ''}"

        # 1536-dim embedding
        embedding = get_embedding(embed_text)

        row = {
            "program_name": program_name,
            "organization": organization,
            "website": info.get("website"),
            "storage_path": f"{BUCKET}/{key}",
            "summary_gpt": summary,
            "topics": info.get("topics", []),
            "stages": info.get("stages", []),
            "embedding": embedding,
            "last_parsed": "now()",  # let Postgres set to now()
        }

        upsert_program(row)
        return True, f"upserted ({program_name} — {organization})"
    except Exception as e:
        return False, f"ERROR: {e}\n{traceback.format_exc()}"
    
    import os, re

def parse_names_from_object_name(obj_name: str):
    base = os.path.splitext(os.path.basename(obj_name))[0]
    # Split on " - " or " — "
    m = re.split(r"\s[-–—]\s", base, maxsplit=1)
    if len(m) == 2:
        prog, org = m[0].strip(), m[1].strip()
        # sanity: don't return empty strings
        prog = prog or None
        org  = org or None
        return prog, org
    return None, None

def main(limit: Optional[int] = None, offset: int = 0, sleep_secs: float = 0.5):
    print(f"📦 Bucket: {BUCKET}")
    keys = list_all_txt_keys(BUCKET)
    print(f"🔍 Found {len(keys)} .txt objects")

    if offset:
        keys = keys[offset:]
    if limit is not None:
        keys = keys[:limit]

    ok = 0
    fail = 0
    for i, key in enumerate(keys, 1):
        print(f"➡️  [{i}/{len(keys)}] {key}")
        success, msg = process_key(key)
        if success:
            ok += 1
            print(f"   ✅ {msg}")
        else:
            fail += 1
            print(f"   ❌ {msg}")
        time.sleep(sleep_secs)

    print("\n──────── Run summary ────────")
    print(f"✅ Upserted: {ok}")
    print(f"❌ Failed:   {fail}")

if __name__ == "__main__":
    # Usage: python parse_bucket_to_table.py [limit] [offset]
    # Example: python parse_bucket_to_table.py 25 0
    arg_limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    arg_offset = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    main(limit=arg_limit, offset=arg_offset)
