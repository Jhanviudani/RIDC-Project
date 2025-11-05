# functions.py
import os
import re
import json
import math
from typing import Any, Dict, List, Optional

import pandas as pd
import pgeocode
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL


# ---------------- Secrets ----------------
def get_secret(key: str, fallback=None):
    """
    Retrieves a secret from Streamlit's secrets or environment variables.
    """
    try:
        return os.getenv(key) or st.secrets.get(key) or fallback
    except Exception:
        return os.getenv(key) or fallback


# ---------------- DB Connection (Supabase/Postgres via pooler) ----------------
def connect_db():
    user = get_secret("SUPABASE_USER")
    pwd  = get_secret("SUPABASE_PASSWORD")
    host = get_secret("SUPABASE_HOST")
    db   = get_secret("SUPABASE_DB", "postgres")
    port = int(get_secret("SUPABASE_PORT", "6543"))
    ref  = get_secret("SUPABASE_PROJECT_REF")

    if not all([user, pwd, host, db, port, ref]):
        raise RuntimeError("Missing DB secrets. Need USER, PASSWORD, HOST, PORT, DB, PROJECT_REF.")

    url = URL.create(
        drivername="postgresql+psycopg",
        username=user,
        password=pwd,
        host=host,
        port=port,
        database=db,
        query={
            "sslmode": "require",
            "options": f"project={ref}",  # required by Supabase pooler
        },
    )
    engine = create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 20})
    with engine.connect() as c:
        c.execute(text("SELECT 1"))
    return engine


# ---------------- Simple query helpers ----------------
def get_data(sql: str, engine) -> pd.DataFrame:
    return pd.read_sql(text(sql), engine)


def insert_data_to_supabase(df: pd.DataFrame, table: str) -> None:
    """Append DataFrame to a table. Adjust dtype mapping as needed."""
    eng = connect_db()
    df.to_sql(table, eng, if_exists="append", index=False, method="multi")


# ---------------- Visualization helpers ----------------
def hex_to_rgb(h: str) -> List[int]:
    h = h.strip().lstrip("#")
    return [int(h[i:i+2], 16) for i in (0, 2, 4)]


def extract_unique_items(df: pd.DataFrame, col: str) -> List[str]:
    """
    Turn delimited strings into a unique, sorted list.
    e.g. 'A, B; C' -> ['A', 'B', 'C']
    """
    items: set = set()
    if col not in df.columns:
        return []
    for v in df[col].dropna().astype(str):
        for part in str(v).replace(";", ",").split(","):
            t = part.strip()
            if t:
                items.add(t)
    return sorted(items)


# ---------------- Geocoding / Distance ----------------
def _haversine_miles(lat1, lon1, lat2, lon2) -> Optional[float]:
    try:
        φ1, λ1, φ2, λ2 = map(math.radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
        dφ = φ2 - φ1
        dλ = λ2 - λ1
        a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
        return 3958.7613 * 2 * math.asin(math.sqrt(a))
    except Exception:
        return None


def estimate_zipcode_distance(zip1: Any, zip2: Any) -> Optional[float]:
    """Rough ZIP→ZIP distance using centroid lat/lon."""
    if not zip1 or not zip2:
        return None
    nomi = pgeocode.Nominatim("us")
    r1 = nomi.query_postal_code(str(zip1))
    r2 = nomi.query_postal_code(str(zip2))
    if pd.isna(r1.latitude) or pd.isna(r2.latitude):
        return None
    return _haversine_miles(r1.latitude, r1.longitude, r2.latitude, r2.longitude)


# ---------------- JSON nesting ----------------
def df_to_json_nest(
    parents: pd.DataFrame,
    children: pd.DataFrame,
    join_key: str,
    child_key: str
) -> List[Dict[str, Any]]:
    """
    Nest children under parents[join_key] into parent[child_key] list.
    """
    parents = parents.copy()
    children = children.copy()
    if join_key not in parents.columns:
        return []
    if join_key not in children.columns:
        parents[child_key] = [[] for _ in range(len(parents))]
        return parents.to_dict(orient="records")

    grouped = {k: g.drop(columns=[join_key]).to_dict(orient="records")
               for k, g in children.groupby(join_key)}
    out = []
    for _, row in parents.iterrows():
        key = row[join_key]
        d = row.to_dict()
        d[child_key] = grouped.get(key, [])
        out.append(d)
    return out


# ---------------- Catalog (ROL0DEX_EXTERNAL ONLY) ----------------
def build_program_catalog(engine) -> pd.DataFrame:
    """
    Catalog sourced exclusively from rolodex_external.
    Columns: provider_name, program_name, website, services, verticals, product_type,
             county, address, scraped_description, latitude, longitude,
             provider_id, program_id (synthetic)
    """
    q = """
    SELECT
      org_name                             AS provider_name,
      program_name,
      COALESCE(website,'')                 AS website,
      COALESCE(primary_service,'')         AS services,
      COALESCE(verticals_summary,'')       AS verticals,
      COALESCE(product_types_summary,'')   AS product_type,
      COALESCE(county_hq,'')               AS county,
      COALESCE(address,'')                 AS address,
      COALESCE(full_description, description, '') AS scraped_description,
      latitude, longitude
    FROM rolodex_external;
    """
    df = get_data(q, engine).copy()
    if df.empty:
        return pd.DataFrame(columns=[
            "provider_id","provider_name","program_id","program_name","website",
            "county","address","services","verticals","product_type",
            "scraped_description","latitude","longitude"
        ])

    # Synthetic ids (stable-ish)
    df["provider_id"] = df["provider_name"].fillna("").apply(
        lambda s: abs(hash("roloext:"+s)) % 1_000_000_000
    ).astype("int64")

    df["program_id"] = df.apply(
        lambda r: abs(hash(f"{r['provider_id']}::{r.get('program_name','')}")) % 1_000_000_000, axis=1
    ).astype("int64")

    keep = [
        "provider_id","provider_name","program_id","program_name","website",
        "county","address","services","verticals","product_type",
        "scraped_description","latitude","longitude"
    ]
    return df[keep]


def build_matching_payload(engine, entrepreneur_zip: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Return a list of providers from rolodex_external only, each with a 'programs' list.
    Adds 'distance' (miles) when entrepreneur_zip is provided.
    """
    df = build_program_catalog(engine)
    if df.empty:
        return []

    # Provider shells
    prov = (
        df[["provider_id","provider_name","county","address","latitude","longitude"]]
        .drop_duplicates()
        .assign(zipcode=None, description="")
        .reset_index(drop=True)
    )

    # Programs
    progs = df[[
        "provider_id","program_id","program_name","website",
        "scraped_description","services","verticals","product_type"
    ]].copy()

    payload = df_to_json_nest(
        parents=prov[["provider_id","provider_name","address","zipcode","county","latitude","longitude"]],
        children=progs,
        join_key="provider_id",
        child_key="programs"
    )

    # Distance enrichment (optional)
    if entrepreneur_zip:
        ent_lat = ent_lon = None
        try:
            rec = pgeocode.Nominatim("us").query_postal_code(str(entrepreneur_zip))
            if pd.notna(rec.latitude) and pd.notna(rec.longitude):
                ent_lat, ent_lon = float(rec.latitude), float(rec.longitude)
        except Exception:
            pass

        def hav(lat, lon):
            try:
                if any(v is None for v in [ent_lat, ent_lon, lat, lon]): return None
                φ1, λ1, φ2, λ2 = map(math.radians, [ent_lat, ent_lon, float(lat), float(lon)])
                dφ, dλ = (φ2-φ1), (λ2-λ1)
                a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
                return 3958.7613 * 2 * math.asin(math.sqrt(a))
            except Exception:
                return None

        for p in payload:
            p["distance"] = hav(p.get("latitude"), p.get("longitude"))

    return payload


# ---------------- LLM helpers ----------------
def match_programs_to_entrepreneur(
    entrepreneur_info: Dict[str, Any],
    programs_payload_json: str,
    model
) -> str:
    """
    Calls the LLM to select & score programs (rolodex_external-only payload).
    """
    prompt = f"""
You are a program matching assistant. Consider the entrepreneur info and the list of providers/programs.

Entrepreneur JSON:
{json.dumps(entrepreneur_info, ensure_ascii=False)}

Providers + Programs JSON:
{programs_payload_json}

Task:
1) Select 5–12 best-matching programs overall.
2) Consider distance (if present), identity fit, service fit, and need satisfaction.
3) Return ONLY a JSON array of objects with these keys:
   ["provider_id","provider_name","program_name","need_satisfied","distance","distance_score","identity_score","service_score","need_satisfaction_score","final_score"]

Notes:
- Scores: 0–10 (floats allowed).
- If final_score is missing, we'll compute it as the average of the four sub-scores.
- Do NOT include any prose or markdown; return only a JSON array.
"""
    try:
        resp = model.invoke(
            [
                {"role": "system", "content": "Return only JSON arrays. No prose."},
                {"role": "user", "content": prompt},
            ],
            model_kwargs={"response_format": {"type": "json_object"}}
        )
    except TypeError:
        resp = model.invoke(
            [
                {"role": "system", "content": "Return only JSON arrays. No prose."},
                {"role": "user", "content": prompt},
            ]
        )
    return getattr(resp, "content", str(resp))


def coerce_json_array(text: str) -> List[Dict[str, Any]]:
    """
    Parse LLM output into a Python list. If the model returned extra prose,
    extract the first JSON array block safely.
    """
    if not text:
        raise ValueError("Empty model response")

    # Perfect case
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and "matches" in obj and isinstance(obj["matches"], list):
            return obj["matches"]
    except Exception:
        pass

    # Extract first [...] block
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception as e:
            raise ValueError(f"Found JSON-looking array but failed to parse: {e}")

    raise ValueError("No JSON array found in model output")


def summarize_user_identity_and_needs(entrepreneur_info: Dict[str, Any], model) -> str:
    prompt = f"""
Summarize (1 short paragraph) the entrepreneur identity, location, vertical, growth stage,
and key needs based on this JSON:
{json.dumps(entrepreneur_info, ensure_ascii=False)}
"""
    resp = model.invoke(prompt)
    return resp.content if hasattr(resp, "content") else str(resp)


def summarize_recommendations(entrepreneur_info: Dict[str, Any], matches: List[Dict[str, Any]], model) -> str:
    prompt = f"""
Write a short narrative (4–6 sentences) explaining why the top programs fit this entrepreneur's needs.
Entrepreneur:
{json.dumps(entrepreneur_info, ensure_ascii=False)}

Matches:
{json.dumps(matches[:10], ensure_ascii=False)}
"""
    resp = model.invoke(prompt)
    return resp.content if hasattr(resp, "content") else str(resp)


# ---------------- NL search + answer (ROL0DEX_EXTERNAL ONLY) ----------------
def nl_search_programs(engine, query: str, limit: int = 20) -> pd.DataFrame:
    """
    Keyword/hybrid search over the rolodex_external-only catalog.
    """
    df = build_program_catalog(engine)
    if df.empty or not query:
        return df.head(0)

    q = query.lower()
    synonyms = {
        "funding": ["funding","fund","capital","grant","grants","loan","loans","finance","financing",
                    "investor","investment","seed","pre-seed","angel","vc","venture"],
        "mentor": ["mentor","mentorship","coaching","advisor","advising","accelerator","incubator"],
        "prototype": ["prototype","prototyping","lab","equipment","maker","fabrication","testing","demonstration","pilot","hardware"],
        "agriculture": ["agri","agro","agtech","agritech","agriculture","farming","farm"],
    }
    tokens = [t for t in re.split(r"[^a-z0-9]+", q) if t]
    keywords = set(tokens)
    for base, alts in synonyms.items():
        if base in q:
            keywords.update(alts)

    def score_row(row: pd.Series) -> int:
        text = " ".join(
            str(row.get(c, "")) for c in
            ["program_name","provider_name","services","verticals","product_type","scraped_description","county"]
        ).lower()
        score = 0
        for kw in keywords:
            if kw and kw in text:
                if kw in {"funding","fund","grant","grants","loan","vc","angel","investment"}:
                    score += 4
                elif kw in {"agri","agtech","agriculture","farming","farm"}:
                    score += 3
                else:
                    score += 1
        if ("fund" in q or "grant" in q or "loan" in q) and (
            "fund" in text or "grant" in text or "loan" in text or "invest" in text):
            score += 5
        if ("ag" in q) and ("agriculture" in text or "agtech" in text or "farm" in text):
            score += 4
        return score

    df = df.copy()
    df["__score"] = df.apply(score_row, axis=1)
    df = df[df["__score"] > 0].sort_values("__score", ascending=False).head(limit)
    return df.drop(columns=["__score"], errors="ignore")


def answer_query_over_catalog(model, user_query: str, records_df: pd.DataFrame) -> str:
    """
    Have the LLM write a concise, actionable answer using ONLY the provided records.
    (No internal source labels.)
    """
    from langchain_core.messages import SystemMessage, HumanMessage  # lazy import
    records = records_df.fillna("").to_dict(orient="records")
    messages = [
        SystemMessage(content=(
            "You recommend entrepreneurship programs/providers using ONLY the given records. "
            "Prefer highly relevant, concrete matches. Keep answers brief and actionable."
        )),
        HumanMessage(content=(
            "User query:\n"
            f"{user_query}\n\n"
            "Records (JSON list of programs/providers):\n"
            f"{json.dumps(records, ensure_ascii=False)}\n\n"
            "Produce a short answer followed by a bullet list of 5–10 top matches. "
            "For each bullet include: program_name — provider_name — why it fits — county — website. "
            "Do not mention any internal sources."
        ))
    ]
    resp = model.invoke(messages)
    return getattr(resp, "content", str(resp))
