# functions.py
import json
import math
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import pgeocode
import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy import create_engine, text

import os
import pandas as pd
import pgeocode
import requests
import streamlit as st
import pydeck as pdk
import plotly.express as px
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from langchain_core.messages import HumanMessage
from math import radians, sin, cos, sqrt, atan2
from sqlalchemy.engine import URL





# -------- Secrets --------
def get_secret(key, fallback=None):
    """
    Retrieves a secret from Streamlit's secrets or environment variables.
    """
    try:
       return os.getenv(key) or st.secrets.get(key) or fallback
    except Exception:
        return os.getenv(key) or fallback

# --- add if missing ---
import json, re
from typing import Any, Dict, List

def match_programs_to_entrepreneur(entrepreneur: Dict[str, Any],
                                   providers_payload_json: str,
                                   model) -> str:
    """
    Ask the LLM to rank programs. It MUST return ONLY a JSON array (no prose).
    We pass provider+programs for both sources (providers & rolodex).
    """
    sys = (
        "You are a program-matching engine. "
        "Return ONLY a valid JSON array of matches. No prose, no markdown, no prefix/suffix."
    )
    user = f"""
ENTREPRENEUR:
{json.dumps(entrepreneur, ensure_ascii=False)}

PROVIDERS_AND_PROGRAMS_JSON:
{providers_payload_json}

Task:
1) Consider entrepreneur identity (growth_stage, vertical, county, zipcode, etc.) and needs (needs_needed).
2) Evaluate each provider's programs for fit (services, verticals/product_type, description).
3) Compute scores 0–10 for: distance_score, identity_score, service_score, need_satisfaction_score.
4) Add final_score = average of the four scores.
5) Choose the top ~15 programs overall.

Output format (array only):
[
  {{
    "provider_id": <string or number>,
    "provider_name": "<name>",
    "program_name": "<name>",
    "source": "providers" | "rolodex",
    "need_satisfied": "<short label>",
    "distance": <miles or null>,
    "distance_score": <0-10>,
    "identity_score": <0-10>,
    "service_score": <0-10>,
    "need_satisfaction_score": <0-10>,
    "final_score": <0-10>
  }},
  ...
]
IMPORTANT: Return ONLY the JSON array; no commentary or Markdown.
"""
    # Ask for JSON strictly; models supporting JSON mode will honor it,
    # others will still try to follow the instruction.
    try:
        resp = model.invoke(
            [
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ],
            # Works on modern OpenAI models; ignored by others
            model_kwargs={"response_format": {"type": "json_object"}}
        )
    except TypeError:
        # if model_kwargs not supported by this LangChain version
        resp = model.invoke(
            [
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ]
        )

    # LangChain ChatOpenAI returns an object with .content
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
        # Some models return {"matches":[...]}
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



# -------- DB Connection (Supabase/Postgres) --------
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
            "options": f"project={ref}",  # 👈 required for the pooler
        },
    )

    engine = create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 20})
    with engine.connect() as c:
        c.execute(text("SELECT 1"))
    return engine

# -------- Simple query helpers --------
def get_data(sql: str, engine) -> pd.DataFrame:
    return pd.read_sql(text(sql), engine)

def insert_data_to_supabase(df: pd.DataFrame, table: str) -> None:
    """Append DataFrame to a table. Adjust dtype mapping as needed."""
    eng = connect_db()
    df.to_sql(table, eng, if_exists="append", index=False, method="multi")

# -------- Visualization helpers --------
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

# -------- Geocoding / Distance --------
def add_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 'latitude' and 'longitude' from a 'zipcode' column.
    Leaves other rows untouched.
    """
    df = df.copy()
    if "zipcode" not in df.columns or df["zipcode"].isna().all():
        df["latitude"] = None
        df["longitude"] = None
        return df

    nomi = pgeocode.Nominatim("us")
    zips = pd.Series(df["zipcode"].astype(str).unique()).dropna()
    coords = nomi.query_postal_code(zips.tolist())
    lut = {str(z): (row.latitude, row.longitude) for z, row in zip(zips, coords.itertuples(index=False))}

    df["latitude"]  = df["zipcode"].astype(str).map(lambda z: lut.get(z, (None, None))[0])
    df["longitude"] = df["zipcode"].astype(str).map(lambda z: lut.get(z, (None, None))[1])
    return df

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

# -------- JSON nesting --------
def df_to_json_nest(parents: pd.DataFrame, children: pd.DataFrame, join_key: str, child_key: str) -> List[Dict[str, Any]]:
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

# -------- Providers + Rolodex (single payload for matching) --------
def build_matching_payload(engine, entrepreneur_zip: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Return a single list of providers (from BOTH the form tables and the rolodex),
    where each provider has a 'programs' list. Also add a 'source' and, when
    possible, a 'distance' field (miles) from the entrepreneur_zip.
    """
    # ---------- A) Providers (form flow) ----------
    q_prov = """
      SELECT DISTINCT ON (provider_id)
             provider_id, provider_name, address, description, zipcode, county, "BBB", programs_available
      FROM providers
      ORDER BY provider_id, date_intake_form DESC;
    """
    q_prog = """
      SELECT DISTINCT ON (provider_id, program_id)
             provider_id, program_id, program_name, website,
             contact_name, contact_email, services,
             CONCAT_WS(' - ',
               CONCAT('ALL: ',core_audience_all),
               CONCAT('Ecosystem Org: ',core_audience_ecosystem),
               CONCAT('Entrepreneur: ',core_audience_entrepreneur),
               CONCAT('Startups: ',core_audience_startups),
               CONCAT('SMEs/Companies: ',core_audience_sme),
               CONCAT('University Students: ',core_audience_ustudents),
               CONCAT('K-12 Students: ',core_audience_k12students)) AS core_audience,
             CONCAT_WS(' - ',
               CASE WHEN growth_stage_discovery = 1 THEN 'Discovery: Poorly suited'
                    WHEN growth_stage_discovery = 2 THEN 'Discovery: Somewhat suited'
                    WHEN growth_stage_discovery = 3 THEN 'Discovery: Moderately suited'
                    WHEN growth_stage_discovery = 4 THEN 'Discovery: Well suited'
                    WHEN growth_stage_discovery = 5 THEN 'Discovery: Perfectly suited'
                    ELSE 'Discovery: Not suited' END,
               CASE WHEN growth_stage_early = 1 THEN 'Early: Poorly suited'
                    WHEN growth_stage_early = 2 THEN 'Early: Somewhat suited'
                    WHEN growth_stage_early = 3 THEN 'Early: Moderately suited'
                    WHEN growth_stage_early = 4 THEN 'Early: Well suited'
                    WHEN growth_stage_early = 5 THEN 'Early: Perfectly suited'
                    ELSE 'Early: Not suited' END,
               CASE WHEN growth_stage_growth = 1 THEN 'Growth: Poorly suited'
                    WHEN growth_stage_growth = 2 THEN 'Growth: Somewhat suited'
                    WHEN growth_stage_growth = 3 THEN 'Growth: Moderately suited'
                    WHEN growth_stage_growth = 4 THEN 'Growth: Well suited'
                    WHEN growth_stage_growth = 5 THEN 'Growth: Perfectly suited'
                    ELSE 'Growth: Not suited' END,
               CASE WHEN growth_stage_mature = 1 THEN 'Mature: Poorly suited'
                    WHEN growth_stage_mature = 2 THEN 'Mature: Somewhat suited'
                    WHEN growth_stage_mature = 3 THEN 'Mature: Moderately suited'
                    WHEN growth_stage_mature = 4 THEN 'Mature: Well suited'
                    WHEN growth_stage_mature = 5 THEN 'Mature: Perfectly suited'
                    ELSE 'Mature: Not suited' END
             ) AS growth_stage,
             verticals, product_type, scraped_description
      FROM programs
      ORDER BY provider_id, program_id, date_intake_form DESC;
    """
    df_prov = get_data(q_prov, engine)
    df_prog = get_data(q_prog, engine)
    payload_prov = df_to_json_nest(df_prov, df_prog, "provider_id", "programs")
    for p in payload_prov:
        p["source"] = "providers"

    # ---------- B) Rolodex (synthetic providers/programs) ----------
    q_rolo = """
      SELECT
        org_name AS provider_name,
        program_name,
        COALESCE(website,'') AS website,
        COALESCE(full_description, description, '') AS scraped_description,
        COALESCE(primary_service,'') AS services,
        COALESCE(attributes->>'Vertical(s) Summary','')     AS verticals,
        COALESCE(attributes->>'Product Type(s) Summary','') AS product_type,
        COALESCE(county_hq,'') AS county,
        latitude, longitude
      FROM rolodex_points;
    """
    try:
        df_rolo = get_data(q_rolo, engine)
    except Exception:
        df_rolo = pd.DataFrame()

    payload_rolo: List[Dict[str, Any]] = []
    if not df_rolo.empty:
        # provider shells, one per org_name
        prov_rolo = (
            df_rolo[["provider_name", "county", "latitude", "longitude"]]
            .drop_duplicates()
            .assign(address="", description="", zipcode=None, BBB=None, programs_available=None)
            .reset_index(drop=True)
        )
        prov_rolo["provider_id"] = prov_rolo.index.map(lambda i: f"rolo-{i}")

        # map provider_name -> synthetic id
        id_map = dict(zip(prov_rolo["provider_name"], prov_rolo["provider_id"]))
        prog_rolo = df_rolo.copy().reset_index(drop=True)
        prog_rolo["provider_id"] = prog_rolo["provider_name"].map(id_map)
        prog_rolo["program_id"]  = prog_rolo.index.map(lambda i: f"rp-{i}")

        keep = ["provider_id","program_id","program_name","website","scraped_description","services","verticals","product_type"]
        payload_rolo = df_to_json_nest(
            prov_rolo[["provider_id","provider_name","address","description","zipcode","county","latitude","longitude"]],
            prog_rolo[keep],
            "provider_id",
            "programs"
        )
        for p in payload_rolo:
            p["source"] = "rolodex"

    merged = payload_prov + payload_rolo

    # ---------- C) Distance enrichment (optional) ----------
    if entrepreneur_zip:
        ent_lat = ent_lon = None
        try:
            nomi = pgeocode.Nominatim("us")
            rec = nomi.query_postal_code(str(entrepreneur_zip))
            if pd.notna(rec.latitude) and pd.notna(rec.longitude):
                ent_lat, ent_lon = float(rec.latitude), float(rec.longitude)
        except Exception:
            pass

        def _hav(lat, lon):
            try:
                if any(v is None for v in [ent_lat, ent_lon, lat, lon]):
                    return None
                φ1, λ1, φ2, λ2 = map(math.radians, [ent_lat, ent_lon, float(lat), float(lon)])
                dφ, dλ = (φ2 - φ1), (λ2 - λ1)
                a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
                return 3958.7613 * 2 * math.asin(math.sqrt(a))
            except Exception:
                return None

        for p in merged:
            if p.get("source") == "providers":
                z = p.get("zipcode")
                p["distance"] = estimate_zipcode_distance(entrepreneur_zip, z) if z else None
            else:
                p["distance"] = _hav(p.get("latitude"), p.get("longitude"))

    return merged

# -------- LLM helpers already used by your app --------
def match_programs_to_entrepreneur(entrepreneur_info: Dict[str, Any], programs_payload_json: str, model) -> str:
    """
    Calls the LLM to select & score programs. We keep it simple — the payload already
    includes Providers + Rolodex.
    """
    prompt = f"""
You are a program matching assistant. Consider the entrepreneur info and the full list of providers/programs (from BOTH 'providers' and 'rolodex').

Entrepreneur JSON:
{json.dumps(entrepreneur_info, ensure_ascii=False)}

All Providers + Programs JSON:
{programs_payload_json}

Task:
1) Select 5–12 best-matching programs overall.
2) Consider distance (if present), identity fit, service fit, and need satisfaction.
3) Return ONLY a JSON array of objects with these keys:
   ["provider_name","program_name","need_satisfied","distance_score","identity_score","service_score","need_satisfaction_score","source"]
   - Scores: 0–10
   - 'source' should be 'providers' or 'rolodex' (pass through from input if available).
"""
    resp = model.invoke(prompt)
    return resp.content if hasattr(resp, "content") else str(resp)

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

# --- imports this function relies on (add if missing) ---
from typing import Any, Dict, List, Optional
import math, pgeocode, pandas as pd

def build_matching_payload(engine, entrepreneur_zip: Optional[str] = None) -> List[Dict[str, Any]]:
    """Merge providers (form tables) + rolodex into one list of providers,
    each with a 'programs' list and optional 'distance' in miles."""
    # A) Providers (form intake)
    q_prov = """
      SELECT DISTINCT ON (provider_id)
             provider_id, provider_name, address, description, zipcode, county, "BBB", programs_available
      FROM providers
      ORDER BY provider_id, date_intake_form DESC;
    """
    q_prog = """
      SELECT DISTINCT ON (provider_id, program_id)
             provider_id, program_id, program_name, website,
             contact_name, contact_email, services,
             CONCAT_WS(' - ',
               CONCAT('ALL: ',core_audience_all),
               CONCAT('Ecosystem Org: ',core_audience_ecosystem),
               CONCAT('Entrepreneur: ',core_audience_entrepreneur),
               CONCAT('Startups: ',core_audience_startups),
               CONCAT('SMEs/Companies: ',core_audience_sme),
               CONCAT('University Students: ',core_audience_ustudents),
               CONCAT('K-12 Students: ',core_audience_k12students)) AS core_audience,
             CONCAT_WS(' - ',
               CASE WHEN growth_stage_discovery = 1 THEN 'Discovery: Poorly suited'
                    WHEN growth_stage_discovery = 2 THEN 'Discovery: Somewhat suited'
                    WHEN growth_stage_discovery = 3 THEN 'Discovery: Moderately suited'
                    WHEN growth_stage_discovery = 4 THEN 'Discovery: Well suited'
                    WHEN growth_stage_discovery = 5 THEN 'Discovery: Perfectly suited'
                    ELSE 'Discovery: Not suited' END,
               CASE WHEN growth_stage_early = 1 THEN 'Early: Poorly suited'
                    WHEN growth_stage_early = 2 THEN 'Early: Somewhat suited'
                    WHEN growth_stage_early = 3 THEN 'Early: Moderately suited'
                    WHEN growth_stage_early = 4 THEN 'Early: Well suited'
                    WHEN growth_stage_early = 5 THEN 'Early: Perfectly suited'
                    ELSE 'Early: Not suited' END,
               CASE WHEN growth_stage_growth = 1 THEN 'Growth: Poorly suited'
                    WHEN growth_stage_growth = 2 THEN 'Growth: Somewhat suited'
                    WHEN growth_stage_growth = 3 THEN 'Growth: Moderately suited'
                    WHEN growth_stage_growth = 4 THEN 'Growth: Well suited'
                    WHEN growth_stage_growth = 5 THEN 'Growth: Perfectly suited'
                    ELSE 'Growth: Not suited' END,
               CASE WHEN growth_stage_mature = 1 THEN 'Mature: Poorly suited'
                    WHEN growth_stage_mature = 2 THEN 'Mature: Somewhat suited'
                    WHEN growth_stage_mature = 3 THEN 'Mature: Moderately suited'
                    WHEN growth_stage_mature = 4 THEN 'Mature: Well suited'
                    WHEN growth_stage_mature = 5 THEN 'Mature: Perfectly suited'
                    ELSE 'Mature: Not suited' END
             ) AS growth_stage,
             verticals, product_type, scraped_description
      FROM programs
      ORDER BY provider_id, program_id, date_intake_form DESC;
    """
    df_prov = get_data(q_prov, engine)
    df_prog = get_data(q_prog, engine)
    payload_prov = df_to_json_nest(df_prov, df_prog, "provider_id", "programs")
    for p in payload_prov:
        p["source"] = "providers"

    # B) Rolodex (turn each org into a provider with its programs)
    q_rolo = """
      SELECT
        org_name AS provider_name,
        program_name,
        COALESCE(website,'') AS website,
        COALESCE(full_description, description, '') AS scraped_description,
        COALESCE(primary_service,'') AS services,
        COALESCE(attributes->>'Vertical(s) Summary','')     AS verticals,
        COALESCE(attributes->>'Product Type(s) Summary','') AS product_type,
        COALESCE(county_hq,'') AS county,
        latitude, longitude
      FROM rolodex_points;
    """
    try:
        df_rolo = get_data(q_rolo, engine)
    except Exception:
        df_rolo = pd.DataFrame()

    payload_rolo: List[Dict[str, Any]] = []
    if not df_rolo.empty:
        prov = (
            df_rolo[["provider_name","county","latitude","longitude"]]
            .drop_duplicates().assign(address="", description="", zipcode=None, BBB=None, programs_available=None)
            .reset_index(drop=True)
        )
        prov["provider_id"] = prov.index.map(lambda i: f"rolo-{i}")
        id_map = dict(zip(prov["provider_name"], prov["provider_id"]))

        progs = df_rolo.copy().reset_index(drop=True)
        progs["provider_id"] = progs["provider_name"].map(id_map)
        progs["program_id"]  = progs.index.map(lambda i: f"rp-{i}")

        keep = ["provider_id","program_id","program_name","website","scraped_description","services","verticals","product_type"]
        payload_rolo = df_to_json_nest(
            prov[["provider_id","provider_name","address","description","zipcode","county","latitude","longitude"]],
            progs[keep],
            "provider_id","programs"
        )
        for p in payload_rolo:
            p["source"] = "rolodex"

    merged = payload_prov + payload_rolo

    # C) Distance (optional)
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

        for p in merged:
            if p.get("source") == "providers":
                z = p.get("zipcode")
                p["distance"] = estimate_zipcode_distance(entrepreneur_zip, z) if z else None
            else:
                p["distance"] = hav(p.get("latitude"), p.get("longitude"))

    return merged

# --- Simple catalog + NL search + answer helpers ---

import re
import json
import pandas as pd
from typing import List, Dict, Any

def build_program_catalog(engine) -> pd.DataFrame:
    """
    Returns a unified catalog of programs from:
      - providers/programs (form intake)
      - rolodex_points (sheet)
    Columns: provider_name, program_name, services, verticals, product_type,
             county, address, website, scraped_description, source, provider_id, program_id
    """
    # A) FORM PROGRAMS
    q_form = """
    SELECT  DISTINCT ON (t2.provider_id, t2.program_id)
        t2.provider_id, t2.provider_name, t2.program_id, t2.program_name,
        t2.website, t2.contact_name, t2.contact_email,
        t1.county, t1.address,
        t2.services, t2.verticals, t2.product_type,
        t2.scraped_description
    FROM  programs t2
    INNER JOIN providers t1 ON t1.provider_id = t2.provider_id
    ORDER BY t2.provider_id, t2.program_id, t2.date_intake_form DESC;
    """
    df_form = get_data(q_form, engine).copy()
    if df_form.empty:
        df_form = pd.DataFrame(columns=[
            "provider_id","provider_name","program_id","program_name","website",
            "contact_name","contact_email","county","address",
            "services","verticals","product_type","scraped_description"
        ])
    df_form["source"] = "providers"

    # B) ROLODEX PROGRAMS
    q_rolo = """
    SELECT
      org_name         AS provider_name,
      program_name,
      COALESCE(website,'') AS website,
      ''               AS contact_name,
      ''               AS contact_email,
      COALESCE(county_hq,'') AS county,
      COALESCE(address,'')   AS address,
      COALESCE(primary_service,'') AS services,
      COALESCE(attributes->>'Vertical(s) Summary','')     AS verticals,
      COALESCE(attributes->>'Product Type(s) Summary','') AS product_type,
      COALESCE(full_description, description, '')         AS scraped_description,
      latitude, longitude
    FROM rolodex_points
    """
    try:
        df_rolo = get_data(q_rolo, engine).copy()
    except Exception:
        df_rolo = pd.DataFrame(columns=[
            "provider_name","program_name","website","contact_name","contact_email",
            "county","address","services","verticals","product_type","scraped_description",
            "latitude","longitude"
        ])

    if not df_rolo.empty:
        # Synthetic ids for rolodex
        df_rolo["provider_id"] = df_rolo["provider_name"].fillna("").apply(
            lambda s: - (abs(hash("rolo:"+s)) % 1_000_000_000)
        ).astype("int64")
        df_rolo["program_id"] = df_rolo.apply(
            lambda r: abs(hash(f"{r['provider_id']}::{r.get('program_name','')}")) % 1_000_000_000, axis=1
        ).astype("int64")
    else:
        df_rolo["provider_id"] = pd.Series(dtype="int64")
        df_rolo["program_id"]  = pd.Series(dtype="int64")
    df_rolo["source"] = "rolodex"

    # C) UNION
    keep = [
        "provider_id","provider_name","program_id","program_name","website",
        "county","address","services","verticals","product_type",
        "scraped_description","source"
    ]
    for c in keep:
        if c not in df_form.columns: df_form[c] = ""
        if c not in df_rolo.columns: df_rolo[c] = ""
    df_all = pd.concat([df_form[keep], df_rolo[keep]], ignore_index=True)
    return df_all


def nl_search_programs(engine, query: str, limit: int = 20) -> pd.DataFrame:
    """
    Lightweight keyword/hybrid search over the unified catalog.
    Returns top rows with a simple score (not shown) and trims to `limit`.
    """
    df = build_program_catalog(engine)
    if df.empty or not query:
        return df.head(0)

    q = query.lower()

    # Small synonym map to catch common intents
    synonyms = {
        "funding": ["funding","fund","capital","grant","grants","loan","loans","finance","financing",
                    "investor","investment","seed","pre-seed","angel","vc","venture"],
        "mentor": ["mentor","mentorship","coaching","advisor","advising","accelerator","incubator"],
        "prototype": ["prototype","prototyping","lab","equipment","maker","fabrication",
                      "testing","demonstration","pilot","hardware"],
        "agriculture": ["agri","agro","agtech","agritech","agriculture","farming","farm"],
    }

    # Terms from query + synonyms
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
                # Heavier weight for likely-intent words
                if kw in {"funding","fund","grant","grants","loan","vc","angel","investment"}:
                    score += 4
                elif kw in {"agri","agtech","agriculture","farming","farm"}:
                    score += 3
                else:
                    score += 1
        # Extra boosts for common combos
        if ("fund" in q or "grant" in q or "loan" in q) and (
            "fund" in text or "grant" in text or "loan" in text or "invest" in text
        ):
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
            "For each bullet include: program_name — provider_name — why it fits — county — website — source "
            "(providers|rolodex). If nothing fits, say so clearly."
        ))
    ]
    resp = model.invoke(messages)
    return getattr(resp, "content", str(resp))
