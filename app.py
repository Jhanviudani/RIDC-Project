import os
import re
import time
from datetime import datetime, timezone
from typing import List, Dict, Any

import pandas as pd
import streamlit as st
from supabase import create_client, Client

# -----------------------------
# Config & Setup
# -----------------------------
st.set_page_config(page_title="Founder → Program Matcher", page_icon="🤝", layout="wide")

# Read Supabase creds from Streamlit secrets or env
SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL", ""))
SUPABASE_ANON_KEY = st.secrets.get("SUPABASE_ANON_KEY", os.getenv("SUPABASE_ANON_KEY", ""))

@st.cache_resource(show_spinner=False)
def get_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        st.error("Missing Supabase credentials. Add SUPABASE_URL and SUPABASE_ANON_KEY to .streamlit/secrets.toml or environment.")
        st.stop()
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

sb = get_client()

# -----------------------------
# Helpers
# -----------------------------
STAGES = ["Discovery", "Early", "Growth", "Mature"]
VERTICAL_HINTS = [
    "Technology", "Software", "Manufacturing", "Autonomy/AI/Robotics",
    "Bio/Life Sciences", "Health/Medtech", "Consumer", "Creative Economy",
]
AUDIENCE_TAGS = [
    "Entrepreneurs", "Startups", "SMEs/Companies", "University Students", "K-12 Students"
]

NEED_SPLIT_RE = re.compile(r"[\n\r;,\u2022]+")


def safe_text(x: Any) -> str:
    if x is None:
        return ""
    if not isinstance(x, str):
        x = str(x)
    return x.strip()


def tokenize(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return [t for t in s.split() if t]


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# -----------------------------
# Data Access
# -----------------------------
@st.cache_data(ttl=60)
def load_programs() -> pd.DataFrame:
    # Pull from `programs` (and optionally join names from providers if needed)
    rows = sb.table("program_parsed").select("*").execute().data or []
    df = pd.DataFrame(rows)
    # Ensure expected columns exist
    must_have = [
        "program_id", "program_name", "website", "services", "core_audience_all",
        "core_audience_ecosystem", "core_audience_entrepreneur", "core_audience_startups",
        "core_audience_sme", "core_audience_ustudents", "core_audience_k12students",
        "growth_stage_discovery", "growth_stage_early", "growth_stage_growth", "growth_stage_mature",
        "verticals", "product_type", "scraped_description", "provider_id", "provider_name",
    ]
    for c in must_have:
        if c not in df.columns:
            df[c] = None
    return df


# -----------------------------
# Matching (simple, transparent)
# -----------------------------

def score_program(program: pd.Series, stage: str, verticals: List[str], audience_tags: List[str], needs_text: str) -> Dict[str, Any]:
    reasons = []
    score = 0.0

    # Stage fit (max ~35)
    stage_map = {
        "Discovery": safe_text(program.get("growth_stage_discovery")),
        "Early": safe_text(program.get("growth_stage_early")),
        "Growth": safe_text(program.get("growth_stage_growth")),
        "Mature": safe_text(program.get("growth_stage_mature")),
    }
    stage_hit = 0
    if stage in stage_map and str(stage_map[stage]).strip() not in ("", "0", "False", "false", "None"):
        score += 35
        stage_hit = 1
        reasons.append(f"Stage match: {stage}")

    # Audience fit (max ~20)
    audiences_blob = " ".join([
        safe_text(program.get("core_audience_all")),
        safe_text(program.get("core_audience_ecosystem")),
        safe_text(program.get("core_audience_entrepreneur")),
        safe_text(program.get("core_audience_startups")),
        safe_text(program.get("core_audience_sme")),
        safe_text(program.get("core_audience_ustudents")),
        safe_text(program.get("core_audience_k12students")),
    ]).lower()
    audience_hits = [t for t in audience_tags if t.lower().split("/")[0] in audiences_blob]
    if audience_hits:
        add = min(20, 10 + 5 * (len(audience_hits) - 1))  # 10 for first, +5 each up to 20
        score += add
        reasons.append(f"Audience fit: {', '.join(audience_hits)} (+{add})")

    # Vertical fit (max ~20)
    prog_verticals = safe_text(program.get("verticals")).lower()
    v_hits = [v for v in verticals if v.lower() in prog_verticals]
    if v_hits:
        add = 20
        score += add
        reasons.append(f"Vertical fit: {', '.join(v_hits)} (+{add})")

    # Keyword similarity between founder needs and program description/services (max ~25)
    corpus = " ".join([
        safe_text(program.get("services")),
        safe_text(program.get("scraped_description")),
        safe_text(program.get("product_type")),
    ])
    sim = jaccard(tokenize(needs_text), tokenize(corpus))
    kw_points = min(25, round(sim * 100 * 0.25, 1))  # scale similarity → up to 25
    if kw_points > 0:
        score += kw_points
        reasons.append(f"Keyword similarity (+{kw_points})")

    # Soft boost if stage not annotated but everything else looks good
    if not stage_hit and (len(audience_hits) + len(v_hits) >= 2) and score < 60:
        score += 8
        reasons.append("Heuristic boost for multi-dimension fit (+8)")

    score = min(100.0, round(score, 1))
    return {"score": score, "reasons": reasons}


def run_matching(programs_df: pd.DataFrame, form_vals: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for _, p in programs_df.iterrows():
        res = score_program(
            p,
            stage=form_vals["stage"],
            verticals=form_vals["verticals"],
            audience_tags=form_vals["audiences"],
            needs_text=form_vals["needs_text"],
        )
        rows.append({
            "program_id": p.get("id"),
            "program_name": p.get("program_name"),
            "provider_name": p.get("provider_name"),
            "website": p.get("website"),
            "score": res["score"],
            "reasons": "; ".join(res["reasons"]) if res["reasons"] else "",
        })
    out = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return out


# -----------------------------
# Persist Intake → DB
# -----------------------------

def insert_entrepreneur(profile: Dict[str, Any]) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    payload = {
        "entrepreneur_id": profile["entrepreneur_id"],
        "date_intake": now.isoformat(),
        "name": profile.get("name"),
        "business_name": profile.get("business_name"),
        "email": profile.get("email"),
        "phone": profile.get("phone"),
        "address": profile.get("address"),
        "zipcode": profile.get("zipcode"),
        "website": profile.get("website"),
        "profile": profile.get("profile"),
        "growth_stage": profile.get("stage"),
        "vertical": ", ".join(profile.get("verticals", [])),
        "county": profile.get("county"),
    }
    sb.table("entrepreneurs").insert(payload).execute()
    return payload


def insert_needs(entrepreneur_id: int, needs_text: str) -> List[Dict[str, Any]]:
    now = datetime.now(timezone.utc)
    needs = [n.strip() for n in NEED_SPLIT_RE.split(needs_text) if n.strip()]
    rows = [{
        "entrepreneur_id": entrepreneur_id,
        "date_intake": now.isoformat(),
        "service": None,
        "need": n,
    } for n in needs]
    if rows:
        sb.table("entrepreneur_needs").insert(rows).execute()
    return rows


def upsert_matches(entrepreneur_id: int, matches_df: pd.DataFrame):
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for _, r in matches_df.iterrows():
        rows.append({
            "entrepreneur_id": entrepreneur_id,
            "provider_id": r.get("program_id"),  # using program_id here; adjust if you want provider_id instead
            "program_name": r.get("program_name"),
            "need_statisfied": "overall",  # simple placeholder
            "growth_score": None,
            "identity_score": None,
            "service_score": None,
            "need_satisfaction_score": r.get("score"),
            "explanation": r.get("reasons"),
            "final_score": r.get("score"),
            "date": now,
        })
    if rows:
        sb.table("needs_match").insert(rows).execute()


# -----------------------------
# UI
# -----------------------------

st.title("🤝 Founder → Program Matcher (MVP)")

with st.sidebar:
    st.markdown("### How this works")
    st.markdown(
        "This MVP collects a quick founder intake, scores public programs with a transparent, rule-based matcher, and lets you export or save matches to the database."
    )
    st.markdown("You can replace the simple matcher with a more advanced model later without changing the UI.")

programs_df = load_programs()

with st.expander("Preview Programs (from DB)", expanded=False):
    st.dataframe(programs_df[[
        "program_id", "program_name", "provider_name", "website", "verticals"
    ]].head(20), use_container_width=True)

st.subheader("1) Founder Intake")
col1, col2, col3 = st.columns(3)
with col1:
    name = st.text_input("Founder name")
    email = st.text_input("Email")
    phone = st.text_input("Phone")
with col2:
    business_name = st.text_input("Business name")
    website = st.text_input("Company website")
    county = st.text_input("County (optional)")
with col3:
    address = st.text_input("Address (optional)")
    zipcode = st.text_input("ZIP (optional)")

stage = st.selectbox("Current stage", STAGES, index=1)
verticals = st.multiselect("Vertical(s)", VERTICAL_HINTS)
audiences = st.multiselect("Core audience tags (who is this program for?)", AUDIENCE_TAGS, default=["Entrepreneurs"]) 

needs_text = st.text_area("Describe your needs (bullets or sentences)",
                          placeholder="e.g., prototyping lab access; early grant guidance; manufacturing partner connections; customer discovery in medtech")

left, right = st.columns([1,1])
with left:
    run_btn = st.button("Run Matching", type="primary", use_container_width=True)
with right:
    clear_btn = st.button("Clear form", use_container_width=True)
    if clear_btn:
        st.experimental_rerun()

if run_btn:
    if not name or not needs_text.strip():
        st.warning("Please provide at least your name and your needs.")
        st.stop()

    # Create a lightweight entrepreneur_id (for demo): hash of name+email+time
    entrepreneur_id = abs(hash(f"{name}-{email}-{time.time()}")) % (10**12)

    profile = {
        "entrepreneur_id": entrepreneur_id,
        "name": name,
        "business_name": business_name,
        "email": email,
        "phone": phone,
        "address": address,
        "zipcode": zipcode,
        "website": website,
        "profile": None,
        "stage": stage,
        "verticals": verticals,
        "county": county,
    }

    with st.spinner("Saving intake..."):
        try:
            insert_entrepreneur(profile)
            insert_needs(entrepreneur_id, needs_text)
            st.success("Intake saved.")
        except Exception as e:
            st.info(f"Continuing without saving to DB (error: {e})")

    with st.spinner("Scoring programs..."):
        match_df = run_matching(
            programs_df,
            {
                "stage": stage,
                "verticals": verticals,
                "audiences": audiences,
                "needs_text": needs_text,
            },
        )

    st.subheader("2) Matches")
    top_n = st.slider("How many results?", 5, 50, 15)
    st.dataframe(match_df.head(top_n), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        csv = match_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, file_name="matches.csv", mime="text/csv", use_container_width=True)

    with c2:
        if st.button("Save top results to DB", use_container_width=True):
            try:
                upsert_matches(entrepreneur_id, match_df.head(top_n))
                st.success("Saved matches to needs_match table.")
            except Exception as e:
                st.error(f"Failed to save matches: {e}")

    with c3:
        st.write("")
        st.caption("Tip: You can later swap in an ML/LLM scorer; just replace `score_program()` and keep the rest.")

st.markdown("---")
st.subheader("Admin: Quick search")
q = st.text_input("Filter programs (client-side)")
if q:
    ql = q.lower()
    filt = programs_df[
        programs_df.apply(lambda r: ql in safe_text(r.get("program_name")).lower() or ql in safe_text(r.get("verticals")).lower(), axis=1)
    ]
    st.dataframe(filt[["program_id", "program_name", "provider_name", "website", "verticals"]], use_container_width=True)
else:
    st.caption("Enter a search term to filter the programs list.")
