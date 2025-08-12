import os
import re
import glob
import json
import math
import time
import uuid
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------
# Config
# ---------------------------
CSV_PATH = "programs.csv"
SCRAPED_DIR = "scraped"   # directory with your pre-scraped .txt files
OPENAI_MODEL = "gpt-4o-mini"  # fast & cheap; change if you want
RANK_BATCH = 20                 # how many programs to score per LLM call (chunked)
TOP_K = 5

# ---------------------------
# Helpers
# ---------------------------
def norm(s: str) -> str:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    return str(s).strip()

def safe_join_pieces(pieces):
    """Join any mixture of strings/NaNs safely and return '' if nothing usable."""
    clean = [norm(x) for x in pieces if norm(x)]
    return ", ".join(clean)

def infer_cost_label(cost_raw: str, membership_raw: str) -> str:
    c = norm(cost_raw).lower()
    m = norm(membership_raw).lower()

    if "free" in c:
        return "Free"
    if any(k in c for k in ["fee", "cost", "$", "paid", "tuition", "price"]):
        return "Paid"

    if "member" in c or "membership" in c or "dues" in c:
        return "Membership"
    if "member" in m or "membership" in m or "dues" in m:
        return "Membership"

    if c == "" and m == "":
        return "Unknown"
    return "Unknown"

def load_scraped_texts(scraped_dir: str):
    """Return a list of (path, text, basename_lower)."""
    out = []
    if not os.path.isdir(scraped_dir):
        return out
    for p in glob.glob(os.path.join(scraped_dir, "**/*.txt"), recursive=True):
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                t = f.read()
            out.append((p, t, os.path.basename(p).lower()))
        except Exception:
            pass
    return out

def find_best_text_for_program(text_index, program_name, org_name):
    """Heuristic filename match against program and organization names."""
    pn = norm(program_name).lower()
    on = norm(org_name).lower()
    if not text_index:
        return None
    # prefer program name; then organization
    for needle in [pn, on]:
        if not needle:
            continue
        # pick first that contains the needle in basename
        for (path, text, base) in text_index:
            if needle and needle in base:
                return text
    # fallback: None
    return None

def chunked(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

# ---------------------------
# LLM (OpenAI via env var)
# ---------------------------
def call_openai(system, user, model=OPENAI_MODEL):
    """
    Minimal OpenAI client using the modern API (expects OPENAI_API_KEY in env).
    """
    import openai  # openai>=1.0
    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":system},
            {"role":"user","content":user}
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content

# ---------------------------
# Matching logic
# ---------------------------
def build_program_payload(row, long_text: str):
    # Things the LLM will see/score on
    return {
        "program_name": norm(row.get("Program Name")),
        "organization": norm(row.get("Organization")),
        "address": norm(row.get("Address")),
        "lat": norm(row.get("Latitude Coordinates")),
        "lon": norm(row.get("Longitude Coordinates")),
        "cost_raw": norm(row.get("Cost")),
        "membership_raw": norm(row.get("Membership Options")),
        "long_text": long_text[:8000] if long_text else "",  # cap to avoid long prompts
        # quick bullet-ish tags: scan all boolean-like columns
        "tags": ", ".join([c for c in row.index if str(row[c]).strip().lower() in ["yes","y","true","1"]])[:300],
    }

def score_batch(need_text: str, programs_payload: list):
    """
    Ask the LLM to score a batch of programs.
    Returns list of dicts with scores and short rationales.
    """
    system = (
        "You are a pragmatic startup navigator. Score support programs for an entrepreneur's needs. "
        "Return STRICT JSON ONLY. No commentary."
    )
    # compact input for the model
    pack = []
    for i, p in enumerate(programs_payload):
        pack.append({
            "id": i,
            "program_name": p["program_name"],
            "organization": p["organization"],
            "address": p["address"],
            "tags": p["tags"],
            "summary": p["long_text"][:1200],  # shorter summary per item
        })

    user = f"""
FOUNDER_NEEDS:
{need_text}

PROGRAMS:
{json.dumps(pack, ensure_ascii=False)}

INSTRUCTIONS:
For each program, return a JSON array of objects like:
[
  {{
    "id": <int>,             // id I provided
    "match_score": <0-100>,  // higher = better match to needs
    "rationale": "<<= 30 words on why it fits>"
  }},
  ...
]
Use only information in the provided summary/tags. Output JSON only.
"""
    raw = call_openai(system, user)
    # tolerant JSON parse
    start = min([i for i in [raw.find("["), raw.find("{")] if i >= 0] or [0])
    jtxt = raw[start:]
    try:
        data = json.loads(jtxt)
    except Exception:
        # fallback: try to wrap in [] if a single object
        if jtxt.strip().startswith("{") and jtxt.strip().endswith("}"):
            data = [json.loads(jtxt)]
        else:
            data = []

    scored = {}
    for obj in data:
        try:
            scored[int(obj["id"])] = {
                "match_score": float(obj.get("match_score", 0)),
                "rationale": norm(obj.get("rationale")),
            }
        except Exception:
            continue
    return [scored.get(i, {"match_score":0.0, "rationale":""}) for i in range(len(programs_payload))]

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Founder ↔ Program Matcher (Local)", layout="wide")

st.sidebar.header("Settings")
st.sidebar.caption("All data stays local. No databases involved.")
st.sidebar.markdown("**LLM model:** `" + OPENAI_MODEL + "`")

st.title("🤝 Founder ↔ Program Matcher (Local)")
need_text = st.text_area(
    "Describe your needs (be specific):",
    placeholder="e.g., Robotics hardware founder seeking prototyping lab, early grant guidance, manufacturing partners, and clinical pilots.",
    height=120
)

col_btn1, col_btn2 = st.columns([1,1])
run = col_btn1.button("Find Matches", type="primary")
if col_btn2.button("Clear"):
    need_text = ""
    st.experimental_rerun()

# Load CSV
if not os.path.exists(CSV_PATH):
    st.error(f"CSV not found at {CSV_PATH}")
    st.stop()

df = pd.read_csv(CSV_PATH)

# Clean up column names we care about (avoid KeyError if missing)
for required in ["Program Name", "Organization", "Address", "Latitude Coordinates", "Longitude Coordinates", "Cost", "Membership Options"]:
    if required not in df.columns:
        df[required] = np.nan

# Preload texts once
text_index = load_scraped_texts(SCRAPED_DIR)

if run and need_text.strip():
    # Build payloads with attached local texts
    payloads = []
    source_rows = []
    for _, row in df.iterrows():
        text = find_best_text_for_program(
            text_index,
            row.get("Program Name"),
            row.get("Organization"),
        )
        payloads.append(build_program_payload(row, text))
        source_rows.append(row)

    # Score in batches
    scores_all = []
    for batch in chunked(payloads, RANK_BATCH):
        batch_scores = score_batch(need_text, batch)
        scores_all.extend(batch_scores)

    # Stitch scores back
    results = []
    for row, pld, sc in zip(source_rows, payloads, scores_all):
        results.append({
            "Program Name": pld["program_name"],
            "Organization": pld["organization"],
            "Address": pld["address"] or "—",
            "Latitude": float(pld["lat"]) if str(pld["lat"]).replace(".","",1).isdigit() else np.nan,
            "Longitude": float(pld["lon"]) if str(pld["lon"]).replace(".","",1).isdigit() else np.nan,
            "Cost Label": infer_cost_label(pld["cost_raw"], pld["membership_raw"]),
            "Raw Cost": pld["cost_raw"],
            "Membership": pld["membership_raw"],
            "Match score": round(float(sc.get("match_score", 0)), 1),
            "Rationale": sc.get("rationale", ""),
        })

    out = pd.DataFrame(results).sort_values("Match score", ascending=False).head(TOP_K)

    st.subheader("Top 5 Matches")
    for _, r in out.iterrows():
        with st.container(border=True):
            title = f"{r['Program Name']} - {r['Organization']}".strip(" -")
            st.markdown(f"**{title}**")
            st.caption(r.get("Rationale") or "—")
            c1, c2, c3 = st.columns([1,1,2])
            with c1:
                st.metric("Match score", r["Match score"])
            with c2:
                st.metric("Cost", r["Cost Label"])
            with c3:
                st.caption(r["Address"])

            # Show quick tags about cost/membership if present
            small = []
            if norm(r["Raw Cost"]):
                small.append(f"Cost notes: {r['Raw Cost']}")
            if norm(r["Membership"]):
                small.append(f"Membership: {r['Membership']}")
            if small:
                st.caption(" • ".join(small))

    st.subheader("Map of Matches")
    if out[["Latitude","Longitude"]].notna().all(axis=1).any():
        m = out.dropna(subset=["Latitude","Longitude"])[["Program Name","Organization","Latitude","Longitude"]]
        m = m.rename(columns={"Latitude":"lat","Longitude":"lon"})
        st.map(m, latitude="lat", longitude="lon", size=60)
    else:
        st.caption("No latitude/longitude available for these picks.")
else:
    st.info("Enter your needs and click **Find Matches**. The app uses your local `programs.csv` and any `.txt` program descriptions from the `scraped/` folder.")
