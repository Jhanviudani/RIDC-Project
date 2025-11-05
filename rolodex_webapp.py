# rolodex_webapp.py
import json
import os
import math
import pandas as pd
import pgeocode
import pydeck as pdk
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

import functions as fn

load_dotenv()

# ---------------- Cache: DB & LLM ----------------
@st.cache_resource
def get_db_engine():
    return fn.connect_db()

@st.cache_resource
def get_llm_model():
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,
        max_tokens=2000,
        openai_api_key=fn.get_secret("OPENAI_API_KEY"),
    )

# ---------------- About ----------------
def render_about_tab(engine=None):
    import pandas as pd
    import streamlit as st

    st.markdown("## 🌟 SWPA Innovation Ecosystem")
    st.caption("Discover programs, spaces, and support across Southwestern PA — built collaboratively with our community.")

    # --- Top CTA buttons
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        st.link_button("🚀 Entrepreneur Needs Survey — Get recommendations / request a call", "https://forms.gle/eMw5PY9QeTXDqPhy6")
    with c2:
        st.link_button("🧰 Service Provider Program Registration — List your program", "https://forms.gle/aae3SA6YJaZ7d1et5")
    with c3:
        st.markdown(
            "<div style='padding:.6rem .8rem;border:1px solid #eee;border-radius:10px;'>"
            "We’re in <b>beta</b>. <b>Your feedback directly shapes this product.</b>"
            "</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # --- Quick who/what
    colA, colB = st.columns(2)
    with colA:
        st.markdown("### Who is this for?")
        st.markdown(
            "- Entrepreneurs & small business owners\n"
            "- Startup founders & students\n"
            "- Ecosystem builders & service providers"
        )
        st.markdown("### What can you do here?")
        st.markdown(
            "- 🗺️ Explore the region’s innovation map\n"
            "- 🔍 Browse & filter **programs** (Rolodex External)\n"
            "- 🎯 Get **personalized recommendations** (private lookup)\n"
            "- 💬 Ask the database questions in plain English"
        )
    with colB:
        st.markdown("### How it works (30-second tour)")
        st.markdown(
            "1. **We unify data** from a curated Rolodex.\n"
            "2. **You search or ask** what you need (funding, labs, mentors, etc.).\n"
            "3. **We recommend** high-fit programs and explain why."
        )
        st.markdown("### Privacy in brief")
        st.markdown(
            "- No public list of entrepreneurs\n"
            "- Chat answers hide backend sources\n"
            "- Data used only to improve recommendations"
        )

    # --- Optional live metrics from rolodex_external only
    if engine is not None:
        try:
            rolo_n = int(pd.read_sql("SELECT COUNT(*) AS n FROM rolodex_external;", engine).iloc[0]["n"])
            st.markdown("---")
            st.markdown("### Snapshot")
            m1, _m2, _m3 = st.columns(3)
            m1.metric("Rolodex entries", f"{rolo_n:,}")
        except Exception:
            pass

    st.markdown("---")

    # --- Feedback (simple form -> feedback table)
    st.markdown("### We’d love your feedback")
    with st.form("about_feedback"):
        fb_col1, fb_col2 = st.columns([2, 1])
        with fb_col1:
            feedback_text = st.text_area(
                "What’s working? What’s missing? Any bugs or ideas?",
                placeholder="Share a quick note…"
            )
        with fb_col2:
            contact_email = st.text_input("Email (optional)")
        submitted = st.form_submit_button("Send feedback")
        if submitted:
            if feedback_text.strip():
                try:
                    df = pd.DataFrame([{
                        "page": "about",
                        "feedback_text": feedback_text.strip(),
                        "contact_email": (contact_email or "").strip()
                    }])
                    fn.insert_data_to_supabase(df, "feedback")
                    st.success("Thanks! Your feedback has been recorded. 🙌")
                except Exception:
                    st.info("Thanks! If this didn’t save, please use this form instead:")
                    st.link_button("Open feedback form", "https://forms.gle/eMw5PY9QeTXDqPhy6")
            else:
                st.warning("Please add a bit of feedback before submitting.")

    # --- FAQ (HTML <details> to avoid nested expander errors)
    st.markdown(
        """
        <details style="margin-top: .75rem;">
          <summary style="font-weight:600; cursor:pointer;">FAQ (short)</summary>
          <div style="margin-top:.5rem">
            <p><b>Where does the data come from?</b> A curated Rolodex of programs across the ecosystem.</p>
            <p><b>Are recommendations private?</b> Yes — we don't list entrepreneurs publicly and we hide internal sources in the UI.</p>
            <p><b>How can I get my program listed?</b> Use the Provider Intake button above.</p>
          </div>
        </details>
        """,
        unsafe_allow_html=True
    )

# ---------------- Rolodex Overview (map) ----------------
def render_overview_tab(engine):
    st.subheader("📍 Map: Rolodex External")

    q_rolo = """
    SELECT
      org_name   AS name,
      COALESCE(address,'') AS address,
      COALESCE(county_hq,'') AS county,
      latitude, longitude
    FROM rolodex_external;
    """
    try:
        df_rolo = fn.get_data(q_rolo, engine).dropna(subset=["latitude", "longitude"])
    except Exception:
        df_rolo = pd.DataFrame(columns=["name", "address", "county", "latitude", "longitude"])

    if df_rolo.empty:
        st.info("No map points to show yet.")
        return

    df_rolo["color"] = [fn.hex_to_rgb("#FFA500")] * len(df_rolo)

    # SWPA counties outline
    geojson_url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    try:
        us_counties = requests.get(geojson_url, timeout=10).json()
    except Exception:
        us_counties = {"type": "FeatureCollection", "features": []}
    swpa_fips = ["42003", "42005", "42007", "42019", "42051", "42059", "42063", "42073", "42125", "42129"]
    swpa_geo = {
        "type": "FeatureCollection",
        "features": [f for f in us_counties.get("features", []) if f.get("id") in swpa_fips],
    }

    layers = [
        pdk.Layer(
            "GeoJsonLayer",
            data=swpa_geo,
            opacity=0.2,
            stroked=True,
            filled=True,
            get_fill_color=[200, 200, 200],
            get_line_color=[0, 0, 0],
            get_line_width=2,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=df_rolo,
            get_position=["longitude", "latitude"],
            get_color="color",
            get_radius=1000,
            pickable=True,
            radius_min_pixels=5,
            radius_max_pixels=15,
        ),
    ]

    view_state = pdk.ViewState(
        latitude=df_rolo["latitude"].mean(),
        longitude=df_rolo["longitude"].mean(),
        zoom=8,
    )

    st.pydeck_chart(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=view_state,
            layers=layers,
            tooltip={"html": "<b>{name}</b><br/>{county}<br/>{address}"},
        )
    )

    st.markdown(f"**Rolodex External points:** {len(df_rolo)}")

# ---------------- Needs (ROL0DEX_EXTERNAL ONLY) ----------------
def render_needs_tab(engine):
    st.subheader("📊 Services by County (from Rolodex External)")
    q = """
    SELECT
      COALESCE(county_hq,'') AS county,
      COALESCE(primary_service,'') AS primary_service
    FROM rolodex_external;
    """
    df = fn.get_data(q, engine)

    if df.empty:
        st.info("No data available.")
        return

    services = sorted(df["primary_service"].dropna().unique().tolist())
    default_pick = services[: min(6, len(services))]
    selected = st.multiselect("Filter by service(s)", services, default=default_pick)
    filtered = df[df["primary_service"].isin(selected)] if selected else df

    counts = filtered.groupby(["county", "primary_service"]).size().reset_index(name="count")
    fig = px.bar(counts, x="county", y="count", color="primary_service", title="Service availability by county")
    fig.update_layout(barmode="stack", title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)

# ---------------- Programs (ROL0DEX_EXTERNAL ONLY) ----------------
def render_programs_tab(engine):
    st.subheader("🔍📄 Programs (from Rolodex External)")
    q = """
    SELECT
      org_name AS provider_name,
      program_name,
      COALESCE(website,'') AS website,
      COALESCE(county_hq,'') AS county,
      COALESCE(address,'') AS address,
      COALESCE(primary_service,'') AS services,
      COALESCE(verticals_summary,'') AS verticals,
      COALESCE(product_types_summary,'') AS product_type,
      COALESCE(full_description, description, '') AS scraped_description
    FROM rolodex_external;
    """
    df = fn.get_data(q, engine)

    if df.empty:
        st.info("No programs found yet.")
        return

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        provider = st.selectbox("Provider", ["All"] + sorted(df["provider_name"].dropna().unique().tolist()))
    with col2:
        county = st.selectbox("County", ["All"] + sorted(df["county"].dropna().unique().tolist()))
    with col3:
        v = st.selectbox("Vertical", ["All"] + fn.extract_unique_items(df, "verticals"))
    p = st.selectbox("Product Type", ["All"] + fn.extract_unique_items(df, "product_type"))

    f = df.copy()
    if provider != "All": f = f[f["provider_name"] == provider]
    if county   != "All": f = f[f["county"] == county]
    if v        != "All": f = f[f["verticals"].str.contains(v, na=False)]
    if p        != "All": f = f[f["product_type"].str.contains(p, na=False)]

    st.markdown(f"**{len(f)}** program(s) match the selected filters.")
    st.dataframe(f, use_container_width=True)

# ---------------- Matching (ROL0DEX_EXTERNAL ONLY + privacy-friendly) ----------------
def render_matching_tab(engine, model):
    """Renders the Matching tool tab from rolodex_external only, with private entrepreneur lookup."""
    import json
    import pandas as pd
    import difflib, re

    st.subheader("🎯 Program Recommendations ")

    # Load entrepreneurs + needs (for profile only)
    q_ent = """
      SELECT DISTINCT ON (entrepreneur_id)
             entrepreneur_id, name, business_name, email, phone, address, zipcode,
             website, profile, growth_stage, vertical, county
      FROM entrepreneurs
      ORDER BY entrepreneur_id, date_intake DESC;
    """
    q_needs = """
      SELECT DISTINCT ON (entrepreneur_id, need, date_intake)
             entrepreneur_id, service, need
      FROM entrepreneur_needs
      ORDER BY entrepreneur_id, need, date_intake DESC;
    """
    df_entrep = fn.get_data(q_ent, engine)
    df_needs  = fn.get_data(q_needs, engine)

    if df_entrep.empty:
        st.info("No entrepreneurs found yet.")
        return

    # Privacy-friendly entrepreneur selection (no public dropdown)
    typed_name = st.text_input(
        "Enter the name you used on the form",
        value="",
        help="We’ll look up your record privately. Try the same spacing/punctuation."
    ).strip()

    if not typed_name:
        st.info("Enter your name to load your profile and needs.")
        return

    # 1) Exact match on 'name' or 'business_name'
    exact = df_entrep[
        (df_entrep["name"].fillna("").str.casefold() == typed_name.casefold()) |
        (df_entrep["business_name"].fillna("").str.casefold() == typed_name.casefold())
    ]
    candidates = exact

    # 2) Contains fallback
    if candidates.empty:
        patt = re.escape(typed_name)
        contains = df_entrep[
            df_entrep["name"].fillna("").str.contains(patt, case=False) |
            df_entrep["business_name"].fillna("").str.contains(patt, case=False)
        ]
        candidates = contains

    # 3) Fuzzy fallback
    if candidates.empty:
        keys = (df_entrep["name"].fillna("") + " | " + df_entrep["business_name"].fillna("")).tolist()
        best = difflib.get_close_matches(typed_name, keys, n=1, cutoff=0.8)
        if best:
            ix = keys.index(best[0])
            candidates = df_entrep.iloc[[ix]]

    if candidates.empty:
        st.warning("We couldn’t find a record with that name. Please check the spelling or try a different variation.")
        return

    # Prefer exact matches if multiple
    if len(candidates) > 1:
        exact_rows = candidates[
            (candidates["name"].fillna("").str.casefold() == typed_name.casefold()) |
            (candidates["business_name"].fillna("").str.casefold() == typed_name.casefold())
        ]
        if not exact_rows.empty:
            candidates = exact_rows

    row = candidates.iloc[0]
    eid = row["entrepreneur_id"]
    needs_for_e = df_needs[df_needs["entrepreneur_id"] == eid][["service", "need"]]

    entrepreneur = row.to_dict()
    entrepreneur["needs_needed"] = needs_for_e.to_dict(orient="records")

    # Build provider/program payload (rolodex_external only)
    try:
        payload = fn.build_matching_payload(
            engine,
            entrepreneur_zip=entrepreneur.get("zipcode")
        )
    except Exception as e:
        st.error(f"Failed to build provider payload: {e}")
        return

    full_payload = [p for p in payload if p.get("programs")]
    if not full_payload:
        st.info("No programs available to evaluate.")
        return

    st.caption(f"Evaluating {len(full_payload)} organizations from rolodex_external.")

    # Chunking for LLM context
    def _batches(items, size):
        for i in range(0, len(items), size):
            yield i // size + 1, items[i:i+size]
    BATCH_SIZE = 60

    run = st.button("Run Program Recommendation Assistant")
    if not run:
        return

    progress = st.progress(0.0)
    all_matches = []
    num_batches = (len(full_payload) + BATCH_SIZE - 1) // BATCH_SIZE

    for b_idx, batch in _batches(full_payload, BATCH_SIZE):
        st.write(f"Processing batch {b_idx}/{num_batches} · providers in batch: {len(batch)}")
        batch_json = json.dumps(batch)

        resp_text = fn.match_programs_to_entrepreneur(entrepreneur, batch_json, model)
        try:
            parsed = fn.coerce_json_array(resp_text)
            for m in parsed:
                m["batch_index"] = b_idx  # internal; will drop in display
            all_matches.extend(parsed)
        except Exception as e:
            st.error(f"Failed to generate recommendations: {e}")
            with st.expander(f"Raw model output (batch {b_idx})"):
                st.code((resp_text or "")[:4000])
        progress.progress(b_idx / float(num_batches))

    if not all_matches:
        st.error("No matches returned from any batch.")
        return

    # Deduplicate & finalize scores
    deduped, seen = [], set()
    for m in all_matches:
        key = (str(m.get("provider_id")), m.get("program_name"))
        if key in seen:
            continue
        seen.add(key)

        for k in ("distance_score", "identity_score", "service_score", "need_satisfaction_score"):
            try:
                m[k] = float(m.get(k, 0) or 0)
            except Exception:
                m[k] = 0.0
        if "final_score" not in m:
            m["final_score"] = (
                m["distance_score"]
                + m["identity_score"]
                + m["service_score"]
                + m["need_satisfaction_score"]
            ) / 4.0

        deduped.append(m)

    # Summaries
    st.subheader("Entrepreneur summary")
    try:
        summary = fn.summarize_user_identity_and_needs(entrepreneur, model)
        st.write(summary)
    except Exception:
        st.write("Summary unavailable.")

    st.subheader("Program Recommendations")
    try:
        rec_summary = fn.summarize_recommendations(entrepreneur, deduped, model)
        st.write(rec_summary)
    except Exception:
        st.write("Recommendation summary unavailable.")

    # Structured table: drop batch index and include website (via catalog merge)
    matches_df = pd.DataFrame(deduped).sort_values("final_score", ascending=False)
    try:
        catalog = fn.build_program_catalog(engine)[["provider_name","program_name","website"]]
        matches_df = matches_df.merge(catalog, how="left", on=["provider_name","program_name"])
    except Exception:
        if "website" not in matches_df.columns:
            matches_df["website"] = None

    if "batch_index" in matches_df.columns:
        matches_df = matches_df.drop(columns=["batch_index"], errors="ignore")
    # Avoid showing any internal source fields if present
    matches_df = matches_df.drop(columns=["source"], errors="ignore")

    st.subheader(f"Structured Recommendations · {len(matches_df)} unique results")
    st.dataframe(
        matches_df[
            ["program_name", "provider_name", "website", "final_score",
             "distance_score", "identity_score", "service_score", "need_satisfaction_score",
             "need_satisfied", "explanation"]
        ].dropna(axis=1, how="all", errors="ignore"),
        use_container_width=True
    )

    # Optional export to DB
    if st.button("Send Results to Database"):
        try:
            matches_df["date"] = pd.Timestamp.now()
            fn.insert_data_to_supabase(matches_df, "needs_match")
            st.success("Saved to database.")
        except Exception as e:
            st.error(f"Failed to save: {e}")

# ---------------- Chat (ROL0DEX_EXTERNAL ONLY) ----------------
def render_chat_tab(engine, model):
    """Chat with the catalog (rolodex_external only)."""
    st.subheader("💬 Ask the Ecosystem")
    st.caption("Ask questions like: “Is there a provider that can help with grants for a manufacturing startup?”")

    if "qa_msgs" not in st.session_state:
        st.session_state.qa_msgs = [
            {"role": "assistant",
             "content": "Hi! Ask what you need (funding, prototyping, mentorship, specific verticals, etc.)."}
        ]

    for m in st.session_state.qa_msgs:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_q = st.chat_input("Type your question")
    if not user_q:
        return

    st.session_state.qa_msgs.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    # This relies on fn.nl_search_programs() which should already be rolodex_external-only
    hits = fn.nl_search_programs(engine, user_q, limit=20)

    if not hits.empty:
        with st.expander("Retrieved matches (top 20)"):
            st.dataframe(
                hits[["program_name","provider_name","services","verticals","product_type","county","website"]],
                use_container_width=True
            )

    if hits.empty:
        answer = "I couldn’t find anything relevant in the catalog. Try different words (e.g., 'grant', 'loan', 'agriculture')."
    else:
        answer = fn.answer_query_over_catalog(model, user_q, hits)

    st.session_state.qa_msgs.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

# ---------------- Main ----------------
def main():
    st.title("SWPA Innovation Ecosystem")

    engine = get_db_engine()
    model = get_llm_model()

    st.caption("Open the sections you want below. No tabs — everything on one page.")
    show_all = st.checkbox("Open all sections", value=False)

    with st.expander("ℹ️ About", expanded=show_all):
        render_about_tab(engine)

    with st.expander("🗺️ Rolodex Overview (Map)", expanded=show_all):
        render_overview_tab(engine)

    with st.expander("📊 Search Based on Needs and Services", expanded=show_all):
        render_needs_tab(engine)

    with st.expander("🔍 Search Based on Service Providers and Programs", expanded=show_all):
        render_programs_tab(engine)

    with st.expander("🎯 Matching Tool (Personalized Recommendations)", expanded=show_all):
        render_matching_tab(engine, model)

    with st.expander("💬 Chat with the Database", expanded=show_all):
        render_chat_tab(engine, model)

if __name__ == "__main__":
    main()
