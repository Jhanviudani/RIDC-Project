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

# ---------------- Tabs ----------------
def render_about_tab():
    st.subheader("About this app")
    st.write(
        """
        This app maps SWPA’s innovation ecosystem, analyzes entrepreneur needs,
        and recommends programs using data from **Providers** (intake form) and our own **internal database**.


        """
    )
    st.markdown("---")
    st.write("🚀 Entrepreneurs --> Complete this form to get registered in the system and receive an initial list of resources as well as direct outreach from a representative to help you find resources. Alternatively, you can chat with the chatbot on the chat tab. \n [Entrepreneur Intake Form](https://forms.gle/eMw5PY9QeTXDqPhy6).")
    st.write("🧰 Service providers --> Have a program to support entrepreneurs? Add it using this form. \n [Service Provider Intake Form](https://forms.gle/aae3SA6YJaZ7d1et5)")

def render_overview_tab(engine):
    st.subheader("📍 Map: Providers, Entrepreneurs, Rolodex")

    # Providers & Entrepreneurs (ZIP → lat/long)
    q_zip = """
    SELECT provider_id AS id, provider_name AS name, address, zipcode, 'Provider' AS user
    FROM providers
    UNION
    SELECT entrepreneur_id AS id, business_name AS name, address, zipcode, 'Entrepreneur' AS user
    FROM entrepreneurs;
    """
    df_zip = fn.get_data(q_zip, engine)
    df_zip = fn.add_coordinates(df_zip)                # adds latitude/longitude from zipcode
    map_df_pe = df_zip.dropna(subset=["latitude", "longitude"])

    # Rolodex points (already have lat/long)
    q_rolo = """
    SELECT
      org_name AS name,
      COALESCE(address,'') AS address,
      latitude, longitude,
      'Rolodex' AS user
    FROM rolodex_points;
    """
    try:
        df_rolo = fn.get_data(q_rolo, engine).dropna(subset=["latitude", "longitude"])
    except Exception:
        df_rolo = pd.DataFrame(columns=["name", "address", "latitude", "longitude", "user"])

    # Colors
    dynamic_color_map = {
        "Provider": fn.hex_to_rgb("#00FF00"),
        "Entrepreneur": fn.hex_to_rgb("#0000FF"),
        "Rolodex": fn.hex_to_rgb("#FFA500"),
    }
    if not map_df_pe.empty:
        map_df_pe["color"] = map_df_pe["user"].map(dynamic_color_map)
    if not df_rolo.empty:
        df_rolo["color"] = df_rolo["user"].map(dynamic_color_map)

    # SWPA counties outline
    geojson_url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    swpa_fips = ["42003", "42005", "42007", "42019", "42051", "42059", "42063", "42073", "42125", "42129"]
    us_counties = requests.get(geojson_url).json()
    swpa_geo = {
        "type": "FeatureCollection",
        "features": [f for f in us_counties["features"] if f["id"] in swpa_fips],
    }

    # Layers
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
            data=map_df_pe,
            get_position=["longitude", "latitude"],
            get_color="color",
            get_radius=1000,
            pickable=True,
            radius_min_pixels=5,
            radius_max_pixels=15,
        ),
    ]
    if not df_rolo.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=df_rolo,
                get_position=["longitude", "latitude"],
                get_color="color",
                get_radius=1000,
                pickable=True,
                radius_min_pixels=5,
                radius_max_pixels=15,
            )
        )

    # Center the map
    combined = pd.concat([map_df_pe, df_rolo], ignore_index=True) if not df_rolo.empty else map_df_pe
    if combined.empty:
        st.info("No map points to show yet.")
        return

    view_state = pdk.ViewState(
        latitude=combined["latitude"].mean(),
        longitude=combined["longitude"].mean(),
        zoom=8,
    )

    # Legend
    st.markdown(
        """
        <div style='display:flex;gap:16px;align-items:center;margin-bottom:8px'>
          <span style='display:flex;gap:6px;align-items:center'><div style='width:14px;height:14px;background:#00FF00'></div>Provider</span>
          <span style='display:flex;gap:6px;align-items:center'><div style='width:14px;height:14px;background:#0000FF'></div>Entrepreneur</span>
          <span style='display:flex;gap:6px;align-items:center'><div style='width:14px;height:14px;background:#FFA500'></div>Rolodex</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.pydeck_chart(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=view_state,
            layers=layers,
            tooltip={"html": "<b>{name}</b><br/>{user}<br/>{address}"},
        )
    )

    st.markdown(
        f"""
        **Stats**
        - Providers: {len(map_df_pe[map_df_pe['user']=='Provider'])}
        - Entrepreneurs: {len(map_df_pe[map_df_pe['user']=='Entrepreneur'])}
        - Rolodex points: {len(df_rolo) if not df_rolo.empty else 0}
        """
    )

def render_needs_tab(engine):
    st.subheader("📊 Entrepreneur Needs")
    q_needs = """
    SELECT en.entrepreneur_id, e.county, en.need, en.service
    FROM entrepreneur_needs en
    JOIN entrepreneurs e
      ON en.entrepreneur_id = e.entrepreneur_id
     AND en.date_intake = e.date_intake;
    """
    needs_df = fn.get_data(q_needs, engine)

    if needs_df.empty:
        st.info("No needs data available.")
        return

    services = sorted(needs_df["service"].dropna().unique().tolist())
    selected = st.multiselect("Filter by service(s)", services, default=services)
    filtered = needs_df[needs_df["service"].isin(selected)] if selected else needs_df

    counts = filtered.groupby(["county", "need"]).size().reset_index(name="count")
    fig = px.bar(counts, x="county", y="count", color="need", title="Needs by County")
    fig.update_layout(barmode="stack", title_x=0.5)
    st.plotly_chart(fig)

def render_programs_tab(engine):
    st.subheader("🔍📄 Programs (Providers + Rolodex)")

    # A) Provider programs
    q_form = """
    SELECT DISTINCT ON (t2.provider_id, t2.program_id)
      t2.provider_id, t2.provider_name, t2.program_id, t2.program_name,
      t2.website, t2.contact_name, t2.contact_email,
      t1.county, t1.address,
      t2.services, t2.verticals, t2.product_type,
      t2.scraped_description
    FROM programs t2
    JOIN providers t1 ON t1.provider_id = t2.provider_id
    ORDER BY t2.provider_id, t2.program_id, t2.date_intake_form DESC;
    """
    df_form = fn.get_data(q_form, engine)
    df_form["source"] = "providers"

    # B) Rolodex programs (flattened)
    q_rolo = """
    SELECT
      org_name AS provider_name,
      program_name,
      COALESCE(website,'') AS website,
      '' AS contact_name,
      '' AS contact_email,
      COALESCE(county_hq,'') AS county,
      COALESCE(address,'') AS address,
      COALESCE(primary_service,'') AS services,
      COALESCE(attributes->>'Vertical(s) Summary','')     AS verticals,
      COALESCE(attributes->>'Product Type(s) Summary','') AS product_type,
      COALESCE(full_description, description, '') AS scraped_description
    FROM rolodex_points;
    """
    try:
        df_rolo = fn.get_data(q_rolo, engine)
    except Exception:
        df_rolo = pd.DataFrame()
    df_rolo["source"] = "rolodex"

    # C) Union
    common = [
        "provider_name", "program_name", "website", "contact_name", "contact_email",
        "county", "address", "services", "verticals", "product_type",
        "scraped_description", "source"
    ]
    for c in common:
        if c not in df_form: df_form[c] = ""
        if c not in df_rolo: df_rolo[c] = ""
    df_all = pd.concat([df_form[common], df_rolo[common]], ignore_index=True)

    if df_all.empty:
        st.info("No programs found yet.")
        return

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        provider = st.selectbox("Provider", ["All"] + sorted(df_all["provider_name"].dropna().unique().tolist()))
    with col2:
        county = st.selectbox("County", ["All"] + sorted(df_all["county"].dropna().unique().tolist()))
    with col3:
        src = st.selectbox("Source", ["All", "providers", "rolodex"])

    verticals = fn.extract_unique_items(df_all, "verticals")
    product_types = fn.extract_unique_items(df_all, "product_type")
    v = st.selectbox("Vertical", ["All"] + verticals)
    p = st.selectbox("Product Type", ["All"] + product_types)

    f = df_all.copy()
    if provider != "All": f = f[f["provider_name"] == provider]
    if county   != "All": f = f[f["county"] == county]
    if src      != "All": f = f[f["source"] == src]
    if v        != "All": f = f[f["verticals"].str.contains(v, na=False)]
    if p        != "All": f = f[f["product_type"].str.contains(p, na=False)]

    st.markdown(f"**{len(f)}** program(s) match the selected filters.")
    st.dataframe(f, use_container_width=True)



def render_matching_tab(engine, model):
    """Renders the Matching tool tab (Providers + Rolodex) with robust JSON parsing and privacy-friendly selection."""
    import json
    import pandas as pd
    import difflib, re

    st.subheader("🎯 Program Recommendations ")

    # --------- Load entrepreneurs + needs ----------
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

    # --------- Privacy-friendly entrepreneur selection (no public dropdown) ----------
    typed_name = st.text_input(
        "Enter the name you used on the form",
        value="",
        help="We’ll look up your record privately. Try the same spacing/punctuation."
    ).strip()

    if not typed_name:
        st.info("Enter your name to load your profile and needs.")
        return

    # 1) Exact, case-insensitive match on either 'name' or 'business_name'
    exact = df_entrep[
        (df_entrep["name"].fillna("").str.casefold() == typed_name.casefold()) |
        (df_entrep["business_name"].fillna("").str.casefold() == typed_name.casefold())
    ]
    candidates = exact

    # 2) If no exact match, try "contains" (still case-insensitive)
    if candidates.empty:
        patt = re.escape(typed_name)
        contains = df_entrep[
            df_entrep["name"].fillna("").str.contains(patt, case=False) |
            df_entrep["business_name"].fillna("").str.contains(patt, case=False)
        ]
        candidates = contains

    # 3) If still nothing, use a simple fuzzy top-1 against concatenated name fields
    if candidates.empty:
        keys = (df_entrep["name"].fillna("") + " | " + df_entrep["business_name"].fillna("")).tolist()
        best = difflib.get_close_matches(typed_name, keys, n=1, cutoff=0.8)
        if best:
            ix = keys.index(best[0])
            candidates = df_entrep.iloc[[ix]]

    if candidates.empty:
        st.warning("We couldn’t find a record with that name. Please check the spelling or try a different variation.")
        return

    # Prefer exact-match rows first, if multiple
    if len(candidates) > 1:
        exact_rows = candidates[
            (candidates["name"].fillna("").str.casefold() == typed_name.casefold()) |
            (candidates["business_name"].fillna("").str.casefold() == typed_name.casefold())
        ]
        if not exact_rows.empty:
            candidates = exact_rows

    # Final pick
    row = candidates.iloc[0]
    eid = row["entrepreneur_id"]
    needs_for_e = df_needs[df_needs["entrepreneur_id"] == eid][["service", "need"]]

    entrepreneur = row.to_dict()
    entrepreneur["needs_needed"] = needs_for_e.to_dict(orient="records")

    # --------- Build combined provider payload (providers + rolodex) ----------
    try:
        payload = fn.build_matching_payload(
            engine,
            entrepreneur_zip=entrepreneur.get("zipcode")
        )
    except Exception as e:
        st.error(f"Failed to build provider payload: {e}")
        return

    # Keep only providers that actually have at least one program
    full_payload = [p for p in payload if p.get("programs")]
    if not full_payload:
        st.info("No programs available to evaluate.")
        return

    total_form  = sum(1 for p in full_payload if p.get("source") == "providers")
    total_rolo  = sum(1 for p in full_payload if p.get("source") == "rolodex")
    st.caption(f"Evaluating {len(full_payload)} providers "
               f"({total_form} from form intake / {total_rolo} from rolodex).")

    # Helper to chunk the payload so we don't overflow the model context
    def _batches(items, size):
        for i in range(0, len(items), size):
            yield i // size + 1, items[i:i+size]

    BATCH_SIZE = 60  # tune if you want larger/smaller batches

    run = st.button("Run Program Recommendation Assistant")
    if not run:
        return

    progress = st.progress(0.0)
    all_matches = []
    num_batches = (len(full_payload) + BATCH_SIZE - 1) // BATCH_SIZE

    for b_idx, batch in _batches(full_payload, BATCH_SIZE):
        st.write(f"Processing")
        batch_json = json.dumps(batch)

        # Call the LLM and parse JSON safely
        resp_text = fn.match_programs_to_entrepreneur(entrepreneur, batch_json, model)
        try:
            parsed = fn.coerce_json_array(resp_text)
            # tag results with batch for debugging
            for m in parsed:
                m["batch_index"] = b_idx
            all_matches.extend(parsed)
        except Exception as e:
            st.error(f"Failed to generate recommendations: {e}")
            with st.expander(f"Raw model output (batch {b_idx})"):
                st.code((resp_text or "")[:4000])
        progress.progress(b_idx / float(num_batches))

    if not all_matches:
        st.error("No matches returned from any batch.")
        return

    # --------- Deduplicate & finalize scores ----------
    deduped, seen = [], set()
    for m in all_matches:
        key = (str(m.get("provider_id")), m.get("program_name"))
        if key in seen:
            continue
        seen.add(key)

        # ensure numeric scores and a final_score
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

    # --------- Summaries ----------
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

    # --------- Table ----------
    matches_df = pd.DataFrame(deduped).sort_values("final_score", ascending=False)

    # Remove debug/batch columns, backend source, and reorder for clarity
    matches_df = matches_df.drop(columns=["batch_index", "source"], errors="ignore")

    # If 'website' not already present, fill from payload (match by provider_id/program_name)
    if "website" not in matches_df.columns:
        matches_df["website"] = None
    for i, row_ in matches_df.iterrows():
        if not row_.get("website"):
            pid = str(row_.get("provider_id"))
            pname = str(row_.get("program_name", "")).lower()
            for prov in full_payload:
                if str(prov.get("provider_id")) == pid:
                    for prog in prov.get("programs", []):
                        if pname == str(prog.get("program_name", "")).lower():
                            matches_df.at[i, "website"] = prog.get("website")
                            break

    # Display
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

    # Debug preview of payload actually used
    with st.expander("Preview: first 3 providers from the full payload"):
        preview = []
        for p in full_payload[:3]:
            preview.append({
                "provider_name": p.get("provider_name"),
                "distance": p.get("distance"),
                "num_programs": len(p.get("programs", [])),
                "first_program": (p.get("programs", [{}])[0].get("program_name")
                                  if p.get("programs") else None),
            })
        st.json(preview)




def render_chat_tab(engine, model):
    """Chat with the programs/providers database ."""
    st.subheader("💬 Ask the Ecosystem")
    st.caption("Ask natural-language questions like: "
               "“Is there a service provider that can help with funding for my agrotech startup?”")

    # Chat history
    if "qa_msgs" not in st.session_state:
        st.session_state.qa_msgs = [
            {"role": "assistant",
             "content": "Hi! Ask what you need (funding, prototyping, mentorship, specific verticals, etc.)."}
        ]

    # Render history
    for m in st.session_state.qa_msgs:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Input
    user_q = st.chat_input("Type your question")
    if not user_q:
        return

    # Show user message
    st.session_state.qa_msgs.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    # Retrieve candidates from BOTH sources
    hits = fn.nl_search_programs(engine, user_q, limit=20)

    # Show retrieved rows for transparency
    if not hits.empty:
        with st.expander("Retrieved matches (top 20)"):
            st.dataframe(
                hits[["program_name","provider_name","services","verticals","product_type","county","website"]],
                use_container_width=True
            )

    # Let the model compose the answer using only those rows
    if hits.empty:
        answer = "I couldn’t find anything relevant in the catalog. Try different words (e.g., 'grant', 'loan', 'agriculture')."
    else:
        answer = fn.answer_query_over_catalog(model, user_q, hits)

    # Show assistant answer and store
    st.session_state.qa_msgs.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)



def main():
    st.title("SWPA Innovation Ecosystem")

    engine = get_db_engine()
    model = get_llm_model()

    about, overview, needs, programs, matching, ask = st.tabs(
        ["About", "Rolodex Overview", "Needs", "Programs", "Matching tool", "Ask the DB"]
    )
    with about:    render_about_tab()
    with overview: render_overview_tab(engine)
    with needs:    render_needs_tab(engine)
    with programs: render_programs_tab(engine)
    with matching: render_matching_tab(engine, model)
    with ask: render_chat_tab(engine, model)

if __name__ == "__main__":
    main()
