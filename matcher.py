import requests
import openai
import pandas as pd
from bs4 import BeautifulSoup
import time

openai.api_key = "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"

SUMMARY_CHAR_LIMIT = 400
RANK_RATIONALE_WORDS = 35

def fetch_text_from_url(url):
    try:
        res = requests.get(url, timeout=10, headers={"User-Agent": "RolodexScraper/2.0"})
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "iframe"]): tag.decompose()
        return soup.get_text(separator="\n", strip=True)[:3500]
    except Exception as e:
        print(f"❌ Failed to fetch {url}: {e}")
        return ""

def summarize_program(name, founder_needs, raw_text):
    prompt = f"""You are a research assistant summarizing entrepreneurship support programs.

Founder Needs:
{founder_needs}

Program Name: {name}
Website Text:
{raw_text}

Output: One 1–2 sentence summary (max {SUMMARY_CHAR_LIMIT} characters) explaining how this program helps founders like the one described."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You summarize startup support programs."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        text = response.choices[0].message.content.strip()
        print(f"🧾 GPT-3.5 Summary for {name}:{text}")
        return text[:SUMMARY_CHAR_LIMIT]
    except Exception as e:
        print(f"❌ Summary failed for {name}: {e}")
        return "(Summary not available)"

def score_program(name, founder_needs, summary):
    prompt = f"""
You are evaluating how well a startup support program fits the needs of a founder.

Founder Needs:
{founder_needs}

Program: {name}
Description: {summary}

Instructions:
- Relevance Score: 0–100
- Stage Fit: 0–100
- Overall Score: 0.55*relevance + 0.45*stage_fit
- Rationale: Brief explanation (<= {RANK_RATIONALE_WORDS} words)

Respond with JSON like:
{{"relevance": 75, "stage_fit": 60, "overall": 68.25, "rationale": "Helps with prototyping, early-stage funding."}}
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You score startup programs."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        text = response.choices[0].message.content.strip()
        print(f"📊 GPT-3.5 Score JSON for {name}:\n{text}")
        return eval(text) if text.startswith("{") else {
            "relevance": 0, "stage_fit": 0, "overall": 0, "rationale": "Invalid JSON"
        }
    except Exception as e:
        print(f"❌ Scoring failed for {name}: {e}")
        return {"relevance": 0, "stage_fit": 0, "overall": 0, "rationale": "Scoring failed."}


def summarize_and_score_programs(df, founder_needs):
    df = df.fillna("")
    summaries, scores, rationales = [], [], []

    for i, row in df.iterrows():
        name = row.get("Program Name") or f"Program {i+1}"
        url = row.get("Website") or row.get("url", "")
        print(f"🔄 Processing {i+1}/{len(df)}: {name}")

        if not url or not url.startswith("http"):
            print(f"⚠️ Skipping invalid URL: {url}")
            summaries.append("(No URL)")
            scores.append(0)
            rationales.append("Missing or invalid URL")
            continue

        text = fetch_text_from_url(url)
        summary = summarize_program(name, founder_needs, text)
        score = score_program(name, founder_needs, summary)

        summaries.append(summary)
        scores.append(score.get("overall", 0))
        rationales.append(score.get("rationale", ""))

        time.sleep(1.5)

    new_cols = pd.DataFrame({
        "GPT Summary": summaries,
        "Score": scores,
        "Rationale": rationales
    })

    return pd.concat([df.reset_index(drop=True), new_cols], axis=1)
