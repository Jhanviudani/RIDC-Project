# search_programs.py
import os
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI

load_dotenv()
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
client   = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

need_text = "Robotics hardware founder seeking prototyping lab access and early grant guidance in medtech"

q_emb = client.embeddings.create(model=EMBED_MODEL, input=need_text).data[0].embedding

# Supabase doesn’t accept vector literals in RPC by default; do a client-side ranking:
rows = supabase.table("program_parsed").select("id, program_name, organization, summary_gpt,embedding").execute().data

import numpy as np
qe = np.array(q_emb, dtype=float)

def cos(a, b):
    a, b = np.array(a), np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)

scored = []
for r in rows:
    e = r.get("embedding")
    if not e:
        continue
    scored.append((cos(qe, e), r))

scored.sort(reverse=True)  # high cosine = closer
for s, r in scored[:15]:
    print(f"{s:.3f}  {r['program_name']} — {r.get('provider_name')}")
