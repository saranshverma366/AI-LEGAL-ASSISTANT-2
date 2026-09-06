"""
AI Legal Assistant - all-in-one Flask application (India / BNS 2023)
======================================================================
Single-file app that combines:
  1. A BERT-based semantic search engine (sentence-transformers) that matches
     a user's plain-English question to the right legal-advice entry and BNS
     (Bharatiya Nyaya Sanhita) sections. Falls back automatically to a
     TF-IDF cosine-similarity matcher if sentence-transformers / the model
     weights are not available (e.g. no internet in this environment) so the
     app always works.
  2. A legal-advice + BNS-section + DLSA-locator database (loaded from
     legal_database.json, kept next to this file).
  3. An FIR (First Information Report) draft builder.
  4. The frontend (HTML/CSS/JS), served directly by Flask - no separate
     server or build step needed.

Run:
    pip install flask sentence-transformers scikit-learn --break-system-packages
    python app.py
Then open http://127.0.0.1:5000

NOTE: sentence-transformers needs to download a small BERT model
(~80MB, 'all-MiniLM-L6-v2') the first time you run it, so an internet
connection is required at least once. If you don't want that dependency,
just don't install sentence-transformers - the app will automatically use
the TF-IDF fallback matcher instead (no internet or GPU required).

DISCLAIMER: This tool gives general legal information only. It is not a
substitute for a licensed advocate. Section numbers/punishments must be
verified against the official Bare Act before relying on them for a real case.
"""

import json
import os
import datetime
from flask import Flask, request, jsonify, Response, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "legal_database.json")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# 1. LOAD THE DATABASE
# ---------------------------------------------------------------------------
with open(DB_PATH, "r", encoding="utf-8") as f:
    DB = json.load(f)

LEGAL_ADVICE = DB["legal_advice"]
BNS_SECTIONS = DB["bns_sections"]
DLSA = DB["dlsa_locator"]
BNS_BY_SECTION = {s["section"]: s for s in BNS_SECTIONS}

# Build the text corpus the model will search over: one "document" per
# legal-advice entry, made of its title + keywords + category (kept short,
# since that's what a user's query will resemble).
ADVICE_CORPUS = [
    f"{a['title']}. Category: {a['category']}. Keywords: {', '.join(a['keywords'])}"
    for a in LEGAL_ADVICE
]

# ---------------------------------------------------------------------------
# 2. THE "BERT" SEMANTIC MATCHER (with automatic offline fallback)
# ---------------------------------------------------------------------------
# We try to use a real BERT-family sentence embedding model
# (sentence-transformers/all-MiniLM-L6-v2 is a distilled BERT, purpose-built
# for exactly this kind of semantic similarity search and is the standard
# lightweight choice for this task). If that stack isn't installed / cannot
# reach the internet to download weights, we transparently fall back to a
# TF-IDF + cosine-similarity matcher (scikit-learn) so the assistant still
# works end-to-end.

MATCHER_BACKEND = "unavailable"
_model = None
_advice_embeddings = None
_tfidf_vectorizer = None
_tfidf_matrix = None


def _try_load_bert():
    global _model, _advice_embeddings, MATCHER_BACKEND
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")  # distilled BERT
        _advice_embeddings = _model.encode(ADVICE_CORPUS, convert_to_numpy=True)
        MATCHER_BACKEND = "bert"
        print("[legal-assistant] Loaded BERT (sentence-transformers) matcher.")
        return True
    except Exception as e:  # noqa: BLE001 - we want ANY failure to fall back
        print(f"[legal-assistant] BERT backend unavailable ({e}). Falling back to TF-IDF.")
        return False


def _load_tfidf_fallback():
    global _tfidf_vectorizer, _tfidf_matrix, MATCHER_BACKEND
    from sklearn.feature_extraction.text import TfidfVectorizer
    _tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    _tfidf_matrix = _tfidf_vectorizer.fit_transform(ADVICE_CORPUS)
    MATCHER_BACKEND = "tfidf-fallback"
    print("[legal-assistant] Loaded TF-IDF fallback matcher.")


if not _try_load_bert():
    _load_tfidf_fallback()


def semantic_search(query: str, top_k: int = 3):
    """Return the top_k legal-advice entries most similar to the query,
    each with a 0-1 similarity score, using whichever backend is active."""
    query = (query or "").strip()
    if not query:
        return []

    if MATCHER_BACKEND == "bert":
        import numpy as np
        q_emb = _model.encode([query], convert_to_numpy=True)[0]
        # cosine similarity
        sims = _advice_embeddings @ q_emb / (
            (np.linalg.norm(_advice_embeddings, axis=1) * np.linalg.norm(q_emb)) + 1e-8
        )
        order = sims.argsort()[::-1][:top_k]
        return [(LEGAL_ADVICE[i], float(sims[i])) for i in order]

    else:  # tfidf-fallback
        from sklearn.metrics.pairwise import cosine_similarity
        q_vec = _tfidf_vectorizer.transform([query])
        sims = cosine_similarity(q_vec, _tfidf_matrix)[0]
        order = sims.argsort()[::-1][:top_k]
        return [(LEGAL_ADVICE[i], float(sims[i])) for i in order]


# ---------------------------------------------------------------------------
# 3. FIR (First Information Report) DRAFT BUILDER
# ---------------------------------------------------------------------------
def build_fir_draft(data: dict) -> str:
    """Generate a plain-text FIR draft from structured form data.
    This is a DRAFT to help the complainant organise facts before visiting
    the police station - it is not an official FIR and does not replace the
    officer-in-charge's own recording of the complaint under Section 173
    BNSS (2023)."""

    def g(key, default="Not provided"):
        val = data.get(key)
        return val.strip() if isinstance(val, str) and val.strip() else default

    suggested_sections = data.get("suggested_bns_sections") or []
    sections_text = ", ".join(
        f"BNS Section {s}" + (f" ({BNS_BY_SECTION[s]['title']})" if s in BNS_BY_SECTION else "")
        for s in suggested_sections
    ) or "To be determined by the Investigating Officer"

    today = datetime.date.today().strftime("%d-%m-%Y")

    draft = f"""
DRAFT - FIRST INFORMATION REPORT (FOR PERSONAL REFERENCE ONLY)
================================================================
This is a self-prepared draft to help you present a clear, complete
complaint at the police station. It is NOT an official FIR. The police
officer-in-charge will record the actual FIR in the prescribed format.

Date of drafting this complaint: {today}
Police Station (jurisdiction): {g('police_station')}
District / State: {g('district')}, {g('state')}

1. COMPLAINANT DETAILS
-----------------------
Name              : {g('complainant_name')}
Father's/Husband's Name : {g('complainant_relative_name')}
Age               : {g('complainant_age')}
Gender            : {g('complainant_gender')}
Address           : {g('complainant_address')}
Phone Number      : {g('complainant_phone')}
ID Proof Type/No. : {g('complainant_id')}

2. INCIDENT DETAILS
--------------------
Date of Incident  : {g('incident_date')}
Time of Incident  : {g('incident_time')}
Place of Incident : {g('incident_place')}

Description of the Incident (facts, in chronological order):
{g('incident_description', '[Describe exactly what happened, in the order it happened. Stick to facts you personally witnessed or can support with evidence.]')}

3. ACCUSED / OPPOSITE PARTY DETAILS (if known)
------------------------------------------------
Name(s)           : {g('accused_name')}
Description/Address (if known) : {g('accused_description')}
Relationship to Complainant (if any) : {g('accused_relationship')}

4. PROPERTY / INJURY DETAILS (if applicable)
-----------------------------------------------
Property Lost/Damaged (description & approx. value): {g('property_details')}
Injuries Sustained (if any) : {g('injury_details')}

5. WITNESSES (if any)
-----------------------
{g('witness_details', '[Name, address and phone number of anyone who saw or has knowledge of the incident]')}

6. EVIDENCE AVAILABLE
------------------------
{g('evidence_details', '[e.g. photos, CCTV, medical certificate, messages, documents, receipts]')}

7. RELIEF / ACTION REQUESTED
--------------------------------
I request that this complaint be registered as an FIR and appropriate
legal action be taken against the persons responsible under the relevant
provisions of law, including but not limited to: {sections_text}

8. DECLARATION
------------------
I declare that the information given above is true to the best of my
knowledge and belief. I am willing to cooperate with the investigation
and provide any further information/evidence required.

Signature of Complainant : ______________________
Date                     : {today}

------------------------------------------------------------------------
NEXT STEPS:
  - Take 2 copies of this draft to the police station named above.
  - Under Section 173 BNSS (2023), the police MUST register an FIR for a
    cognizable offence - refusal can be reported to the Superintendent of
    Police [Sec. 173(4) BNSS] or the jurisdictional Magistrate [Sec. 175(3)
    BNSS].
  - Always ask for and keep a free copy of the FIR once registered.
  - For free legal assistance, call the NALSA helpline: 15100.
------------------------------------------------------------------------
""".strip("\n")
    return draft


# ---------------------------------------------------------------------------
# 4. API ROUTES
# ---------------------------------------------------------------------------
@app.route("/api/status")
def api_status():
    return jsonify({
        "status": "ok",
        "matcher_backend": MATCHER_BACKEND,
        "advice_entries": len(LEGAL_ADVICE),
        "bns_sections": len(BNS_SECTIONS),
    })


@app.route("/api/ask", methods=["POST"])
def api_ask():
    payload = request.get_json(force=True, silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    query = payload.get("query", "")
    if not isinstance(query, str) or not query.strip():
        return jsonify({"error": "Please provide a 'query' describing your legal issue."}), 400

    matches = semantic_search(query, top_k=3)
    results = []
    for advice, score in matches:
        bns_details = [BNS_BY_SECTION[s] for s in advice.get("related_bns", []) if s in BNS_BY_SECTION]
        results.append({
            "id": advice["id"],
            "category": advice["category"],
            "title": advice["title"],
            "advice": advice["advice"],
            "confidence": round(score, 3),
            "related_bns_sections": bns_details,
        })

    return jsonify({
        "query": query,
        "matcher_backend": MATCHER_BACKEND,
        "results": results,
        "disclaimer": DB["meta"]["disclaimer"],
    })


@app.route("/api/bns", methods=["GET"])
def api_bns_list():
    """List all BNS sections, or search them with ?q=keyword"""
    q = request.args.get("q", "").strip().lower()
    if not q:
        return jsonify({"count": len(BNS_SECTIONS), "sections": BNS_SECTIONS})
    filtered = [
        s for s in BNS_SECTIONS
        if q in s["section"].lower() or q in s["title"].lower()
        or q in s["description"].lower() or q in str(s.get("old_ipc", "")).lower()
    ]
    return jsonify({"count": len(filtered), "sections": filtered})


@app.route("/api/bns/<section>", methods=["GET"])
def api_bns_detail(section):
    s = BNS_BY_SECTION.get(section)
    if not s:
        return jsonify({"error": f"BNS Section {section} not found in this database."}), 404
    return jsonify(s)


@app.route("/api/dlsa", methods=["GET"])
def api_dlsa():
    return jsonify(DLSA)


@app.route("/api/fir/generate", methods=["POST"])
def api_fir_generate():
    payload = request.get_json(force=True, silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    draft = build_fir_draft(payload)
    return jsonify({"fir_draft": draft})


@app.route("/api/fir/download", methods=["POST"])
def api_fir_download():
    payload = request.get_json(force=True, silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    draft = build_fir_draft(payload)
    filename = f"FIR_Draft_{datetime.date.today().strftime('%Y%m%d')}.txt"
    return Response(
        draft,
        mimetype="text/plain",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ---------------------------------------------------------------------------
# 5. FRONTEND (embedded HTML/CSS/JS, served at "/")
# ---------------------------------------------------------------------------
FRONTEND_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Legal Assistant - India (BNS 2023)</title>
<style>
  :root{
    --navy:#0f2942; --navy-2:#16354f; --gold:#c9a24b; --gold-2:#e0be6f;
    --bg:#f4f1ea; --card:#ffffff; --ink:#1c2b3a; --muted:#5c6b7a; --border:#e2dfd6;
    --danger:#a83232; --radius:10px;
  }
  *{box-sizing:border-box;}
  body{margin:0;font-family:"Georgia","Times New Roman",serif;background:var(--bg);color:var(--ink);}
  header{background:linear-gradient(135deg,var(--navy),var(--navy-2));color:#fff;padding:22px 28px;
         display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;}
  header h1{margin:0;font-size:22px;letter-spacing:.5px;}
  header .scale{font-size:26px;margin-right:10px;}
  header .badge{font-family:Arial,sans-serif;font-size:11px;background:rgba(255,255,255,.12);
                 padding:5px 10px;border-radius:20px;border:1px solid rgba(255,255,255,.25);}
  nav{display:flex;gap:6px;background:#fff;border-bottom:2px solid var(--gold);padding:0 20px;
      overflow-x:auto;}
  nav button{font-family:Arial,sans-serif;border:none;background:none;padding:14px 16px;
             font-size:14px;cursor:pointer;color:var(--muted);border-bottom:3px solid transparent;
             white-space:nowrap;}
  nav button.active{color:var(--navy);border-bottom-color:var(--gold);font-weight:bold;}
  main{max-width:900px;margin:0 auto;padding:24px 18px 60px;}
  .tab{display:none;} .tab.active{display:block;animation:fade .25s ease;}
  @keyframes fade{from{opacity:0;transform:translateY(4px);}to{opacity:1;transform:none;}}
  .card{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);
        padding:20px;margin-bottom:16px;box-shadow:0 1px 3px rgba(15,41,66,.06);}
  h2{font-size:19px;color:var(--navy);margin-top:0;border-bottom:1px solid var(--border);padding-bottom:8px;}
  label{font-family:Arial,sans-serif;font-size:12.5px;color:var(--muted);display:block;margin:10px 0 4px;}
  input,textarea,select{width:100%;padding:9px 11px;border:1px solid var(--border);border-radius:6px;
        font-family:Arial,sans-serif;font-size:14px;background:#fcfbf8;}
  textarea{min-height:80px;resize:vertical;}
  .row{display:grid;grid-template-columns:1fr 1fr;gap:12px;}
  @media(max-width:600px){.row{grid-template-columns:1fr;}}
  button.primary{font-family:Arial,sans-serif;background:var(--navy);color:#fff;border:none;
      padding:11px 20px;border-radius:6px;cursor:pointer;font-size:14.5px;font-weight:bold;
      margin-top:14px;}
  button.primary:hover{background:var(--navy-2);}
  button.secondary{font-family:Arial,sans-serif;background:#fff;color:var(--navy);
      border:1px solid var(--navy);padding:9px 16px;border-radius:6px;cursor:pointer;font-size:13.5px;
      margin-top:8px;margin-right:8px;}
  .result{border-left:4px solid var(--gold);padding:12px 14px;margin-top:12px;background:#fbf8f1;
          border-radius:0 6px 6px 0;}
  .result h3{margin:0 0 4px;font-size:16px;color:var(--navy);}
  .conf{font-family:Arial,sans-serif;font-size:11px;color:#fff;background:var(--navy);
        padding:2px 8px;border-radius:10px;margin-left:8px;}
  .sec-chip{display:inline-block;font-family:Arial,sans-serif;font-size:11.5px;background:var(--navy);
            color:#fff;padding:3px 9px;border-radius:12px;margin:3px 4px 0 0;cursor:pointer;}
  .disclaimer{font-family:Arial,sans-serif;font-size:12px;color:var(--muted);background:#fff9ec;
              border:1px dashed var(--gold);padding:10px 12px;border-radius:6px;margin-top:16px;}
  .helpline{display:flex;justify-content:space-between;padding:9px 0;border-bottom:1px solid var(--border);
            font-family:Arial,sans-serif;font-size:14px;}
  .helpline b{color:var(--navy);}
  pre.fir{white-space:pre-wrap;font-family:"Courier New",monospace;font-size:12.5px;
          background:#fcfbf8;border:1px solid var(--border);padding:16px;border-radius:6px;
          max-height:480px;overflow:auto;}
  .muted{color:var(--muted);font-family:Arial,sans-serif;font-size:13px;}
  .search-row{display:flex;gap:8px;}
  .search-row input{flex:1;}
  .bns-item{padding:10px 0;border-bottom:1px solid var(--border);font-family:Arial,sans-serif;}
  .bns-item .num{color:var(--gold-2);background:var(--navy);display:inline-block;padding:2px 8px;
                 border-radius:5px;font-size:12px;margin-right:8px;}
  footer{text-align:center;font-family:Arial,sans-serif;font-size:11.5px;color:var(--muted);padding:20px;}
  .spinner{display:inline-block;width:14px;height:14px;border:2px solid #cbd5df;border-top-color:var(--navy);
           border-radius:50%;animation:spin .7s linear infinite;vertical-align:middle;margin-right:6px;}
  @keyframes spin{to{transform:rotate(360deg);}}
</style>
</head>
<body>

<header>
  <div style="display:flex;align-items:center;">
    <span class="scale">&#9878;</span>
    <div>
      <h1>AI Legal Assistant</h1>
      <div style="font-family:Arial,sans-serif;font-size:12px;opacity:.85;">Bharatiya Nyaya Sanhita (BNS) 2023 &middot; India</div>
    </div>
  </div>
  <span class="badge" id="backendBadge">loading model...</span>
</header>

<nav>
  <button class="tab-btn active" data-tab="ask">Ask a Legal Question</button>
  <button class="tab-btn" data-tab="bns">BNS Sections</button>
  <button class="tab-btn" data-tab="dlsa">DLSA / Free Legal Aid</button>
  <button class="tab-btn" data-tab="fir">FIR Builder</button>
</nav>

<main>

  <!-- ASK TAB -->
  <section class="tab active" id="tab-ask">
    <div class="card">
      <h2>Describe your legal issue in plain English</h2>
      <textarea id="askInput" placeholder="e.g. My phone was snatched on the street yesterday and I don't know the person's name..."></textarea>
      <button class="primary" onclick="askQuestion()">Get Legal Guidance</button>
      <div id="askResults"></div>
    </div>
  </section>

  <!-- BNS TAB -->
  <section class="tab" id="tab-bns">
    <div class="card">
      <h2>Search Bharatiya Nyaya Sanhita (BNS) Sections</h2>
      <div class="search-row">
        <input id="bnsSearch" placeholder="Search by section number, offence, or old IPC number (e.g. 'theft', '420', '303')">
        <button class="secondary" onclick="searchBns()">Search</button>
      </div>
      <div id="bnsResults"></div>
    </div>
  </section>

  <!-- DLSA TAB -->
  <section class="tab" id="tab-dlsa">
    <div class="card">
      <h2>Free Legal Aid Helplines</h2>
      <div id="helplines">Loading...</div>
    </div>
    <div class="card">
      <h2>District Legal Services Authority (DLSA) Locator</h2>
      <div id="dlsaInfo">Loading...</div>
    </div>
  </section>

  <!-- FIR BUILDER TAB -->
  <section class="tab" id="tab-fir">
    <div class="card">
      <h2>FIR (First Information Report) Draft Builder</h2>
      <p class="muted">This creates a personal draft to help you organise the facts before visiting the police station. It is not an official FIR.</p>

      <label>Police Station</label><input id="fPS">
      <div class="row">
        <div><label>District</label><input id="fDistrict"></div>
        <div><label>State</label><input id="fState"></div>
      </div>

      <h2 style="margin-top:20px;">Complainant Details</h2>
      <div class="row">
        <div><label>Full Name</label><input id="fName"></div>
        <div><label>Father's / Husband's Name</label><input id="fRelName"></div>
      </div>
      <div class="row">
        <div><label>Age</label><input id="fAge"></div>
        <div><label>Gender</label>
          <select id="fGender"><option>Female</option><option>Male</option><option>Other</option></select>
        </div>
      </div>
      <label>Address</label><textarea id="fAddress"></textarea>
      <div class="row">
        <div><label>Phone Number</label><input id="fPhone"></div>
        <div><label>ID Proof (type &amp; number)</label><input id="fId"></div>
      </div>

      <h2 style="margin-top:20px;">Incident Details</h2>
      <div class="row">
        <div><label>Date of Incident</label><input id="fDate" type="date"></div>
        <div><label>Time of Incident</label><input id="fTime" type="time"></div>
      </div>
      <label>Place of Incident</label><input id="fPlace">
      <label>Describe what happened (facts only, in order)</label>
      <textarea id="fDesc" style="min-height:110px;"></textarea>

      <h2 style="margin-top:20px;">Accused (if known)</h2>
      <div class="row">
        <div><label>Name(s)</label><input id="fAccusedName"></div>
        <div><label>Relationship to you (if any)</label><input id="fAccusedRel"></div>
      </div>
      <label>Description / Address of accused (if known)</label><input id="fAccusedDesc">

      <h2 style="margin-top:20px;">Property / Injury / Evidence</h2>
      <label>Property lost or damaged (description &amp; value)</label><input id="fProperty">
      <label>Injuries sustained (if any)</label><input id="fInjury">
      <label>Witnesses (name, address, phone)</label><textarea id="fWitness"></textarea>
      <label>Evidence available (photos, CCTV, medical report, etc.)</label><textarea id="fEvidence"></textarea>

      <label>Suggested BNS sections to cite (comma-separated, optional — the Ask tab can suggest these)</label>
      <input id="fSections" placeholder="e.g. 303, 351">

      <div>
        <button class="primary" onclick="generateFir()">Generate FIR Draft</button>
        <button class="secondary" onclick="downloadFir()">Download as .txt</button>
      </div>
      <div id="firOutput"></div>
    </div>
  </section>

  <div class="disclaimer">
    <b>Disclaimer:</b> This assistant provides general legal information for educational purposes only.
    It is not a substitute for advice from a licensed advocate. Always verify section numbers and
    procedure with a qualified lawyer or the official Bare Act. For free legal aid, dial <b>15100</b>.
  </div>
</main>

<footer>AI Legal Assistant &middot; Built on the Bharatiya Nyaya Sanhita, 2023 &middot; Not a law firm, not legal advice.</footer>

<script>
const API = "";

// ---- Tab switching ----
document.querySelectorAll(".tab-btn").forEach(btn=>{
  btn.addEventListener("click", ()=>{
    document.querySelectorAll(".tab-btn").forEach(b=>b.classList.remove("active"));
    document.querySelectorAll(".tab").forEach(t=>t.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("tab-"+btn.dataset.tab).classList.add("active");
  });
});

// ---- Status badge ----
fetch(API+"/api/status").then(r=>r.json()).then(d=>{
  const badge = document.getElementById("backendBadge");
  badge.textContent = d.matcher_backend === "bert" ? "BERT model active" : "TF-IDF fallback active";
}).catch(()=>{ document.getElementById("backendBadge").textContent = "offline"; });

// ---- Ask tab ----
async function askQuestion(){
  const q = document.getElementById("askInput").value;
  const box = document.getElementById("askResults");
  if(!q.trim()){ box.innerHTML = "<p class='muted'>Please describe your issue first.</p>"; return; }
  box.innerHTML = "<p class='muted'><span class='spinner'></span>Analysing your question...</p>";
  try{
    const res = await fetch(API+"/api/ask", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({query:q})});
    const data = await res.json();
    if(data.error){ box.innerHTML = "<p class='muted'>"+data.error+"</p>"; return; }
    if(!data.results.length){ box.innerHTML = "<p class='muted'>No matching guidance found. Try rephrasing, or call 15100 for direct help.</p>"; return; }
    box.innerHTML = data.results.map(r=>{
      const secs = (r.related_bns_sections||[]).map(s=>
        `<span class="sec-chip" onclick="prefillFirSections('${s.section}')" title="${s.title}">BNS ${s.section} — ${s.title}</span>`
      ).join("");
      return `<div class="result">
        <h3>${r.category} <span class="conf">${Math.round(r.confidence*100)}% match</span></h3>
        <div style="white-space:pre-line;">${r.advice}</div>
        <div style="margin-top:8px;">${secs}</div>
      </div>`;
    }).join("");
  }catch(e){
    box.innerHTML = "<p class='muted'>Something went wrong reaching the server.</p>";
  }
}

function prefillFirSections(section){
  const el = document.getElementById("fSections");
  const existing = el.value.split(",").map(s=>s.trim()).filter(Boolean);
  if(!existing.includes(section)) existing.push(section);
  el.value = existing.join(", ");
  document.querySelectorAll(".tab-btn").forEach(b=>b.classList.remove("active"));
  document.querySelectorAll(".tab").forEach(t=>t.classList.remove("active"));
  document.querySelector('[data-tab="fir"]').classList.add("active");
  document.getElementById("tab-fir").classList.add("active");
}

// ---- BNS tab ----
async function searchBns(){
  const q = document.getElementById("bnsSearch").value;
  const box = document.getElementById("bnsResults");
  box.innerHTML = "<p class='muted'>Searching...</p>";
  const res = await fetch(API+"/api/bns?q="+encodeURIComponent(q));
  const data = await res.json();
  renderBns(data.sections);
}
function renderBns(sections){
  const box = document.getElementById("bnsResults");
  if(!sections.length){ box.innerHTML="<p class='muted'>No sections found.</p>"; return; }
  box.innerHTML = sections.map(s=>`
    <div class="bns-item">
      <span class="num">Sec ${s.section}</span><b>${s.title}</b>
      <span class="muted"> (old IPC ${s.old_ipc})</span>
      <div class="muted" style="margin-top:4px;">${s.description}</div>
      <div style="margin-top:4px;"><b>Punishment:</b> ${s.punishment}</div>
    </div>`).join("");
}
fetch(API+"/api/bns").then(r=>r.json()).then(d=>renderBns(d.sections.slice(0,8)));

// ---- DLSA tab ----
fetch(API+"/api/dlsa").then(r=>r.json()).then(d=>{
  document.getElementById("helplines").innerHTML = d.national_helplines.map(h=>
    `<div class="helpline"><span>${h.name} <span class="muted">(${h.notes})</span></span><b>${h.number}</b></div>`
  ).join("");
  document.getElementById("dlsaInfo").innerHTML = `
    <p class="muted">${d.structure_note}</p>
    <p><b>How to apply for free legal aid:</b></p>
    <ol class="muted">${d.how_to_apply.map(x=>`<li>${x}</li>`).join("")}</ol>
    <p><b>Verified sample entry</b></p>
    <div class="bns-item">
      <b>${d.sample_verified_entry.authority}</b><br>
      <span class="muted">${d.sample_verified_entry.address}</span><br>
      <span class="muted">Toll-free: ${d.sample_verified_entry.toll_free}</span>
    </div>
    <p class="muted" style="margin-top:8px;">${d.note_on_full_coverage}</p>
  `;
});

// ---- FIR builder ----
function collectFirForm(){
  const sectionsRaw = document.getElementById("fSections").value;
  const sections = sectionsRaw.split(",").map(s=>s.trim()).filter(Boolean);
  return {
    police_station: document.getElementById("fPS").value,
    district: document.getElementById("fDistrict").value,
    state: document.getElementById("fState").value,
    complainant_name: document.getElementById("fName").value,
    complainant_relative_name: document.getElementById("fRelName").value,
    complainant_age: document.getElementById("fAge").value,
    complainant_gender: document.getElementById("fGender").value,
    complainant_address: document.getElementById("fAddress").value,
    complainant_phone: document.getElementById("fPhone").value,
    complainant_id: document.getElementById("fId").value,
    incident_date: document.getElementById("fDate").value,
    incident_time: document.getElementById("fTime").value,
    incident_place: document.getElementById("fPlace").value,
    incident_description: document.getElementById("fDesc").value,
    accused_name: document.getElementById("fAccusedName").value,
    accused_relationship: document.getElementById("fAccusedRel").value,
    accused_description: document.getElementById("fAccusedDesc").value,
    property_details: document.getElementById("fProperty").value,
    injury_details: document.getElementById("fInjury").value,
    witness_details: document.getElementById("fWitness").value,
    evidence_details: document.getElementById("fEvidence").value,
    suggested_bns_sections: sections,
  };
}

async function generateFir(){
  const box = document.getElementById("firOutput");
  box.innerHTML = "<p class='muted'><span class='spinner'></span>Generating draft...</p>";
  const res = await fetch(API+"/api/fir/generate", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(collectFirForm())});
  const data = await res.json();
  box.innerHTML = `<pre class="fir">${data.fir_draft.replace(/</g,"&lt;")}</pre>`;
}

async function downloadFir(){
  const res = await fetch(API+"/api/fir/download", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(collectFirForm())});
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = "FIR_Draft.txt"; a.click();
  URL.revokeObjectURL(url);
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "frontend.html")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"[legal-assistant] Matcher backend in use: {MATCHER_BACKEND}")
    app.run(host="0.0.0.0", port=5000, debug=True)
